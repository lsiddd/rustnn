//! Treino: o estado que atravessa um passo, o passo em si e os lacos que o
//! chamam.
//!
//! O passo do rustvit existe uma unica vez, em [`Trainer::step`]. O laco de
//! epocas, o `--bench` e o `--fingerprint` chamam esse mesmo codigo: um
//! fingerprint que validasse uma copia do passo nao validaria nada.

use crate::cli::Config;
use crate::data::{self, Aug, IMG, NCLASS, PIX};
use crate::eval;
use crate::model::ResNet;
use crate::nn::{softmax_ce, Param, Tensor};
use crate::rng::Rng;
use crate::vit::{Cfg, Grads, Out, RustViT};
use crate::{loss, optim};
use std::sync::Arc;
use std::time::Instant;

/// Warmup linear, cosseno ate `lr_min`.
pub fn lr_at(peak: f32, lo: f32, step: usize, total: usize, warm: usize) -> f32 {
    if warm > 0 && step < warm {
        lo + (peak - lo) * (step + 1) as f32 / warm as f32
    } else {
        let t = (step - warm) as f32 / (total - warm).max(1) as f32;
        lo + 0.5 * (peak - lo) * (1.0 + (std::f32::consts::PI * t).cos())
    }
}

/// Curriculo de augmentation.
///
/// Comecar em forca total desperdica a fase inicial de um treino curto, e
/// terminar em forca total impede o modelo de ajustar a distribuicao limpa. A
/// rampa vai ate 40% do treino e o corte comeca em 90%.
pub fn aug_at(cfg: &Config, epoch: usize) -> (Aug, f32, f32) {
    let f = epoch as f32 / cfg.epochs.max(1) as f32;
    let (s, mix) = if !cfg.curriculum {
        (1.0, 1.0)
    } else if f < 0.10 {
        (0.0, 0.0)
    } else if f < 0.40 {
        (((f - 0.10) / 0.30).min(1.0), ((f - 0.10) / 0.30).min(1.0))
    } else if f < 0.90 {
        (1.0, 1.0)
    } else {
        (0.3, 0.0)
    };
    (
        Aug {
            pad: cfg.pad,
            ra_mag: cfg.randaug * s,
            ra_n: cfg.ra_n,
            erase: cfg.erase * s,
            erase_side: cfg.cutout,
        },
        cfg.mixup * mix,
        cfg.cutmix * mix,
    )
}

// ---------------------------------------------------------------- perda

pub struct LossOut {
    pub g: Grads,
    pub total: f32,
    pub ce: f32,
    pub correct: usize,
}

pub fn build_loss(
    out: &Out,
    tf: &[f32],
    tc: &[f32],
    hard: &[u32],
    hardc: &[u32],
    cfg: &Config,
    nf: usize,
    nc: usize,
    n: usize,
) -> LossOut {
    let mut g = Grads {
        logits: vec![0.0; n * nf],
        coarse: vec![0.0; n * nc],
        ..Default::default()
    };
    let (ce, correct) = loss::ce_soft(&out.logits, tf, hard, nf, 1.0, &mut g.logits);
    let mut total = ce;
    if cfg.coarse_w > 0.0 && !cfg.no_taxo {
        let (lc, _) = loss::ce_soft(&out.coarse, tc, hardc, nc, cfg.coarse_w, &mut g.coarse);
        total += cfg.coarse_w * lc;
    }
    if !out.e1.is_empty() {
        let d4 = out.feat.len() / n;
        g.e1 = vec![0.0; n * nf];
        g.e2 = vec![0.0; n * nf];
        g.h1 = vec![0.0; n * d4];
        g.h2 = vec![0.0; n * d4];
        let (l1, _) = loss::ce_soft(&out.e1, tf, hard, nf, cfg.exit_w.0, &mut g.e1);
        let (l2, _) = loss::ce_soft(&out.e2, tf, hard, nf, cfg.exit_w.1, &mut g.e2);
        total += cfg.exit_w.0 * l1 + cfg.exit_w.1 * l2;
        if cfg.alpha > 0.0 {
            // professor destacado do grafo: nada volta por out.logits
            total += cfg.alpha
                * (loss::kl_distill(&out.e1, &out.logits, nf, cfg.temp, cfg.alpha, &mut g.e1)
                    + loss::kl_distill(&out.e2, &out.logits, nf, cfg.temp, cfg.alpha, &mut g.e2));
        }
        if cfg.beta > 0.0 {
            total += loss::hint_mse(&out.h1, &out.feat, n, cfg.beta, &mut g.h1)
                + loss::hint_mse(&out.h2, &out.feat, n, cfg.beta, &mut g.h2);
        }
    }
    LossOut {
        g,
        total,
        ce,
        correct,
    }
}

/// Alvos suaves do lote, ja com a mistura do mixup/cutmix aplicada.
pub fn targets(
    labels: &[u32],
    coarse: &[u32],
    perm: &[usize],
    lam: &[f32],
    taxo: &data::Taxo,
    cfg: &Config,
    tf: &mut [f32],
    tc: &mut [f32],
) {
    let (nf, nc) = (NCLASS, taxo.ncoarse);
    let mut a = vec![0.0f32; nf];
    let mut b = vec![0.0f32; nf];
    let mut ca = vec![0.0f32; nc];
    let mut cb = vec![0.0f32; nc];
    for i in 0..labels.len() {
        let (ya, yb) = (labels[i] as usize, labels[perm[i]] as usize);
        if cfg.no_taxo {
            loss::uniform_target(ya, cfg.smooth, &mut a);
            loss::uniform_target(yb, cfg.smooth, &mut b);
        } else {
            loss::taxo_target(ya, &taxo.groups, &taxo.parent, cfg.smooth, cfg.sib, &mut a);
            loss::taxo_target(yb, &taxo.groups, &taxo.parent, cfg.smooth, cfg.sib, &mut b);
        }
        loss::uniform_target(coarse[i] as usize, cfg.smooth, &mut ca);
        loss::uniform_target(coarse[perm[i]] as usize, cfg.smooth, &mut cb);
        let l = lam[i];
        for j in 0..nf {
            tf[i * nf + j] = l * a[j] + (1.0 - l) * b[j];
        }
        for j in 0..nc {
            tc[i * nc + j] = l * ca[j] + (1.0 - l) * cb[j];
        }
    }
}
/// Um checkpoint que nao grava e uma falha silenciosa: o treino segue horas e
/// so no fim se descobre que nao ha pesos. Reportar e continuar, porque perder
/// a epoca ainda e melhor que perder o treino.
pub fn report_save(path: &str, result: std::io::Result<()>) {
    if let Err(e) = result {
        eprintln!("\naviso: falha ao salvar checkpoint em '{path}': {e}");
    }
}

pub fn fmt_hms(secs: f64) -> String {
    let s = secs as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

pub fn make_cfg(c: &Config) -> Cfg {
    Cfg {
        width: c.width,
        cmid: c.cmid,
        dim: c.dim,
        heads: c.heads,
        depth3: c.depth3,
        dim4: c.dim4,
        heads4: c.heads4,
        depth4: c.depth4,
        mem_at: if c.no_mem {
            Vec::new()
        } else {
            vec![
                1.min(c.depth3.saturating_sub(1)),
                3.min(c.depth3.saturating_sub(1)),
            ]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
        },
        droppath: c.droppath,
        img: IMG,
    }
}

// ---------------------------------------------------------------- passo

/// Referencias mutaveis para todos os parametros de uma vez.
///
/// O `Vec` e o que `clip_`, `step_all` e a EMA consomem, e nenhum deles aceita
/// um visitante. Este e o unico ponto do treino que precisa de `unsafe`.
///
/// # Safety
/// `for_each_param` visita cada parametro exatamente uma vez, entao as
/// referencias levantadas aqui sao duas a duas disjuntas. O `Vec` resultante
/// nao pode sobreviver ao emprestimo de `net`, o que a assinatura garante.
fn params_of(net: &mut RustViT) -> Vec<&mut Param> {
    let mut ps: Vec<&mut Param> = Vec::new();
    net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
    ps
}

/// O que um passo devolve ao laco que o chamou.
pub struct StepOut {
    pub ce: f32,
    pub total: f32,
    pub correct: usize,
    pub lr: f32,
}

/// Modelo, otimizador e todos os buffers de um passo de treino do rustvit.
pub struct Trainer<'a> {
    pub net: RustViT,
    pub opt: optim::Optim,
    pub ema: Option<optim::Ema>,
    pub ema_decay: f32,
    pub cfg: &'a Config,
    pub taxo: &'a data::Taxo,
    pub train: &'a data::Dataset,
    rng: Rng,
    order: Vec<usize>,
    pub x: Arc<Tensor>,
    pub labels: Vec<u32>,
    dom: Vec<u32>,
    pub coarse: Vec<u32>,
    pub tf: Vec<f32>,
    pub tc: Vec<f32>,
    pub steps_per_epoch: usize,
    pub total_steps: usize,
    warm_steps: usize,
    pub step: usize,
}

impl<'a> Trainer<'a> {
    pub fn new(cfg: &'a Config, train: &'a data::Dataset, taxo: &'a data::Taxo, ntrain: usize) -> Self {
        let mut net = RustViT::new(make_cfg(cfg), taxo.parent.clone(), taxo.ncoarse, cfg.seed);
        net.use_exits = !cfg.no_exits;
        let steps_per_epoch = ntrain / cfg.batch;
        let total_steps = steps_per_epoch * cfg.epochs;
        // A janela da media e 1/(1-decay) passos. Um decay fixo de 0.9998 da
        // 5000 passos, que num treino de 300 epocas e um decimo do total e num
        // de 100 e um quarto: a media fica sempre atrasada em relacao a um
        // modelo que ainda melhora meio ponto por epoca. Derivar o decay do
        // orcamento de passos mantem a janela em 10% do treino, seja ele qual
        // for.
        let ema_decay = if cfg.ema_auto && total_steps > 0 {
            (1.0 - 10.0 / total_steps as f32).min(cfg.ema)
        } else {
            cfg.ema
        };
        Trainer {
            net,
            opt: optim::Optim::new(cfg.muon, cfg.ns),
            ema: (ema_decay > 0.0).then(|| optim::Ema::new(ema_decay)),
            ema_decay,
            cfg,
            taxo,
            train,
            rng: Rng::new(cfg.seed ^ 0xABCD),
            order: (0..ntrain).collect(),
            x: Arc::new(Tensor::zeros(cfg.batch, 3, IMG, IMG)),
            labels: vec![0u32; cfg.batch],
            dom: vec![0u32; cfg.batch],
            coarse: vec![0u32; cfg.batch],
            tf: vec![0.0f32; cfg.batch * NCLASS],
            tc: vec![0.0f32; cfg.batch * taxo.ncoarse],
            steps_per_epoch,
            total_steps,
            warm_steps: steps_per_epoch * cfg.warmup,
            step: 0,
        }
    }

    pub fn shuffle(&mut self) {
        self.rng.shuffle(&mut self.order);
    }

    /// Monta o lote `b` da epoca, deixando `x`, `labels`, `coarse`, `tf` e `tc`
    /// prontos. Devolve a permutacao e os lambdas da mistura.
    fn make_batch(&mut self, b: usize, aug: &Aug, amix: f32, acut: f32) -> (Vec<usize>, Vec<f32>) {
        let batch = self.cfg.batch;
        let (perm, lam) = {
            let idx = &self.order[b * batch..(b + 1) * batch];
            let seed = self.rng.next_u64();
            let xm = Arc::make_mut(&mut self.x);
            data::make_batch_train(
                self.train,
                idx,
                aug,
                seed,
                &mut xm.d[..batch * PIX],
                &mut self.labels,
                &mut self.coarse,
            );
            data::mix_batch(&mut xm.d, batch, amix, acut, 0.5, &mut self.rng)
        };
        targets(
            &self.labels,
            &self.coarse,
            &perm,
            &lam,
            self.taxo,
            self.cfg,
            &mut self.tf,
            &mut self.tc,
        );
        // Rotulo dominante da mistura. Contar o acerto contra o rotulo original
        // faria a acuracia de treino despencar quando o mixup troca a imagem
        // inteira (lambda perto de zero), que e o caso comum no comeco da
        // rampa: a rede acerta a imagem que recebeu e o contador reclama de
        // outra.
        for i in 0..batch {
            self.dom[i] = if lam[i] >= 0.5 {
                self.labels[i]
            } else {
                self.labels[perm[i]]
            };
        }
        (perm, lam)
    }

    /// Um passo completo: lote, forward, perda, backward, clip, otimizador,
    /// EMA. Avanca `self.step`.
    pub fn step(&mut self, b: usize, aug: &Aug, amix: f32, acut: f32) -> StepOut {
        self.make_batch(b, aug, amix, acut);

        self.net.drop_seed = self.rng.next_u64();
        let out = self.net.forward(&self.x, true);
        let lo = build_loss(
            &out,
            &self.tf,
            &self.tc,
            &self.dom,
            &self.coarse,
            self.cfg,
            NCLASS,
            self.taxo.ncoarse,
            self.cfg.batch,
        );
        self.net.backward(&lo.g);

        let lr = lr_at(
            self.cfg.lr,
            self.cfg.lr_min,
            self.step,
            self.total_steps,
            self.warm_steps,
        );
        let (clip, wd) = (self.cfg.clip, self.cfg.wd);
        let mut ps = params_of(&mut self.net);
        if clip > 0.0 {
            optim::clip_(&mut ps, clip);
        }
        self.opt.step_all(&mut ps, lr, wd);
        if let Some(e) = &mut self.ema {
            e.update(&ps);
        }
        self.step += 1;
        StepOut {
            ce: lo.ce,
            total: lo.total,
            correct: lo.correct,
            lr,
        }
    }

    /// Avalia no conjunto de teste. Quando ha EMA, e a media que e avaliada:
    /// e ela que sera salva.
    fn eval_ema(&mut self, test: &data::Dataset, tta: bool) -> (f32, f32) {
        let Some(e) = &mut self.ema else {
            return eval::vit(&mut self.net, test, self.cfg.batch, tta);
        };
        let mut ps = params_of(&mut self.net);
        let old = e.swap_in(&mut ps);
        drop(ps);
        let r = eval::vit(&mut self.net, test, self.cfg.batch, tta);
        let mut ps = params_of(&mut self.net);
        optim::Ema::swap_out(&mut ps, old);
        r
    }

    /// Avalia a epoca e grava o checkpoint se ela bateu o recorde. Devolve
    /// perda e acuracia da EMA mais a acuracia crua (NaN fora das epocas
    /// cheias), que e o que o relatorio imprime.
    fn epoch_end(&mut self, test: &data::Dataset, full: bool, best: &mut f32) -> (f32, f32, f32) {
        let cfg = self.cfg;
        let tta_now = cfg.tta && full;
        let (tl, ta) = self.eval_ema(test, tta_now);
        let raw = if full && self.ema.is_some() {
            eval::vit(&mut self.net, test, cfg.batch, tta_now).1
        } else {
            f32::NAN
        };
        // `raw` e NaN fora das epocas cheias: a negacao (e nao `ta >= raw`) e o
        // que faz o NaN cair na EMA, que e a medida que sempre existe.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let use_ema = self.ema.is_some() && !(raw > ta);
        let show = if raw.is_nan() { ta } else { ta.max(raw) };
        if show > *best {
            *best = show;
            if let Some(p) = cfg.save.as_deref() {
                self.save_best(p, use_ema);
            }
        }
        (tl, ta, raw)
    }

    /// Salva o modelo que a metrica escolheu: a media quando ela ganha, os
    /// pesos crus quando nao.
    fn save_best(&mut self, path: &str, use_ema: bool) {
        if !use_ema {
            return report_save(path, self.net.save(path));
        }
        let mut ps = params_of(&mut self.net);
        let old = self.ema.as_mut().expect("use_ema exige EMA").swap_in(&mut ps);
        drop(ps);
        report_save(path, self.net.save(path));
        let mut ps = params_of(&mut self.net);
        optim::Ema::swap_out(&mut ps, old);
    }

    /// Hash FNV-1a de todo o estado que o treino evolui: pesos, momentos do
    /// otimizador, media exponencial, contador de passos e a ultima perda.
    ///
    /// Os momentos entram porque um bug que so afeta o segundo momento do
    /// AdamW levaria dezenas de passos para aparecer nos pesos, e a EMA porque
    /// e ela que vai para o checkpoint.
    pub fn fingerprint(&mut self, last_loss: f32) -> u64 {
        let mut hash = FNV_OFFSET;
        self.net.for_each_param(&mut |p| {
            for slice in [&p.v, &p.m, &p.v2] {
                for &value in slice {
                    hash = fnv_f32(hash, value);
                }
            }
        });
        if let Some(e) = &self.ema {
            for w in &e.w {
                for &value in w {
                    hash = fnv_f32(hash, value);
                }
            }
        }
        for byte in (self.opt.t as u64).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        fnv_f32(hash, last_loss)
    }
}

// ---------------------------------------------------------------- fingerprint

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_f32(mut hash: u64, value: f32) -> u64 {
    for byte in value.to_bits().to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Executa o prefixo do treino real em uma unica thread e imprime o hash do
/// estado resultante. E o portao que autoriza qualquer refatoracao do passo.
pub fn fingerprint(cfg: &Config, train: &data::Dataset, taxo: &data::Taxo, ntrain: usize) {
    assert!(cfg.fingerprint > 0);
    assert!(
        ntrain >= cfg.batch,
        "--fingerprint requer ao menos um batch completo"
    );
    let mut tr = Trainer::new(cfg, train, taxo, ntrain);
    let mut last_loss = f32::NAN;
    for step in 0..cfg.fingerprint {
        let b = step % tr.steps_per_epoch;
        if b == 0 {
            tr.shuffle();
        }
        let (aug, amix, acut) = aug_at(cfg, step / tr.steps_per_epoch);
        last_loss = tr.step(b, &aug, amix, acut).total;
    }
    let hash = tr.fingerprint(last_loss);
    println!("fingerprint: {hash:016x} | loss {last_loss:.9}");
}

// ---------------------------------------------------------------- lacos

pub fn run_vit(
    cfg: &Config,
    train: &data::Dataset,
    test: &data::Dataset,
    taxo: &data::Taxo,
    ntrain: usize,
) {
    let mut tr = Trainer::new(cfg, train, taxo, ntrain);
    if let Some(p) = &cfg.resume {
        tr.net.load(p).expect("falha ao carregar checkpoint");
        println!("checkpoint carregado de {p}");
    }
    let macs = tr.net.macs();
    let nparams = tr.net.num_params();
    println!(
        "modelo {} | {:.2} M parametros | {:.1} MMAC/img | taxonomia {}x{} | threads {}",
        tr.net.name,
        nparams as f64 / 1e6,
        macs as f64 / 1e6,
        taxo.ncoarse,
        taxo.groups[0].len(),
        rayon::current_num_threads()
    );

    if cfg.eval_only {
        let (l, a) = eval::vit(&mut tr.net, test, cfg.batch, cfg.tta);
        println!("teste: loss {l:.4}  acc {:.2}%", a * 100.0);
        return;
    }
    println!(
        "{} steps/epoca x {} epocas | lr {} -> {} | wd {} | muon {} (ns {}) | ema {} | mem {} | exits {} | taxo {}",
        tr.steps_per_epoch,
        cfg.epochs,
        cfg.lr,
        cfg.lr_min,
        cfg.wd,
        cfg.muon as u8,
        cfg.ns,
        cfg.ema,
        !cfg.no_mem,
        !cfg.no_exits,
        !cfg.no_taxo
    );
    if cfg.ema_auto {
        println!(
            "  ema decay {:.5} (janela de {} passos)",
            tr.ema_decay,
            (1.0 / (1.0 - tr.ema_decay)) as usize
        );
    }
    if cfg.bench > 0 {
        return crate::bench::vit(&mut tr, macs);
    }

    let t0 = Instant::now();
    let mut best = 0.0f32;
    for epoch in 0..cfg.epochs {
        tr.shuffle();
        let te = Instant::now();
        let (aug, amix, acut) = aug_at(cfg, epoch);
        let (mut eloss, mut eobj, mut ecorrect) = (0.0f32, 0.0f32, 0usize);

        for b in 0..tr.steps_per_epoch {
            let s = tr.step(b, &aug, amix, acut);
            eloss += s.ce;
            eobj += s.total;
            ecorrect += s.correct;
            if b % 20 == 0 {
                progress(&tr, epoch, b, eloss, ecorrect, s.lr, aug.ra_mag, t0);
            }
        }

        // A media exponencial e o modelo que sera salvo, entao e ela que e
        // avaliada toda epoca. Uma avaliacao completa com TTA sao quatro
        // passagens sobre as dez mil imagens de teste, ou 19% do tempo de uma
        // epoca; a crua e o TTA entram so nas epocas cheias.
        let full = (epoch + 1) % 10 == 0 || epoch + 1 == cfg.epochs;
        let (tl, ta, raw) = tr.epoch_end(test, full, &mut best);
        println!(
            "\repoca {:>3}/{}  train loss {:.4} (obj {:.4}) acc {:.2}%  |  teste acc {:.2}%{}  loss {:.4}  best {:.2}%  {:.0}s  total {}",
            epoch + 1,
            cfg.epochs,
            eloss / tr.steps_per_epoch as f32,
            eobj / tr.steps_per_epoch as f32,
            100.0 * ecorrect as f32 / (tr.steps_per_epoch * cfg.batch) as f32,
            ta * 100.0,
            raw_note(raw),
            tl,
            best * 100.0,
            te.elapsed().as_secs_f64(),
            fmt_hms(t0.elapsed().as_secs_f64())
        );
    }
    println!("melhor acuracia de teste: {:.2}%", best * 100.0);
}

fn progress(
    tr: &Trainer,
    epoch: usize,
    b: usize,
    eloss: f32,
    ecorrect: usize,
    lr: f32,
    ra_mag: f32,
    t0: Instant,
) {
    let done = tr.step as f64 / tr.total_steps as f64;
    let eta = t0.elapsed().as_secs_f64() * (1.0 - done) / done.max(1e-9);
    print!(
        "\r  epoca {:>3}  [{:>4}/{}]  loss {:.3}  acc {:.1}%  lr {:.5}  ra {:.2}  ETA {}   ",
        epoch + 1,
        b,
        tr.steps_per_epoch,
        eloss / (b + 1) as f32,
        100.0 * ecorrect as f32 / ((b + 1) * tr.cfg.batch) as f32,
        lr,
        ra_mag,
        fmt_hms(eta)
    );
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

pub fn run_resnet(cfg: &Config, train: &data::Dataset, test: &data::Dataset, ntrain: usize) {
    let mut net = ResNet::new(cfg.depth, cfg.width, NCLASS, cfg.seed);
    if let Some(p) = &cfg.resume {
        net.load(p).expect("falha ao carregar checkpoint");
    }
    let (nparams, flops) = (net.num_params(), net.flops(IMG, IMG));
    println!(
        "modelo {} | {} parametros | {:.1} MMAC/img | treino {} | teste {} | threads {}",
        net.name,
        nparams,
        flops as f64 / 2e6,
        ntrain,
        test.len,
        rayon::current_num_threads()
    );
    if cfg.eval_only {
        let (l, a) = eval::resnet(&mut net, test, cfg.batch);
        println!("teste: loss {l:.4}  acc {:.2}%", a * 100.0);
        return;
    }
    let aug = Aug {
        pad: cfg.pad,
        ra_mag: 0.0,
        ra_n: 0,
        erase: 1.0,
        erase_side: cfg.cutout,
    };
    if cfg.bench > 0 {
        return crate::bench::resnet(&mut net, cfg, train, &aug);
    }

    let steps_per_epoch = ntrain / cfg.batch;
    let total_steps = steps_per_epoch * cfg.epochs;
    let warm_steps = steps_per_epoch * cfg.warmup;
    let mut rng = Rng::new(cfg.seed ^ 0xABCD);
    let mut order: Vec<usize> = (0..ntrain).collect();
    let mut x = Arc::new(Tensor::zeros(cfg.batch, 3, IMG, IMG));
    let mut labels = vec![0u32; cfg.batch];
    let mut coarse = vec![0u32; cfg.batch];
    let mut dl = vec![0.0f32; cfg.batch * NCLASS];
    let t0 = Instant::now();
    let mut step = 0usize;
    let mut best = 0.0f32;
    for epoch in 0..cfg.epochs {
        rng.shuffle(&mut order);
        let (mut eloss, mut ecorrect) = (0.0f32, 0usize);
        for b in 0..steps_per_epoch {
            let idx = &order[b * cfg.batch..(b + 1) * cfg.batch];
            data::make_batch_train(
                train,
                idx,
                &aug,
                rng.next_u64(),
                &mut Arc::make_mut(&mut x).d[..cfg.batch * PIX],
                &mut labels,
                &mut coarse,
            );
            let logits = net.forward(&x, true);
            let (l, c) = softmax_ce(&logits, &labels, NCLASS, cfg.smooth, &mut dl);
            net.backward(&dl);
            net.step(
                lr_at(cfg.lr, 0.0, step, total_steps, warm_steps),
                cfg.momentum,
                cfg.wd,
            );
            eloss += l;
            ecorrect += c;
            step += 1;
        }
        let (tl, ta) = eval::resnet(&mut net, test, cfg.batch);
        if ta > best {
            best = ta;
            if let Some(p) = &cfg.save {
                report_save(p, net.save(p));
            }
        }
        println!(
            "epoca {:>3}/{}  train loss {:.4} acc {:.2}%  |  teste loss {:.4} acc {:.2}%  (best {:.2}%)  total {}",
            epoch + 1,
            cfg.epochs,
            eloss / steps_per_epoch as f32,
            100.0 * ecorrect as f32 / (steps_per_epoch * cfg.batch) as f32,
            tl,
            ta * 100.0,
            best * 100.0,
            fmt_hms(t0.elapsed().as_secs_f64())
        );
    }
    println!("melhor acuracia de teste: {:.2}%", best * 100.0);
}

/// A acuracia crua so existe nas epocas cheias; nas outras o relatorio nao
/// mostra a coluna em vez de mostrar um NaN.
fn raw_note(raw: f32) -> String {
    if raw.is_nan() {
        String::new()
    } else {
        format!(" (crua {:.2}%)", raw * 100.0)
    }
}
