//! rustnn: redes convolucionais e um transformer hibrido para CIFAR-100, do
//! zero em Rust, sem BLAS nem biblioteca de ML.
//!
//! `main.rs` cobre a linha de comando, o laco de treino e a avaliacao. A rede
//! mora em `vit.rs`, o GEMM em `gemm.rs` e as verificacoes de gradiente em
//! `check.rs`.

// Este e um codigo de kernel numerico: laco por indice e a linguagem do dominio
// (`for j in 0..n { d[j] = ... }` le como a formula) e as assinaturas largas sao
// os parametros de um GEMM, nao acidente de projeto.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

mod attn;
mod conv;
mod data;
mod gemm;
mod linear;
mod loss;
mod model;
mod nn;
mod norm;
mod optim;
mod pool;
mod rng;
mod tok;
mod vit;

use data::{Aug, IMG, NCLASS, PIX};

/// Reciclar os blocos grandes vale mais que qualquer outra otimizacao de memoria
/// deste modelo: ver `pool.rs`.
#[global_allocator]
static ALLOC: pool::Recycling = pool::Recycling;

use model::ResNet;
use nn::{softmax_ce, Param, Tensor};
use rng::Rng;
use std::sync::Arc;
use std::time::Instant;
use vit::{Cfg, Grads, Out, RustViT};

struct Config {
    arch: String,
    data: String,
    epochs: usize,
    batch: usize,
    lr: f32,
    lr_min: f32,
    momentum: f32,
    wd: f32,
    warmup: usize,
    depth: usize,
    width: usize,
    smooth: f32,
    cutout: usize,
    pad: usize,
    seed: u64,
    threads: usize,
    save: Option<String>,
    resume: Option<String>,
    eval_only: bool,
    subset: usize,
    gradcheck: bool,
    bench: usize,
    gemmbench: bool,
    // rustvit
    dim: usize,
    heads: usize,
    depth3: usize,
    dim4: usize,
    heads4: usize,
    depth4: usize,
    cmid: usize,
    droppath: f32,
    muon: bool,
    ns: usize,
    clip: f32,
    ema: f32,
    /// deriva o decay do numero de passos em vez de usar o valor fixo
    ema_auto: bool,
    sib: f32,
    coarse_w: f32,
    exit_w: (f32, f32),
    alpha: f32,
    temp: f32,
    beta: f32,
    randaug: f32,
    ra_n: usize,
    mixup: f32,
    cutmix: f32,
    erase: f32,
    curriculum: bool,
    tta: bool,
    no_mem: bool,
    no_exits: bool,
    no_taxo: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            arch: "rustvit".into(),
            data: "data/cifar-100-binary".into(),
            epochs: 100,
            batch: 256,
            lr: 6e-4,
            lr_min: 1e-5,
            momentum: 0.9,
            wd: 0.06,
            warmup: 10,
            depth: 2,
            width: 32,
            smooth: 0.1,
            cutout: 8,
            pad: 4,
            seed: 1234,
            threads: 0,
            save: None,
            resume: None,
            eval_only: false,
            subset: 0,
            gradcheck: false,
            bench: 0,
            gemmbench: false,
            dim: 256,
            heads: 4,
            depth3: 5,
            dim4: 384,
            heads4: 6,
            depth4: 2,
            cmid: 96,
            droppath: 0.1,
            muon: true,
            ns: 3,
            clip: 1.0,
            ema: 0.9998,
            ema_auto: true,
            sib: 0.7,
            coarse_w: 0.4,
            exit_w: (0.3, 0.6),
            alpha: 1.0,
            temp: 3.0,
            beta: 0.03,
            randaug: 0.3,
            ra_n: 2,
            mixup: 0.8,
            cutmix: 1.0,
            erase: 0.25,
            curriculum: true,
            tta: true,
            no_mem: false,
            no_exits: false,
            no_taxo: false,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "rustnn - redes do zero em Rust para CIFAR-100

USO: rustnn [opcoes]
  --arch <a>         rustvit | resnet                  (rustvit)
  --data <dir>       diretorio com train.bin/test.bin  (data/cifar-100-binary)
  --epochs <n>       numero de epocas                  (100)
  --batch <n>        tamanho do batch                  (256)
  --lr <f>           learning rate de pico             (6e-4 / 0.2 no resnet)
  --wd <f>           weight decay                      (0.06)
  --warmup <n>       epocas de warmup                  (10)
  --seed <n>         semente                           (1234)
  --threads <n>      threads rayon (0 = todas)         (0)
  --subset <n>       usa so N imagens de treino        (0)
  --save <arq>       salva os melhores pesos
  --resume <arq>     carrega pesos antes de treinar
  --eval             so avalia (use com --resume)
  --gradcheck        checagem numerica do backward e sai
  --bench <n>        cronometra n steps de treino e sai

 arquitetura (rustvit)
  --dim <n>          largura do estagio S3             (256)
  --heads <n>        cabecas em S3                     (4)
  --depth3 <n>       blocos em S3                      (5)
  --dim4 <n>         largura do estagio S4             (384)
  --heads4 <n>       cabecas em S4                     (6)
  --depth4 <n>       blocos em S4                      (2)
  --cmid <n>         canais do tronco convolucional    (96)
  --droppath <f>     stochastic depth no ultimo bloco  (0.1)

 otimizacao
  --muon <0|1>       Muon nas matrizes 2D              (1)
  --ns <n>           iteracoes de Newton-Schulz        (3)
  --clip <f>         clip de norma global              (1.0)
  --ema <f>          decay da media exponencial, 0 off (0.9998)

 perda
  --smooth <f>       label smoothing                   (0.1)
  --sib <f>          fracao do smoothing nos irmaos    (0.7)
  --coarse <f>       peso da perda de superclasse      (0.4)
  --alpha <f>        peso da autodestilacao            (1.0)
  --temp <f>         temperatura da destilacao         (3.0)
  --beta <f>         peso do termo de hint             (0.03)

 augmentation
  --randaug <f>      magnitude do RandAugment, 0 off   (0.3)
  --mixup <f>        alpha do mixup                    (0.8)
  --cutmix <f>       alpha do cutmix                   (1.0)
  --erase <f>        probabilidade do random erasing   (0.25)
  --curriculum <0|1> rampa e corte da augmentation     (1)
  --tta <0|1>        media com o espelho no teste      (1)

 ablacao
  --no-mem           desliga a memoria de alta resolucao
  --no-exits         desliga as saidas auxiliares
  --no-taxo          smoothing uniforme, sem taxonomia
"
    );
    std::process::exit(1)
}

fn parse() -> Config {
    let mut c = Config::default();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let mut lr_set = false;
    let mut ema_set = false;
    macro_rules! next {
        () => {{
            i += 1;
            a.get(i).cloned().unwrap_or_else(|| usage())
        }};
    }
    macro_rules! flag {
        () => {
            next!().parse::<i32>().unwrap_or(1) != 0
        };
    }
    while i < a.len() {
        match a[i].as_str() {
            "--arch" => c.arch = next!(),
            "--data" => c.data = next!(),
            "--epochs" => c.epochs = next!().parse().unwrap(),
            "--batch" => c.batch = next!().parse().unwrap(),
            "--lr" => {
                c.lr = next!().parse().unwrap();
                lr_set = true;
            }
            "--wd" => c.wd = next!().parse().unwrap(),
            "--momentum" => c.momentum = next!().parse().unwrap(),
            "--warmup" => c.warmup = next!().parse().unwrap(),
            "--depth" => c.depth = next!().parse().unwrap(),
            "--width" => c.width = next!().parse().unwrap(),
            "--smooth" => c.smooth = next!().parse().unwrap(),
            "--cutout" => c.cutout = next!().parse().unwrap(),
            "--pad" => c.pad = next!().parse().unwrap(),
            "--seed" => c.seed = next!().parse().unwrap(),
            "--threads" => c.threads = next!().parse().unwrap(),
            "--subset" => c.subset = next!().parse().unwrap(),
            "--save" => c.save = Some(next!()),
            "--resume" => c.resume = Some(next!()),
            "--eval" => c.eval_only = true,
            "--gradcheck" => c.gradcheck = true,
            "--bench" => c.bench = next!().parse().unwrap(),
            "--gemmbench" => c.gemmbench = true,
            "--dim" => c.dim = next!().parse().unwrap(),
            "--heads" => c.heads = next!().parse().unwrap(),
            "--depth3" => c.depth3 = next!().parse().unwrap(),
            "--dim4" => c.dim4 = next!().parse().unwrap(),
            "--heads4" => c.heads4 = next!().parse().unwrap(),
            "--depth4" => c.depth4 = next!().parse().unwrap(),
            "--cmid" => c.cmid = next!().parse().unwrap(),
            "--droppath" => c.droppath = next!().parse().unwrap(),
            "--muon" => c.muon = flag!(),
            "--ns" => c.ns = next!().parse().unwrap(),
            "--clip" => c.clip = next!().parse().unwrap(),
            "--ema" => {
                c.ema = next!().parse().unwrap();
                ema_set = true;
            }
            "--sib" => c.sib = next!().parse().unwrap(),
            "--coarse" => c.coarse_w = next!().parse().unwrap(),
            "--alpha" => c.alpha = next!().parse().unwrap(),
            "--temp" => c.temp = next!().parse().unwrap(),
            "--beta" => c.beta = next!().parse().unwrap(),
            "--randaug" => c.randaug = next!().parse().unwrap(),
            "--mixup" => c.mixup = next!().parse().unwrap(),
            "--cutmix" => c.cutmix = next!().parse().unwrap(),
            "--erase" => c.erase = next!().parse().unwrap(),
            "--curriculum" => c.curriculum = flag!(),
            "--tta" => c.tta = flag!(),
            "--no-mem" => c.no_mem = true,
            "--no-exits" => c.no_exits = true,
            "--no-taxo" => c.no_taxo = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("argumento desconhecido: {other}");
                usage()
            }
        }
        i += 1;
    }
    c.ema_auto = !ema_set;
    if c.arch == "resnet" && !lr_set {
        c.lr = 0.2;
        c.wd = 5e-4;
    }
    c
}

/// Warmup linear, cosseno ate `lr_min`.
fn lr_at(peak: f32, lo: f32, step: usize, total: usize, warm: usize) -> f32 {
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
fn aug_at(cfg: &Config, epoch: usize) -> (Aug, f32, f32) {
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

struct LossOut {
    g: Grads,
    total: f32,
    ce: f32,
    correct: usize,
}

fn build_loss(
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
fn targets(
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

// ---------------------------------------------------------------- avaliacao

fn eval_resnet(net: &mut ResNet, ds: &data::Dataset, batch: usize) -> (f32, f32) {
    let mut x = Arc::new(Tensor::zeros(batch, 3, IMG, IMG));
    let mut labels = vec![0u32; batch];
    let mut dl = vec![0.0f32; batch * NCLASS];
    let (mut loss_, mut correct, mut seen, mut s) = (0.0f32, 0usize, 0usize, 0usize);
    while s < ds.len {
        let bs = batch.min(ds.len - s);
        {
            let xm = Arc::make_mut(&mut x);
            xm.resize(bs, 3, IMG, IMG);
            data::make_batch_eval(ds, s, bs, false, &mut xm.d, &mut labels);
        }
        let logits = net.forward(&x, false);
        let (l, c) = softmax_ce(&logits, &labels[..bs], NCLASS, 0.0, &mut dl[..bs * NCLASS]);
        loss_ += l * bs as f32;
        correct += c;
        seen += bs;
        s += bs;
    }
    (loss_ / seen as f32, correct as f32 / seen as f32)
}

fn eval_vit(net: &mut RustViT, ds: &data::Dataset, batch: usize, tta: bool) -> (f32, f32) {
    let ex = net.use_exits;
    net.use_exits = false;
    let mut x = Arc::new(Tensor::zeros(batch, 3, IMG, IMG));
    let mut labels = vec![0u32; batch];
    let mut dl = vec![0.0f32; batch * NCLASS];
    let (mut loss_, mut correct, mut seen, mut s) = (0.0f32, 0usize, 0usize, 0usize);
    while s < ds.len {
        let bs = batch.min(ds.len - s);
        {
            let xm = Arc::make_mut(&mut x);
            xm.resize(bs, 3, IMG, IMG);
            data::make_batch_eval(ds, s, bs, false, &mut xm.d, &mut labels);
        }
        let mut logits = net.forward(&x, false).logits;
        if tta {
            {
                let xm = Arc::make_mut(&mut x);
                data::make_batch_eval(ds, s, bs, true, &mut xm.d, &mut labels);
            }
            let f = net.forward(&x, false).logits;
            logits.iter_mut().zip(&f).for_each(|(a, b)| *a = 0.5 * (*a + b));
        }
        let (l, c) = softmax_ce(&logits, &labels[..bs], NCLASS, 0.0, &mut dl[..bs * NCLASS]);
        loss_ += l * bs as f32;
        correct += c;
        seen += bs;
        s += bs;
    }
    net.use_exits = ex;
    (loss_ / seen as f32, correct as f32 / seen as f32)
}

fn fmt_hms(secs: f64) -> String {
    let s = secs as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

fn make_cfg(c: &Config) -> Cfg {
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

include!("check.rs");

/// Cronometra o driver de GEMM nas formas exatas que o modelo usa.
fn gemmbench() {
    let mut rng = Rng::new(7);
    let shapes: [(usize, usize, usize, bool, bool, &str); 6] = [
        (16384, 768, 256, false, true, "S3 qkv fwd      "),
        (16384, 704, 256, false, true, "S3 swiglu wg    "),
        (16384, 256, 768, false, false, "S3 qkv dX       "),
        (768, 256, 16384, true, false, "S3 qkv dW       "),
        (16384, 256, 352, false, true, "S3 swiglu wd fwd"),
        (4096, 384, 1024, false, true, "merge fwd       "),
    ];
    for (m, n, k, at, bt, tag) in shapes {
        let a = rand_vec(m * k, &mut rng);
        let b = rand_vec(k * n, &mut rng);
        let mut c = vec![0.0f32; m * n];
        gemm::matmul(m, n, k, &a, at, &b, bt, &mut c, false);
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            gemm::matmul(m, n, k, &a, at, &b, bt, &mut c, false);
        }
        let el = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "  {tag} {m:>6}x{n:>4}x{k:<6} {:>7.1} ms  {:>6.1} GFLOP/s",
            1000.0 * el,
            2.0 * (m * n * k) as f64 / el / 1e9
        );
    }
}

fn main() {
    let cfg = parse();
    if cfg.gemmbench {
        if cfg.threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(cfg.threads)
                .build_global()
                .unwrap();
        }
        return gemmbench();
    }
    if cfg.gradcheck {
        let ok = layercheck(cfg.seed) & vitcheck(cfg.seed) & gradcheck(cfg.seed);
        std::process::exit(if ok { 0 } else { 1 });
    }
    if cfg.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .build_global()
            .unwrap();
    }

    let (train, test) = match data::load(&cfg.data) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "erro lendo o dataset em '{}': {e}\n\
                 baixe com: curl -LO https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz && tar xzf cifar-100-binary.tar.gz",
                cfg.data
            );
            std::process::exit(1);
        }
    };
    let taxo = data::taxonomy(&train);
    let ntrain = if cfg.subset > 0 {
        cfg.subset.min(train.len)
    } else {
        train.len
    };

    if cfg.arch == "resnet" {
        return run_resnet(&cfg, &train, &test, ntrain);
    }

    let mut net = RustViT::new(make_cfg(&cfg), taxo.parent.clone(), taxo.ncoarse, cfg.seed);
    net.use_exits = !cfg.no_exits;
    if let Some(p) = &cfg.resume {
        net.load(p).expect("falha ao carregar checkpoint");
        println!("checkpoint carregado de {p}");
    }
    let nparams = net.num_params();
    let macs = net.macs();
    println!(
        "modelo {} | {:.2} M parametros | {:.1} MMAC/img | taxonomia {}x{} | threads {}",
        net.name,
        nparams as f64 / 1e6,
        macs as f64 / 1e6,
        taxo.ncoarse,
        taxo.groups[0].len(),
        rayon::current_num_threads()
    );

    if cfg.eval_only {
        let (l, a) = eval_vit(&mut net, &test, cfg.batch, cfg.tta);
        println!("teste: loss {l:.4}  acc {:.2}%", a * 100.0);
        return;
    }

    let steps_per_epoch = ntrain / cfg.batch;
    let total_steps = steps_per_epoch * cfg.epochs;
    let warm_steps = steps_per_epoch * cfg.warmup;
    println!(
        "{} steps/epoca x {} epocas | lr {} -> {} | wd {} | muon {} (ns {}) | ema {} | mem {} | exits {} | taxo {}",
        steps_per_epoch,
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

    let mut opt = optim::Optim::new(cfg.muon, cfg.ns);
    // A janela da media e 1/(1-decay) passos. Um decay fixo de 0.9998 da 5000
    // passos, que num treino de 300 epocas e um decimo do total e num de 100 e
    // um quarto: a media fica sempre atrasada em relacao a um modelo que ainda
    // melhora meio ponto por epoca. Derivar o decay do orcamento de passos
    // mantem a janela em 10% do treino, seja ele qual for.
    let ema_d = if cfg.ema_auto && total_steps > 0 {
        (1.0 - 10.0 / total_steps as f32).min(cfg.ema)
    } else {
        cfg.ema
    };
    let mut ema = if ema_d > 0.0 {
        Some(optim::Ema::new(ema_d))
    } else {
        None
    };
    if cfg.ema_auto {
        println!(
            "  ema decay {ema_d:.5} (janela de {} passos)",
            (1.0 / (1.0 - ema_d)) as usize
        );
    }
    let mut rng = Rng::new(cfg.seed ^ 0xABCD);
    let mut order: Vec<usize> = (0..ntrain).collect();
    let mut x = Arc::new(Tensor::zeros(cfg.batch, 3, IMG, IMG));
    let mut labels = vec![0u32; cfg.batch];
    let mut dom = vec![0u32; cfg.batch];
    let mut coarse = vec![0u32; cfg.batch];
    let mut tf = vec![0.0f32; cfg.batch * NCLASS];
    let mut tc = vec![0.0f32; cfg.batch * taxo.ncoarse];
    let t0 = Instant::now();
    let mut step = 0usize;
    let mut best = 0.0f32;

    if cfg.bench > 0 {
        let (aug, _, _) = aug_at(&cfg, cfg.epochs / 2);
        let ord: Vec<usize> = (0..cfg.batch).collect();
        {
            let xm = Arc::make_mut(&mut x);
            data::make_batch_train(&train, &ord, &aug, 1, &mut xm.d, &mut labels, &mut coarse);
        }
        let perm: Vec<usize> = (0..cfg.batch).collect();
        let lam = vec![1.0f32; cfg.batch];
        targets(&labels, &coarse, &perm, &lam, &taxo, &cfg, &mut tf, &mut tc);
        let (mut tf_, mut tb_, mut ts_) = (0.0f64, 0.0f64, 0.0f64);
        let mut run = |net: &mut RustViT, opt: &mut optim::Optim, acc: bool| {
            let a = Instant::now();
            let out = net.forward(&x, true);
            let lo = build_loss(
                &out,
                &tf,
                &tc,
                &labels,
                &coarse,
                &cfg,
                NCLASS,
                taxo.ncoarse,
                cfg.batch,
            );
            if acc {
                tf_ += a.elapsed().as_secs_f64();
            }
            let a = Instant::now();
            net.backward(&lo.g);
            if acc {
                tb_ += a.elapsed().as_secs_f64();
            }
            let a = Instant::now();
            let mut ps: Vec<&mut Param> = Vec::new();
            net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
            opt.step_all(&mut ps, 0.0, 0.0);
            if acc {
                ts_ += a.elapsed().as_secs_f64();
            }
        };
        for _ in 0..2 {
            run(&mut net, &mut opt, false);
        }
        let t = Instant::now();
        for _ in 0..cfg.bench {
            run(&mut net, &mut opt, true);
        }
        let tot = t.elapsed().as_secs_f64();
        println!(
            "  fwd {:.0} ms | bwd {:.0} ms | opt {:.0} ms",
            1000.0 * tf_ / cfg.bench as f64,
            1000.0 * tb_ / cfg.bench as f64,
            1000.0 * ts_ / cfg.bench as f64
        );
        let imgs = (cfg.bench * cfg.batch) as f64;
        println!(
            "bench: {:.1} img/s | step {:.0} ms | {:.1} GFLOP/s | epoca estimada {:.0}s",
            imgs / tot,
            1000.0 * tot / cfg.bench as f64,
            macs as f64 * 6.0 * imgs / tot / 1e9,
            50000.0 / (imgs / tot)
        );
        return;
    }

    for epoch in 0..cfg.epochs {
        rng.shuffle(&mut order);
        let te = Instant::now();
        let (aug, amix, acut) = aug_at(&cfg, epoch);
        let (mut eloss, mut eobj, mut ecorrect) = (0.0f32, 0.0f32, 0usize);

        for b in 0..steps_per_epoch {
            let idx = &order[b * cfg.batch..(b + 1) * cfg.batch];
            let (perm, lam) = {
                let xm = Arc::make_mut(&mut x);
                data::make_batch_train(
                    &train,
                    idx,
                    &aug,
                    rng.next_u64(),
                    &mut xm.d[..cfg.batch * PIX],
                    &mut labels,
                    &mut coarse,
                );
                data::mix_batch(&mut xm.d, cfg.batch, amix, acut, 0.5, &mut rng)
            };
            targets(&labels, &coarse, &perm, &lam, &taxo, &cfg, &mut tf, &mut tc);
            // Rotulo dominante da mistura. Contar o acerto contra o rotulo
            // original faria a acuracia de treino despencar quando o mixup
            // troca a imagem inteira (lambda perto de zero), que e o caso comum
            // no comeco da rampa: a rede acerta a imagem que recebeu e o
            // contador reclama de outra.
            for i in 0..cfg.batch {
                dom[i] = if lam[i] >= 0.5 { labels[i] } else { labels[perm[i]] };
            }

            net.drop_seed = rng.next_u64();
            let out = net.forward(&x, true);
            let lo = build_loss(
                &out,
                &tf,
                &tc,
                &dom,
                &coarse,
                &cfg,
                NCLASS,
                taxo.ncoarse,
                cfg.batch,
            );
            net.backward(&lo.g);

            let mut ps: Vec<&mut Param> = Vec::new();
            net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
            if cfg.clip > 0.0 {
                optim::clip_(&mut ps, cfg.clip);
            }
            let lr = lr_at(cfg.lr, cfg.lr_min, step, total_steps, warm_steps);
            opt.step_all(&mut ps, lr, cfg.wd);
            opt.t += 1;
            if let Some(e) = &mut ema {
                e.update(&ps);
            }

            eloss += lo.ce;
            eobj += lo.total;
            ecorrect += lo.correct;
            step += 1;

            if b % 20 == 0 {
                let done = step as f64 / total_steps as f64;
                let eta = t0.elapsed().as_secs_f64() * (1.0 - done) / done.max(1e-9);
                print!(
                    "\r  epoca {:>3}  [{:>4}/{}]  loss {:.3}  acc {:.1}%  lr {:.5}  ra {:.2}  ETA {}   ",
                    epoch + 1,
                    b,
                    steps_per_epoch,
                    eloss / (b + 1) as f32,
                    100.0 * ecorrect as f32 / ((b + 1) * cfg.batch) as f32,
                    lr,
                    aug.ra_mag,
                    fmt_hms(eta)
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }

        // A media exponencial e o modelo que sera salvo, entao e ela que e
        // avaliada toda epoca. Uma avaliacao completa com TTA sao quatro
        // passagens sobre as dez mil imagens de teste, ou 19% do tempo de uma
        // epoca; a crua e o TTA entram so nas epocas cheias.
        let full = (epoch + 1) % 10 == 0 || epoch + 1 == cfg.epochs;
        let tta_now = cfg.tta && full;
        let (tl, ta) = if let Some(e) = &mut ema {
            let mut ps: Vec<&mut Param> = Vec::new();
            net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
            let old = e.swap_in(&mut ps);
            drop(ps);
            let r = eval_vit(&mut net, &test, cfg.batch, tta_now);
            let mut ps: Vec<&mut Param> = Vec::new();
            net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
            optim::Ema::swap_out(&mut ps, old);
            r
        } else {
            eval_vit(&mut net, &test, cfg.batch, tta_now)
        };
        let raw = if full && ema.is_some() {
            eval_vit(&mut net, &test, cfg.batch, tta_now).1
        } else {
            f32::NAN
        };
        // `raw` e NaN fora das epocas cheias: a negacao (e nao `ta >= raw`) e o
        // que faz o NaN cair na EMA, que e a medida que sempre existe.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let use_ema = ema.is_some() && !(raw > ta);
        let show = if raw.is_nan() { ta } else { ta.max(raw) };
        if show > best {
            best = show;
            if let Some(p) = &cfg.save {
                if use_ema {
                    let mut ps: Vec<&mut Param> = Vec::new();
                    net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
                    let old = ema.as_mut().unwrap().swap_in(&mut ps);
                    drop(ps);
                    let _ = net.save(p);
                    let mut ps: Vec<&mut Param> = Vec::new();
                    net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
                    optim::Ema::swap_out(&mut ps, old);
                } else {
                    let _ = net.save(p);
                }
            }
        }
        println!(
            "\repoca {:>3}/{}  train loss {:.4} (obj {:.4}) acc {:.2}%  |  teste acc {:.2}%{}  loss {:.4}  best {:.2}%  {:.0}s  total {}",
            epoch + 1,
            cfg.epochs,
            eloss / steps_per_epoch as f32,
            eobj / steps_per_epoch as f32,
            100.0 * ecorrect as f32 / (steps_per_epoch * cfg.batch) as f32,
            ta * 100.0,
            if raw.is_nan() {
                String::new()
            } else {
                format!(" (crua {:.2}%)", raw * 100.0)
            },
            tl,
            best * 100.0,
            te.elapsed().as_secs_f64(),
            fmt_hms(t0.elapsed().as_secs_f64())
        );
    }
    println!("melhor acuracia de teste: {:.2}%", best * 100.0);
}

fn run_resnet(cfg: &Config, train: &data::Dataset, test: &data::Dataset, ntrain: usize) {
    let mut net = ResNet::new(cfg.depth, cfg.width, NCLASS, cfg.seed);
    if let Some(p) = &cfg.resume {
        net.load(p).expect("falha ao carregar checkpoint");
    }
    let nparams = net.num_params();
    println!(
        "modelo {} | {} parametros | {:.1} MMAC/img | treino {} | teste {} | threads {}",
        net.name,
        nparams,
        net.flops(IMG, IMG) as f64 / 2e6,
        ntrain,
        test.len,
        rayon::current_num_threads()
    );
    if cfg.eval_only {
        let (l, a) = eval_resnet(&mut net, test, cfg.batch);
        println!("teste: loss {l:.4}  acc {:.2}%", a * 100.0);
        return;
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
    let aug = Aug {
        pad: cfg.pad,
        ra_mag: 0.0,
        ra_n: 0,
        erase: 1.0,
        erase_side: cfg.cutout,
    };

    if cfg.bench > 0 {
        let idx: Vec<usize> = (0..cfg.batch).collect();
        data::make_batch_train(
            train,
            &idx,
            &aug,
            cfg.seed,
            &mut Arc::make_mut(&mut x).d[..cfg.batch * PIX],
            &mut labels,
            &mut coarse,
        );
        let (mut tf, mut tb, mut ts) = (0.0f64, 0.0f64, 0.0f64);
        let t = Instant::now();
        for _ in 0..cfg.bench {
            let a = Instant::now();
            let lg = net.forward(&x, true);
            softmax_ce(&lg, &labels, NCLASS, cfg.smooth, &mut dl);
            tf += a.elapsed().as_secs_f64();
            let a = Instant::now();
            net.backward(&dl);
            tb += a.elapsed().as_secs_f64();
            let a = Instant::now();
            net.step(0.0, 0.0, 0.0);
            ts += a.elapsed().as_secs_f64();
        }
        let tot = t.elapsed().as_secs_f64();
        let imgs = (cfg.bench * cfg.batch) as f64;
        println!(
            "  fwd {:.0} ms | bwd {:.0} ms | opt {:.0} ms",
            1000.0 * tf / cfg.bench as f64,
            1000.0 * tb / cfg.bench as f64,
            1000.0 * ts / cfg.bench as f64
        );
        println!(
            "bench: {:.1} img/s | step {:.0} ms | {:.1} GFLOP/s | epoca estimada {:.0}s",
            imgs / tot,
            1000.0 * tot / cfg.bench as f64,
            net.flops(IMG, IMG) as f64 * 3.0 * imgs / tot / 1e9,
            50000.0 / (imgs / tot)
        );
        return;
    }

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
        let (tl, ta) = eval_resnet(&mut net, test, cfg.batch);
        if ta > best {
            best = ta;
            if let Some(p) = &cfg.save {
                let _ = net.save(p);
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
