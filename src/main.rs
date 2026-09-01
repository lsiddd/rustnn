mod conv;
mod data;
mod gemm;
mod model;
mod nn;
mod rng;

use data::{IMG, NCLASS, PIX};
use model::ResNet;
use nn::{softmax_ce, Tensor};
use std::sync::Arc;
use rng::Rng;
use std::time::Instant;

struct Config {
    data: String,
    epochs: usize,
    batch: usize,
    lr: f32,
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
}

impl Default for Config {
    fn default() -> Self {
        Config {
            data: "data/cifar-100-binary".into(),
            epochs: 30,
            batch: 128,
            lr: 0.2,
            momentum: 0.9,
            wd: 5e-4,
            warmup: 2,
            depth: 2, // 6n+2 = 14 camadas
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
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "rustnn - CNN residual do zero em Rust para CIFAR-100

USO: rustnn [opcoes]
  --data <dir>       diretorio com train.bin/test.bin  (default data/cifar-100-binary)
  --epochs <n>       numero de epocas                  (30)
  --batch <n>        tamanho do batch                  (128)
  --lr <f>           learning rate de pico             (0.2)
  --wd <f>           weight decay                      (5e-4)
  --momentum <f>     momentum de Nesterov              (0.9)
  --warmup <n>       epocas de warmup linear           (2)
  --depth <n>        blocos por estagio -> 6n+2 camadas(2)
  --width <n>        largura base do 1o estagio        (32)
  --smooth <f>       label smoothing                   (0.1)
  --cutout <n>       lado do cutout em pixels, 0 off   (8)
  --pad <n>          padding do random crop, 0 off     (4)
  --seed <n>         semente                           (1234)
  --threads <n>      threads rayon (0 = todas)         (0)
  --subset <n>       usa so N imagens de treino (debug) (0)
  --save <arq>       salva pesos ao final
  --resume <arq>     carrega pesos antes de treinar
  --eval             so avalia (use com --resume)
  --gradcheck        checagem numerica do backward e sai
  --bench <n>        cronometra n steps de treino (fwd+bwd+step) e sai
"
    );
    std::process::exit(1)
}

fn parse() -> Config {
    let mut c = Config::default();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    macro_rules! next {
        () => {{
            i += 1;
            a.get(i).cloned().unwrap_or_else(|| usage())
        }};
    }
    while i < a.len() {
        match a[i].as_str() {
            "--data" => c.data = next!(),
            "--epochs" => c.epochs = next!().parse().unwrap(),
            "--batch" => c.batch = next!().parse().unwrap(),
            "--lr" => c.lr = next!().parse().unwrap(),
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
            "-h" | "--help" => usage(),
            other => {
                eprintln!("argumento desconhecido: {other}");
                usage()
            }
        }
        i += 1;
    }
    c
}

/// LR com warmup linear seguido de decaimento cosseno.
fn lr_at(cfg: &Config, step: usize, total: usize, warm: usize) -> f32 {
    if step < warm {
        cfg.lr * (step + 1) as f32 / warm as f32
    } else {
        let t = (step - warm) as f32 / (total - warm).max(1) as f32;
        0.5 * cfg.lr * (1.0 + (std::f32::consts::PI * t).cos())
    }
}

fn evaluate(net: &mut ResNet, ds: &data::Dataset, batch: usize) -> (f32, f32) {
    let mut x = Arc::new(Tensor::zeros(batch, 3, IMG, IMG));
    let mut labels = vec![0u32; batch];
    let mut dl = vec![0.0f32; batch * NCLASS];
    let (mut loss, mut correct) = (0.0f32, 0usize);
    let mut seen = 0usize;
    let mut s = 0;
    while s < ds.len {
        let bs = batch.min(ds.len - s);
        {
            let xm = Arc::make_mut(&mut x);
            xm.resize(bs, 3, IMG, IMG);
            data::make_batch_eval(ds, s, bs, &mut xm.d, &mut labels);
        }
        let logits = net.forward(&x, false);
        let (l, c) = softmax_ce(&logits, &labels[..bs], NCLASS, 0.0, &mut dl[..bs * NCLASS]);
        loss += l * bs as f32;
        correct += c;
        seen += bs;
        s += bs;
    }
    (loss / seen as f32, correct as f32 / seen as f32)
}

fn fmt_hms(secs: f64) -> String {
    let s = secs as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// Derivada direcional: compara (L(x+ev)-L(x-ev))/2e com <g, v> numa direcao
/// aleatoria unitaria. Agrega o sinal do gradiente inteiro, entao nao sofre com
/// o ruido de f32 que domina componentes individuais minusculas.
fn dircheck(
    base: &[f32],
    grad: &[f32],
    eps: f32,
    rng: &mut Rng,
    mut loss: impl FnMut(&[f32]) -> f32,
) -> f32 {
    let n = base.len();
    // v ~ N(0,1) sem normalizar: o sinal cresce com sqrt(n) e fica bem acima
    // do ruido de arredondamento de f32 mesmo em tensores grandes.
    let v: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
    let mut c = base.to_vec();
    for i in 0..n {
        c[i] = base[i] + eps * v[i];
    }
    let lp = loss(&c);
    for i in 0..n {
        c[i] = base[i] - eps * v[i];
    }
    let lm = loss(&c);
    let num = (lp - lm) / (2.0 * eps);
    let ana: f32 = grad.iter().zip(&v).map(|(a, b)| a * b).sum();
    (num - ana).abs() / (num.abs() + ana.abs() + 1e-6)
}

fn report(name: &str, err: f32) -> bool {
    let ok = err < 5e-3;
    println!(
        "  {name:<26} erro relativo {err:.6}  {}",
        if ok { "OK" } else { "FALHOU" }
    );
    ok
}

/// Gradcheck por camada, isolando cada backward.
fn layercheck(seed: u64) -> bool {
    use nn::{BatchNorm2d, Conv2d, Linear};
    let mut rng = Rng::new(seed);
    let eps: f32 = std::env::var("GC_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3e-4);
    let mut all = true;
    println!("gradcheck por camada (eps {eps}):");

    for (cin, cout, k, stride, pad, hh, tag) in [
        (3usize, 4usize, 3usize, 1usize, 1usize, 6usize, "conv 3x3/s1"),
        (4, 6, 1, 2, 0, 8, "conv 1x1/s2"),
        (4, 5, 3, 2, 1, 7, "conv 3x3/s2"),
    ] {
        let mut conv = Conv2d::new(cin, cout, k, stride, pad, &mut rng);
        let mut x0 = Tensor::zeros(2, cin, hh, hh);
        x0.d.iter_mut().for_each(|v| *v = rng.normal());
        let x = Arc::new(x0);
        let y0 = conv.forward(&x, true);
        let r: Vec<f32> = (0..y0.d.len()).map(|_| rng.normal()).collect();
        let mut dy = y0.clone();
        dy.d.copy_from_slice(&r);
        let dx = conv.backward(&dy);
        let dw = conv.w.g.clone();

        let xb = x.d.clone();
        let mut xt = Arc::new((*x).clone());
        let e = dircheck(&xb, &dx.d, eps, &mut rng, |c| {
            Arc::make_mut(&mut xt).d.copy_from_slice(c);
            conv.forward(&xt, false).d.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        all &= report(&format!("{tag} dx"), e);

        let wb = conv.w.v.clone();
        let e = dircheck(&wb, &dw, eps, &mut rng, |c| {
            conv.w.v.copy_from_slice(c);
            conv.forward(&x, false).d.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        conv.w.v.copy_from_slice(&wb);
        all &= report(&format!("{tag} dw"), e);
    }

    {
        let c = 4usize;
        let mut bn = BatchNorm2d::new(c);
        bn.g.v.iter_mut().for_each(|v| *v = 0.5 + rng.uniform());
        bn.b.v.iter_mut().for_each(|v| *v = rng.normal());
        let mut x = Tensor::zeros(3, c, 5, 5);
        x.d.iter_mut().for_each(|v| *v = rng.normal() * 2.0 + 1.0);
        let y0 = bn.forward(x.clone(), true);
        let r: Vec<f32> = (0..y0.d.len()).map(|_| rng.normal()).collect();
        let mut dy = y0.clone();
        dy.d.copy_from_slice(&r);
        let dx = bn.backward(&dy);
        let (dg, db) = (bn.g.g.clone(), bn.b.g.clone());

        let xb = x.d.clone();
        let mut xt = x.clone();
        let e = dircheck(&xb, &dx.d, eps, &mut rng, |c| {
            xt.d.copy_from_slice(c);
            bn.forward(xt.clone(), true).d.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        all &= report("batchnorm dx", e);

        let gb = bn.g.v.clone();
        let e = dircheck(&gb, &dg, eps, &mut rng, |c| {
            bn.g.v.copy_from_slice(c);
            bn.forward(x.clone(), true).d.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        bn.g.v.copy_from_slice(&gb);
        all &= report("batchnorm dgamma", e);

        let bb = bn.b.v.clone();
        let e = dircheck(&bb, &db, eps, &mut rng, |c| {
            bn.b.v.copy_from_slice(c);
            bn.forward(x.clone(), true).d.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        bn.b.v.copy_from_slice(&bb);
        all &= report("batchnorm dbeta", e);
    }

    {
        let mut lin = Linear::new(6, 5, &mut rng);
        lin.b.v.iter_mut().for_each(|v| *v = rng.normal());
        let mut x = Tensor::zeros(3, 6, 1, 1);
        x.d.iter_mut().for_each(|v| *v = rng.normal());
        let y0 = lin.forward(&x, true);
        let r: Vec<f32> = (0..y0.len()).map(|_| rng.normal()).collect();
        let dx = lin.backward(&r);
        let dw = lin.w.g.clone();

        let xb = x.d.clone();
        let mut xt = x.clone();
        let e = dircheck(&xb, &dx.d, eps, &mut rng, |c| {
            xt.d.copy_from_slice(c);
            lin.forward(&xt, false).iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        all &= report("linear dx", e);

        let wb = lin.w.v.clone();
        let e = dircheck(&wb, &dw, eps, &mut rng, |c| {
            lin.w.v.copy_from_slice(c);
            lin.forward(&x, false).iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        lin.w.v.copy_from_slice(&wb);
        all &= report("linear dw", e);
    }

    for (cin, cout, stride, tag) in [
        (4usize, 4usize, 1usize, "block id/s1"),
        (4, 8, 2, "block down/s2"),
    ] {
        let mut blk = model::BasicBlock::new(cin, cout, stride, &mut rng);
        blk.for_each_param(&mut |p| p.v.iter_mut().for_each(|v| *v += 0.3 * rng.normal()));
        let mut x0 = Tensor::zeros(2, cin, 8, 8);
        x0.d.iter_mut().for_each(|v| *v = rng.normal());
        let x = Arc::new(x0);
        let y0 = blk.forward(&x, true);
        let r: Vec<f32> = (0..y0.d.len()).map(|_| rng.normal()).collect();
        let mut dy = (*y0).clone();
        dy.d.copy_from_slice(&r);
        let dx = blk.backward(dy);
        let xb = x.d.clone();
        let mut xt = Arc::new((*x).clone());
        let e = dircheck(&xb, &dx.d, eps, &mut rng, |c| {
            Arc::make_mut(&mut xt).d.copy_from_slice(c);
            blk.forward(&xt, true).d.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        all &= report(&format!("{tag} dx"), e);

        let mut base: Vec<f32> = Vec::new();
        let mut grad: Vec<f32> = Vec::new();
        blk.for_each_param(&mut |p| {
            base.extend_from_slice(&p.v);
            grad.extend_from_slice(&p.g);
        });
        let e = dircheck(&base, &grad, eps, &mut rng, |c| {
            let mut off = 0;
            blk.for_each_param(&mut |p| {
                let n = p.v.len();
                p.v.copy_from_slice(&c[off..off + n]);
                off += n;
            });
            blk.forward(&x, true).d.iter().zip(&r).map(|(a, b)| a * b).sum()
        });
        let mut off = 0;
        blk.for_each_param(&mut |p| {
            let n = p.v.len();
            p.v.copy_from_slice(&base[off..off + n]);
            off += n;
        });
        all &= report(&format!("{tag} dparams"), e);
    }

    {
        let (n, nc) = (4usize, 7usize);
        let z: Vec<f32> = (0..n * nc).map(|_| rng.normal()).collect();
        let labels: Vec<u32> = (0..n).map(|_| rng.below(nc) as u32).collect();
        let mut d = vec![0.0f32; n * nc];
        softmax_ce(&z, &labels, nc, 0.1, &mut d);
        let mut tmp = vec![0.0f32; n * nc];
        let e = dircheck(&z, &d, eps, &mut rng, |c| {
            softmax_ce(c, &labels, nc, 0.1, &mut tmp).0
        });
        all &= report("softmax_ce dlogits", e);
    }
    all
}

/// Gradcheck da rede completa, um tensor de parametros por vez.
fn gradcheck(seed: u64) -> bool {
    const NC: usize = 10;
    const BS: usize = 8;
    let eps: f32 = std::env::var("GC_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3e-4);
    let mut net = ResNet::new(1, 4, NC, seed);
    let mut rng = Rng::new(seed ^ 0x5EED);
    // afasta gamma/beta do zero: com beta=0 e gamma=0 muitas pre-ativacoes caem
    // exatamente no kink da ReLU e a diferenca finita central perde o sentido.
    net.for_each_param(&mut |p| p.v.iter_mut().for_each(|v| *v += 0.3 * rng.normal()));
    let mut x0 = Tensor::zeros(BS, 3, 16, 16);
    x0.d.iter_mut().for_each(|v| *v = rng.normal());
    let x = Arc::new(x0);
    let labels: Vec<u32> = (0..BS).map(|_| rng.below(NC) as u32).collect();
    let mut dl = vec![0.0f32; BS * NC];

    let logits = net.forward(&x, true);
    softmax_ce(&logits, &labels, NC, 0.0, &mut dl);
    net.backward(&dl);

    // Um unico dircheck sobre TODOS os parametros concatenados: o sinal e a norma
    // do gradiente inteiro, entao nada fica escondido no ruido de f32.
    let mut base: Vec<f32> = Vec::new();
    let mut grad: Vec<f32> = Vec::new();
    net.for_each_param(&mut |p| {
        base.extend_from_slice(&p.v);
        grad.extend_from_slice(&p.g);
    });
    let e = dircheck(&base, &grad, eps, &mut rng, |c| {
        let mut off = 0;
        net.for_each_param(&mut |p| {
            let n = p.v.len();
            p.v.copy_from_slice(&c[off..off + n]);
            off += n;
        });
        let mut tmp = vec![0.0f32; BS * NC];
        let lg = net.forward(&x, true);
        softmax_ce(&lg, &labels, NC, 0.0, &mut tmp).0
    });
    let mut off = 0;
    net.for_each_param(&mut |p| {
        let n = p.v.len();
        p.v.copy_from_slice(&base[off..off + n]);
        off += n;
    });
    report("resnet completa (todos os params)", e)
}

fn main() {
    let cfg = parse();
    if cfg.gradcheck {
        let ok = layercheck(cfg.seed) & gradcheck(cfg.seed);
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
    let ntrain = if cfg.subset > 0 {
        cfg.subset.min(train.len)
    } else {
        train.len
    };

    let mut net = ResNet::new(cfg.depth, cfg.width, NCLASS, cfg.seed);
    if let Some(p) = &cfg.resume {
        net.load(p).expect("falha ao carregar checkpoint");
        println!("checkpoint carregado de {p}");
    }
    let nparams = net.num_params();
    println!(
        "modelo {} | {} parametros | treino {} | teste {} | threads {}",
        net.name,
        nparams,
        ntrain,
        test.len,
        rayon::current_num_threads()
    );

    if cfg.bench > 0 {
        let mut x = Arc::new(Tensor::zeros(cfg.batch, 3, IMG, IMG));
        let mut labels = vec![0u32; cfg.batch];
        let mut dl = vec![0.0f32; cfg.batch * NCLASS];
        let order: Vec<usize> = (0..cfg.batch).collect();
        let xm = Arc::make_mut(&mut x);
        data::make_batch_train(&train, &order, cfg.pad, cfg.cutout, 1, &mut xm.d, &mut labels);
        // aquece
        for _ in 0..2 {
            let lg = net.forward(&x, true);
            softmax_ce(&lg, &labels, NCLASS, cfg.smooth, &mut dl);
            net.backward(&dl);
            net.step(0.0, 0.0, 0.0);
        }
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
        let gf = net.flops(IMG, IMG) as f64 * 3.0 * imgs / tot / 1e9;
        println!("  {:.1} GFLOP/s (fwd+bwd, ~3x MACs de forward)", gf);
        println!(
            "bench: {:.1} img/s | step {:.1} ms (fwd {:.1} / bwd {:.1} / upd {:.1}) | epoca estimada {:.0}s",
            imgs / tot,
            1000.0 * tot / cfg.bench as f64,
            1000.0 * tf / cfg.bench as f64,
            1000.0 * tb / cfg.bench as f64,
            1000.0 * ts / cfg.bench as f64,
            50000.0 / (imgs / tot)
        );
        return;
    }

    if cfg.eval_only {
        let (l, a) = evaluate(&mut net, &test, cfg.batch);
        println!("teste: loss {l:.4}  acc {:.2}%", a * 100.0);
        return;
    }

    let steps_per_epoch = ntrain / cfg.batch;
    let total_steps = steps_per_epoch * cfg.epochs;
    let warm_steps = steps_per_epoch * cfg.warmup;
    println!(
        "{} steps/epoca x {} epocas = {} steps | lr pico {} | wd {} | cutout {} | smooth {}",
        steps_per_epoch, cfg.epochs, total_steps, cfg.lr, cfg.wd, cfg.cutout, cfg.smooth
    );

    let mut rng = Rng::new(cfg.seed ^ 0xABCD);
    let mut order: Vec<usize> = (0..ntrain).collect();
    let mut x = Arc::new(Tensor::zeros(cfg.batch, 3, IMG, IMG));
    let mut labels = vec![0u32; cfg.batch];
    let mut dl = vec![0.0f32; cfg.batch * NCLASS];
    let t0 = Instant::now();
    let mut step = 0usize;
    let mut best = 0.0f32;

    for epoch in 0..cfg.epochs {
        rng.shuffle(&mut order);
        let te = Instant::now();
        let (mut eloss, mut ecorrect) = (0.0f32, 0usize);

        for b in 0..steps_per_epoch {
            let idx = &order[b * cfg.batch..(b + 1) * cfg.batch];
            data::make_batch_train(
                &train,
                idx,
                cfg.pad,
                cfg.cutout,
                rng.next_u64(),
                &mut Arc::make_mut(&mut x).d[..cfg.batch * PIX],
                &mut labels,
            );
            let logits = net.forward(&x, true);
            let (l, c) = softmax_ce(&logits, &labels, NCLASS, cfg.smooth, &mut dl);
            net.backward(&dl);
            net.step(lr_at(&cfg, step, total_steps, warm_steps), cfg.momentum, cfg.wd);

            eloss += l;
            ecorrect += c;
            step += 1;

            if b % 50 == 0 {
                let done = step as f64 / total_steps as f64;
                let eta = t0.elapsed().as_secs_f64() * (1.0 - done) / done.max(1e-9);
                print!(
                    "\r  epoca {:>3}  [{:>4}/{}]  loss {:.3}  acc {:.1}%  lr {:.4}  ETA {}   ",
                    epoch + 1,
                    b,
                    steps_per_epoch,
                    eloss / (b + 1) as f32,
                    100.0 * ecorrect as f32 / ((b + 1) * cfg.batch) as f32,
                    lr_at(&cfg, step, total_steps, warm_steps),
                    fmt_hms(eta)
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }

        let (tl, ta) = evaluate(&mut net, &test, cfg.batch);
        if ta > best {
            best = ta;
            if let Some(p) = &cfg.save {
                let _ = net.save(p);
            }
        }
        println!(
            "\repoca {:>3}/{}  train loss {:.4} acc {:.2}%  |  teste loss {:.4} acc {:.2}%  (best {:.2}%)  {:.1}s  total {}",
            epoch + 1,
            cfg.epochs,
            eloss / steps_per_epoch as f32,
            100.0 * ecorrect as f32 / (steps_per_epoch * cfg.batch) as f32,
            tl,
            ta * 100.0,
            best * 100.0,
            te.elapsed().as_secs_f64(),
            fmt_hms(t0.elapsed().as_secs_f64())
        );
    }

    println!("melhor acuracia de teste: {:.2}%", best * 100.0);
    if let Some(p) = &cfg.save {
        println!("melhores pesos em {p}");
    }
}
