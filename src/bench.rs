//! Cronometragem. Nao valida nada: existe para medir, e o `regress.sh` compara
//! o resultado com `baseline.txt`.

use crate::data::{self, Aug, IMG, NCLASS, PIX};
use crate::model::ResNet;
use crate::nn::{softmax_ce, Param, Tensor};
use crate::rng::{rand_vec, Rng};
use crate::train::{self, Trainer};
use crate::vit::RustViT;
use crate::{gemm, optim};
use std::sync::Arc;
use std::time::Instant;

/// Cronometra o driver de GEMM nas formas exatas que o modelo usa.
pub fn gemm() {
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

fn report(reps: usize, batch: usize, tot: f64, fwd: f64, bwd: f64, opt: f64, gflop: f64) {
    println!(
        "  fwd {:.0} ms | bwd {:.0} ms | opt {:.0} ms",
        1000.0 * fwd / reps as f64,
        1000.0 * bwd / reps as f64,
        1000.0 * opt / reps as f64
    );
    let imgs = (reps * batch) as f64;
    println!(
        "bench: {:.1} img/s | step {:.0} ms | {:.1} GFLOP/s | epoca estimada {:.0}s",
        imgs / tot,
        1000.0 * tot / reps as f64,
        gflop * imgs / tot / 1e9,
        50000.0 / (imgs / tot)
    );
}

/// Cronometra o passo do rustvit com lr e wd zerados: mede o custo, nao o
/// treino. Fica fora de `Trainer::step` porque precisa cronometrar forward,
/// backward e otimizador separadamente.
pub fn vit(tr: &mut Trainer, macs: u64) {
    let (cfg, taxo, ds) = (tr.cfg, tr.taxo, tr.train);
    let reps = cfg.bench;
    let (aug, _, _) = train::aug_at(cfg, cfg.epochs / 2);
    let ord: Vec<usize> = (0..cfg.batch).collect();
    {
        let xm = Arc::make_mut(&mut tr.x);
        data::make_batch_train(ds, &ord, &aug, 1, &mut xm.d, &mut tr.labels, &mut tr.coarse);
    }
    let lam = vec![1.0f32; cfg.batch];
    train::targets(
        &tr.labels, &tr.coarse, &ord, &lam, taxo, cfg, &mut tr.tf, &mut tr.tc,
    );

    let (mut fwd, mut bwd, mut opt_t) = (0.0f64, 0.0f64, 0.0f64);
    let (x, tf, tc, labels, coarse) = (&tr.x, &tr.tf, &tr.tc, &tr.labels, &tr.coarse);
    let mut run = |net: &mut RustViT, opt: &mut optim::Optim, acc: bool| {
        let a = Instant::now();
        let out = net.forward(x, true);
        let lo = train::build_loss(&out, tf, tc, labels, coarse, cfg, NCLASS, taxo.ncoarse, cfg.batch);
        if acc {
            fwd += a.elapsed().as_secs_f64();
        }
        let a = Instant::now();
        net.backward(&lo.g);
        if acc {
            bwd += a.elapsed().as_secs_f64();
        }
        let a = Instant::now();
        let mut ps: Vec<&mut Param> = Vec::new();
        net.for_each_param(&mut |p| ps.push(unsafe { &mut *(p as *mut Param) }));
        opt.step_all(&mut ps, 0.0, 0.0);
        if acc {
            opt_t += a.elapsed().as_secs_f64();
        }
    };
    for _ in 0..2 {
        run(&mut tr.net, &mut tr.opt, false);
    }
    let t = Instant::now();
    for _ in 0..reps {
        run(&mut tr.net, &mut tr.opt, true);
    }
    let tot = t.elapsed().as_secs_f64();
    report(reps, cfg.batch, tot, fwd, bwd, opt_t, macs as f64 * 6.0);
}

pub fn resnet(net: &mut ResNet, cfg: &crate::cli::Config, ds: &data::Dataset, aug: &Aug) {
    let reps = cfg.bench;
    let mut x = Arc::new(Tensor::zeros(cfg.batch, 3, IMG, IMG));
    let mut labels = vec![0u32; cfg.batch];
    let mut coarse = vec![0u32; cfg.batch];
    let mut dl = vec![0.0f32; cfg.batch * NCLASS];
    let idx: Vec<usize> = (0..cfg.batch).collect();
    data::make_batch_train(
        ds,
        &idx,
        aug,
        cfg.seed,
        &mut Arc::make_mut(&mut x).d[..cfg.batch * PIX],
        &mut labels,
        &mut coarse,
    );
    let (mut fwd, mut bwd, mut opt_t) = (0.0f64, 0.0f64, 0.0f64);
    let t = Instant::now();
    for _ in 0..reps {
        let a = Instant::now();
        let lg = net.forward(&x, true);
        softmax_ce(&lg, &labels, NCLASS, cfg.smooth, &mut dl);
        fwd += a.elapsed().as_secs_f64();
        let a = Instant::now();
        net.backward(&dl);
        bwd += a.elapsed().as_secs_f64();
        let a = Instant::now();
        net.step(0.0, 0.0, 0.0);
        opt_t += a.elapsed().as_secs_f64();
    }
    let tot = t.elapsed().as_secs_f64();
    let gflop = net.flops(IMG, IMG) as f64 * 3.0;
    report(reps, cfg.batch, tot, fwd, bwd, opt_t, gflop);
}
