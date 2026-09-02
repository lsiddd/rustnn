// Verificacao numerica do backward de toda camada, do modelo inteiro e do
// driver de GEMM: 50 checagens, todas comparando a derivada analitica com a
// diferenca central.
//
// Vem por `include!` e nao por `mod` porque as checagens montam o modelo com os
// mesmos `Config` e `make_cfg` do treino, que sao privados de `main.rs`: um
// modulo separado exigiria torna-los publicos so para o teste, e a checagem
// deixaria de exercitar exatamente o caminho que roda de verdade.

/// Derivada direcional: compara (L(x+ev)-L(x-ev))/2e com <g, v> numa direcao
/// aleatoria. Agrega o gradiente inteiro num numero so, entao o sinal fica bem
/// acima do ruido de arredondamento do f32 mesmo em tensores grandes.
fn dircheck(
    base: &[f32],
    grad: &[f32],
    eps: f32,
    rng: &mut Rng,
    mut lossf: impl FnMut(&[f32]) -> f32,
) -> f32 {
    let n = base.len();
    let v: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
    let mut c = base.to_vec();
    for i in 0..n {
        c[i] = base[i] + eps * v[i];
    }
    let lp = lossf(&c);
    for i in 0..n {
        c[i] = base[i] - eps * v[i];
    }
    let lm = lossf(&c);
    let num = (lp - lm) / (2.0 * eps);
    let ana: f32 = grad.iter().zip(&v).map(|(a, b)| a * b).sum();
    (num - ana).abs() / (num.abs() + ana.abs() + 1e-6)
}

fn report(name: &str, err: f32) -> bool {
    report_tol(name, err, 5e-3)
}

fn report_tol(name: &str, err: f32, tol: f32) -> bool {
    let ok = err < tol;
    println!(
        "  {name:<30} erro relativo {err:.6}  {}",
        if ok { "OK" } else { "FALHOU" }
    );
    ok
}

fn gc_eps() -> f32 {
    std::env::var("GC_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3e-4)
}

fn rand_vec(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| rng.normal()).collect()
}

// ---------------------------------------------------------------- rustvit

/// Empacota os parametros de uma camada num vetor so, roda o dircheck sobre
/// todos de uma vez e restaura. Um unico numero cobre a camada inteira.
///
/// E macro e nao funcao porque as duas closures (a que escreve os parametros e
/// a que calcula a perda) precisariam do mesmo emprestimo mutavel.
macro_rules! zero_grads {
    ($obj:expr) => {
        $obj.for_each_param(&mut |p: &mut Param| p.g.iter_mut().for_each(|v| *v = 0.0))
    };
}

macro_rules! check_params {
    ($all:expr, $tag:expr, $obj:expr, $eps:expr, $rng:expr, $loss:expr) => {{
        let (mut base, mut grad) = (Vec::new(), Vec::new());
        $obj.for_each_param(&mut |p: &mut Param| {
            base.extend_from_slice(&p.v);
            grad.extend_from_slice(&p.g);
        });
        let e = dircheck(&base, &grad, $eps, $rng, |c| {
            let mut off = 0;
            $obj.for_each_param(&mut |p: &mut Param| {
                let n = p.v.len();
                p.v.copy_from_slice(&c[off..off + n]);
                off += n;
            });
            $loss
        });
        let mut off = 0;
        $obj.for_each_param(&mut |p: &mut Param| {
            let n = p.v.len();
            p.v.copy_from_slice(&base[off..off + n]);
            off += n;
        });
        $all &= report($tag, e);
    }};
}

fn naive_matmul(m: usize, n: usize, k: usize, a: &[f32], at: bool, b: &[f32], bt: bool) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                let av = if at { a[p * m + i] } else { a[i * k + p] };
                let bv = if bt { b[j * k + p] } else { b[p * n + j] };
                s += av * bv;
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn vitcheck(seed: u64) -> bool {
    use attn::{CrossAttn, SelfAttn};
    use norm::{DwConv3, LayerNorm, SeqPool, SwiGlu};
    use tok::Tok;
    let mut rng = Rng::new(seed ^ 0x1234);
    let eps = gc_eps();
    let mut all = true;
    println!("gradcheck do rustvit (eps {eps}):");

    // 1. o GEMM novo contra o laco triplo escalar, nas quatro combinacoes
    for (m, n, k) in [(23usize, 37usize, 19usize), (64, 48, 96), (7, 16, 259)] {
        for at in [false, true] {
            for bt in [false, true] {
                let a = rand_vec(m * k, &mut rng);
                let b = rand_vec(k * n, &mut rng);
                let want = naive_matmul(m, n, k, &a, at, &b, bt);
                let mut got = vec![0.0f32; m * n];
                gemm::matmul(m, n, k, &a, at, &b, bt, &mut got, false);
                let err = want
                    .iter()
                    .zip(&got)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
                    / want.iter().map(|v| v.abs()).fold(1e-6, f32::max);
                all &= report(&format!("matmul {m}x{n}x{k} at{} bt{}", at as u8, bt as u8), err);
            }
        }
    }

    let (n, t, c) = (3usize, 9usize, 8usize);
    let mut x = Tok::zeros(n, t, c);
    x.d = rand_vec(n * t * c, &mut rng);
    let r = rand_vec(n * t * c, &mut rng);
    let dot = |y: &Tok, r: &[f32]| y.d.iter().zip(r).map(|(a, b)| a * b).sum::<f32>();

    // 2. LayerNorm
    {
        let mut ln = LayerNorm::new(c);
        ln.g.v = rand_vec(c, &mut rng).iter().map(|v| 1.0 + 0.2 * v).collect();
        ln.b.v = rand_vec(c, &mut rng);
        let mut dy = x.zero_like();
        dy.d.copy_from_slice(&r);
        let _ = ln.forward(&x, true);
        let dx = ln.backward(&dy);
        let mut xt = x.clone();
        let e = dircheck(&x.d, &dx.d, eps, &mut rng, |v| {
            xt.d.copy_from_slice(v);
            dot(&ln.forward(&xt, true), &r)
        });
        all &= report("layernorm dx", e);
        zero_grads!(ln);
        let _ = ln.forward(&x, true);
        let _ = ln.backward(&dy);
        check_params!(all, "layernorm dparams", ln, eps, &mut rng, dot(&ln.forward(&x, true), &r));
    }

    // 3. SwiGLU
    {
        let mut sw = SwiGlu::new(c, 16, 2, &mut rng);
        let mut dy = x.zero_like();
        dy.d.copy_from_slice(&r);
        let _ = sw.forward(&x, true);
        let dx = sw.backward(&dy);
        let mut xt = x.clone();
        let e = dircheck(&x.d, &dx.d, eps, &mut rng, |v| {
            xt.d.copy_from_slice(v);
            dot(&sw.forward(&xt, true), &r)
        });
        all &= report("swiglu dx", e);
        zero_grads!(sw);
        let _ = sw.forward(&x, true);
        let _ = sw.backward(&dy);
        check_params!(all, "swiglu dparams", sw, eps, &mut rng, dot(&sw.forward(&x, true), &r));
    }

    // 4. depthwise 3x3 channel-last
    {
        let mut dw = DwConv3::new(c, 3, 3, &mut rng);
        dw.b.v = rand_vec(c, &mut rng);
        let mut dy = x.zero_like();
        dy.d.copy_from_slice(&r);
        let _ = dw.forward(&x, true);
        let dx = dw.backward(&dy);
        let mut xt = x.clone();
        let e = dircheck(&x.d, &dx.d, eps, &mut rng, |v| {
            xt.d.copy_from_slice(v);
            dot(&dw.forward(&xt, true), &r)
        });
        all &= report("dwconv3x3 dx", e);
        zero_grads!(dw);
        let _ = dw.forward(&x, true);
        let _ = dw.backward(&dy);
        check_params!(all, "dwconv3x3 dparams", dw, eps, &mut rng, dot(&dw.forward(&x, true), &r));
    }

    // 5. SeqPool
    {
        let mut sp = SeqPool::new(c, &mut rng);
        let rz = rand_vec(n * c, &mut rng);
        let dotz = |z: &[f32]| z.iter().zip(&rz).map(|(a, b)| a * b).sum::<f32>();
        let _ = sp.forward(&x, true);
        let dx = sp.backward(&rz);
        let mut xt = x.clone();
        let e = dircheck(&x.d, &dx.d, eps, &mut rng, |v| {
            xt.d.copy_from_slice(v);
            dotz(&sp.forward(&xt, true))
        });
        all &= report("seqpool dx", e);
        zero_grads!(sp);
        let _ = sp.forward(&x, true);
        let _ = sp.backward(&rz);
        check_params!(all, "seqpool dparams", sp, eps, &mut rng, dotz(&sp.forward(&x, true)));
    }

    // 6. auto-atencao com LSA e vies relativo
    {
        let mut at = SelfAttn::new(c, 2, 3, 3, 2, &mut rng);
        at.rel.v = rand_vec(at.rel.v.len(), &mut rng);
        let mut dy = x.zero_like();
        dy.d.copy_from_slice(&r);
        let _ = at.forward(&x, true);
        let dx = at.backward(&dy);
        let mut xt = x.clone();
        let e = dircheck(&x.d, &dx.d, eps, &mut rng, |v| {
            xt.d.copy_from_slice(v);
            dot(&at.forward(&xt, true), &r)
        });
        all &= report("selfattn dx", e);
        zero_grads!(at);
        let _ = at.forward(&x, true);
        let _ = at.backward(&dy);
        check_params!(all, "selfattn dparams", at, eps, &mut rng, dot(&at.forward(&x, true), &r));
    }

    // 7. atencao cruzada na memoria de alta resolucao
    {
        let (mh, mw) = (6usize, 6usize);
        let mut cr = CrossAttn::new(c, 2, 3, 3, mh, mw, 2, &mut rng);
        cr.rel.v = rand_vec(cr.rel.v.len(), &mut rng);
        let mem = rand_vec(n * mh * mw * 2 * c, &mut rng);
        let mut dy = x.zero_like();
        dy.d.copy_from_slice(&r);
        let mut dmem = vec![0.0f32; mem.len()];
        let _ = cr.forward(&x, &mem, true);
        let dx = cr.backward(&dy, &mem, &mut dmem);
        let mut xt = x.clone();
        let e = dircheck(&x.d, &dx.d, eps, &mut rng, |v| {
            xt.d.copy_from_slice(v);
            dot(&cr.forward(&xt, &mem, true), &r)
        });
        all &= report("crossattn dx", e);
        let e = dircheck(&mem, &dmem, eps, &mut rng, |v| {
            dot(&cr.forward(&x, v, true), &r)
        });
        all &= report("crossattn dmem", e);
        let mut dm2 = vec![0.0f32; mem.len()];
        zero_grads!(cr);
        let _ = cr.forward(&x, &mem, true);
        let _ = cr.backward(&dy, &mem, &mut dm2);
        check_params!(all, "crossattn dparams", cr, eps, &mut rng, dot(&cr.forward(&x, &mem, true), &r));
    }

    // 8. bloco completo, com memoria e stochastic depth de semente fixa
    {
        let (mh, mw) = (6usize, 6usize);
        let mut blk = vit::Block::new(c, 2, 3, 3, Some((mh, mw)), 2, 0.2, 7, 0.3, &mut rng);
        blk.for_each_param(&mut |p| {
            p.v.iter_mut().for_each(|v| *v += 0.2 * rng.normal());
        });
        let mem = rand_vec(n * mh * mw * 2 * c, &mut rng);
        let mut dmem = vec![0.0f32; mem.len()];
        let mut dy = x.zero_like();
        dy.d.copy_from_slice(&r);
        let _ = blk.forward(x.clone(), Some(&mem), true, 99);
        let dx = blk.backward(dy.clone(), Some(&mem), Some(&mut dmem));
        let mut xt = x.clone();
        let e = dircheck(&x.d, &dx.d, eps, &mut rng, |v| {
            xt.d.copy_from_slice(v);
            dot(&blk.forward(xt.clone(), Some(&mem), true, 99), &r)
        });
        all &= report("block dx", e);
        let mut dm2 = vec![0.0f32; mem.len()];
        zero_grads!(blk);
        let _ = blk.forward(x.clone(), Some(&mem), true, 99);
        let _ = blk.backward(dy, Some(&mem), Some(&mut dm2));
        check_params!(all, "block dparams", blk, eps, &mut rng, dot(&blk.forward(x.clone(), Some(&mem), true, 99), &r));
    }

    // 9. rede inteira, todos os parametros de uma vez
    {
        let parent: Vec<u32> = vec![0, 0, 1, 1, 2, 2];
        let cfg = Cfg {
            width: 4,
            cmid: 6,
            dim: 8,
            heads: 2,
            depth3: 2,
            dim4: 12,
            heads4: 2,
            depth4: 1,
            mem_at: vec![1],
            droppath: 0.2,
            img: 16,
        };
        let nf = parent.len();
        let mut net = RustViT::new(cfg, parent, 3, seed ^ 0xBEEF);
        net.drop_seed = 4242;
        net.for_each_param(&mut |p| p.v.iter_mut().for_each(|v| *v += 0.15 * rng.normal()));
        let bs = 3usize;
        let mut x0 = Tensor::zeros(bs, 3, 16, 16);
        x0.d = rand_vec(bs * 3 * 16 * 16, &mut rng);
        let xa = Arc::new(x0);

        let g = Grads {
            logits: rand_vec(bs * nf, &mut rng),
            coarse: rand_vec(bs * 3, &mut rng),
            e1: rand_vec(bs * nf, &mut rng),
            e2: rand_vec(bs * nf, &mut rng),
            h1: rand_vec(bs * 12, &mut rng),
            h2: rand_vec(bs * 12, &mut rng),
            hfinal: rand_vec(bs * 12, &mut rng),
        };
        // A perda e a forma linear <saida, g>, entao dL/dsaida = g exatamente.
        // Isso exercita as tres cabecas, os dois hints e a memoria de uma vez.
        let lin = |o: &Out, g: &Grads| -> f32 {
            let d = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
            d(&o.logits, &g.logits)
                + d(&o.coarse, &g.coarse)
                + d(&o.e1, &g.e1)
                + d(&o.e2, &g.e2)
                + d(&o.h1, &g.h1)
                + d(&o.h2, &g.h2)
                + d(&o.feat, &g.hfinal)
        };
        let out = net.forward(&xa, true);
        let _ = &out;
        net.backward(&g);
        check_params!(all, "rustvit completo (todos)", net, eps, &mut rng, lin(&net.forward(&xa, true), &g));
    }

    // 10. Newton-Schulz: as saidas devem ter valores singulares perto de 1,
    //     o que equivale a X^T X proximo da identidade.
    {
        for (m, n) in [(12usize, 20usize), (20, 12), (16, 16)] {
            let mut g = rand_vec(m * n, &mut rng);
            optim::newton_schulz(&mut g, m, n, 5, &mut optim::NsScratch::default());
            let k = m.min(n);
            let mut err = 0.0f32;
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0f32;
                    for p in 0..m.max(n) {
                        let (a, b) = if m <= n {
                            (g[i * n + p], g[j * n + p])
                        } else {
                            (g[p * n + i], g[p * n + j])
                        };
                        s += a * b;
                    }
                    err = err.max((s - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            // O quintico do Muon e deliberadamente relaxado: os valores
            // singulares ficam numa banda em torno de 1, nao exatamente em 1.
            // O que precisa valer e que nenhum saia da banda.
            all &= report_tol(&format!("newton-schulz {m}x{n} banda"), err, 0.5);
        }
    }
    all
}

// ---------------------------------------------------------------- resnet

/// Gradcheck por camada do caminho convolucional.
fn layercheck(seed: u64) -> bool {
    use nn::{BatchNorm2d, Conv2d, Linear};
    let mut rng = Rng::new(seed);
    let eps = gc_eps();
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

    // perda com alvos suaves, o caminho que a taxonomia e o mixup usam
    {
        let (n, nc) = (4usize, 7usize);
        let z: Vec<f32> = (0..n * nc).map(|_| rng.normal()).collect();
        let labels: Vec<u32> = (0..n).map(|_| rng.below(nc) as u32).collect();
        let mut t = vec![0.0f32; n * nc];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..nc {
                let v = rng.uniform() + 0.1;
                t[i * nc + j] = v;
                s += v;
            }
            for j in 0..nc {
                t[i * nc + j] /= s;
            }
        }
        let mut d = vec![0.0f32; n * nc];
        loss::ce_soft(&z, &t, &labels, nc, 1.0, &mut d);
        let mut tmp = vec![0.0f32; n * nc];
        let e = dircheck(&z, &d, eps, &mut rng, |c| {
            loss::ce_soft(c, &t, &labels, nc, 1.0, &mut tmp).0
        });
        all &= report("ce_soft dlogits", e);

        let teach: Vec<f32> = (0..n * nc).map(|_| rng.normal()).collect();
        let mut d = vec![0.0f32; n * nc];
        loss::kl_distill(&z, &teach, nc, 2.0, 1.0, &mut d);
        let mut tmp = vec![0.0f32; n * nc];
        let e = dircheck(&z, &d, eps, &mut rng, |c| {
            tmp.iter_mut().for_each(|v| *v = 0.0);
            loss::kl_distill(c, &teach, nc, 2.0, 1.0, &mut tmp)
        });
        all &= report("kl_distill dlogits", e);
    }
    all
}

/// Gradcheck da ResNet completa.
fn gradcheck(seed: u64) -> bool {
    const NC: usize = 10;
    const BS: usize = 8;
    let eps = gc_eps();
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
    report("resnet completa (todos)", e)
}
