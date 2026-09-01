//! Camadas: Conv2d (im2col + GEMM), BatchNorm2d, Linear, pooling e loss.

use crate::conv::{self, ConvSpec, RowMap};
use crate::gemm::{PackedA, KC, MR, NR};
use crate::rng::Rng;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor {
    pub d: Vec<f32>,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Tensor {
    pub fn zeros(n: usize, c: usize, h: usize, w: usize) -> Tensor {
        Tensor {
            d: vec![0.0; n * c * h * w],
            n,
            c,
            h,
            w,
        }
    }
    #[inline]
    pub fn img(&self) -> usize {
        self.c * self.h * self.w
    }
    pub fn resize(&mut self, n: usize, c: usize, h: usize, w: usize) {
        self.n = n;
        self.c = c;
        self.h = h;
        self.w = w;
        self.d.resize(n * c * h * w, 0.0);
    }
}

/// Parametro treinavel: valor, gradiente e momento.
pub struct Param {
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub m: Vec<f32>,
    pub decay: bool,
}

impl Param {
    pub fn new(n: usize, decay: bool) -> Param {
        Param {
            v: vec![0.0; n],
            g: vec![0.0; n],
            m: vec![0.0; n],
            decay,
        }
    }
    pub fn filled(n: usize, val: f32, decay: bool) -> Param {
        let mut p = Param::new(n, decay);
        p.v.iter_mut().for_each(|x| *x = val);
        p
    }
    pub fn kaiming(fan_in: usize, n: usize, rng: &mut Rng) -> Param {
        let mut p = Param::new(n, true);
        let s = (2.0 / fan_in as f32).sqrt();
        p.v.iter_mut().for_each(|x| *x = rng.normal() * s);
        p
    }
    /// SGD com momentum de Nesterov e weight decay desacoplado do gradiente medio.
    pub fn step(&mut self, lr: f32, mom: f32, wd: f32) {
        let wd = if self.decay { wd } else { 0.0 };
        for i in 0..self.v.len() {
            let g = self.g[i] + wd * self.v[i];
            self.m[i] = mom * self.m[i] + g;
            self.v[i] -= lr * (g + mom * self.m[i]);
            self.g[i] = 0.0;
        }
    }
}

// ---------------------------------------------------------------- Conv2d

pub struct Conv2d {
    pub w: Param, // [cout, cin*k*k]
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    /// Cache da entrada para o backward. E um `Arc` porque a mesma ativacao e
    /// entrada de mais de uma camada (a conv do ramo residual e a do atalho, por
    /// exemplo): copiar cada tensor de entrada custava ~380 MB de trafego por
    /// step neste modelo.
    x: Arc<Tensor>,
    wpack: PackedA,
    wtpack: PackedA,
}

impl Conv2d {
    pub fn new(cin: usize, cout: usize, k: usize, stride: usize, pad: usize, rng: &mut Rng) -> Self {
        let fan_in = cin * k * k;
        Conv2d {
            w: Param::kaiming(fan_in, cout * fan_in, rng),
            cin,
            cout,
            k,
            stride,
            pad,
            x: Arc::new(Tensor::zeros(0, 0, 0, 0)),
            wpack: PackedA::new(),
            wtpack: PackedA::new(),
        }
    }

    #[inline]
    fn out_dim(&self, h: usize) -> usize {
        (h + 2 * self.pad - self.k) / self.stride + 1
    }

    fn spec(&self, h: usize, w: usize) -> ConvSpec {
        ConvSpec {
            cin: self.cin,
            cout: self.cout,
            k: self.k,
            stride: self.stride,
            pad: self.pad,
            h,
            w,
            oh: self.out_dim(h),
            ow: self.out_dim(w),
        }
    }

    pub fn forward(&mut self, x: &Arc<Tensor>, train: bool) -> Tensor {
        let sp = self.spec(x.h, x.w);
        let ckk = sp.ckk();
        self.wpack.pack(&self.w.v, self.cout, ckk);
        let mut y = Tensor::zeros(x.n, self.cout, sp.oh, sp.ow);
        let wpack = &self.wpack;
        let rm = RowMap::new(&sp);
        let xstride = self.cin * x.h * x.w;

        y.d.par_chunks_mut(self.cout * sp.hw())
            .zip(x.d.par_chunks(xstride))
            .for_each_init(
                || vec![0.0f32; KC * NR],
                |bp, (yi, xi)| conv::forward_image(&sp, &rm, wpack, xi, yi, bp),
            );
        if train {
            self.x = x.clone();
        }
        y
    }

    /// Acumula dW.
    ///
    /// Cada worker soma num `dw` privado e a reducao vem depois: sem atomics no
    /// caminho quente. Limitar o numero desses buffers com `with_min_len` foi
    /// medido e sai mais caro: o balanceamento que o rayon consegue dividindo
    /// livremente vale mais que as alocacoes economizadas.
    pub fn backward_w(&mut self, dy: &Tensor) {
        let x = &self.x;
        let sp = self.spec(x.h, x.w);
        let (ckk, hw) = (sp.ckk(), sp.hw());
        let cout = self.cout;
        let xstride = self.cin * x.h * x.w;
        let rm = RowMap::new(&sp);

        let dw = dy
            .d
            .par_chunks(cout * hw)
            .zip(x.d.par_chunks(xstride))
            .fold(
                || {
                    (
                        vec![0.0f32; cout * ckk],
                        vec![0.0f32; 2 * KC * NR],
                        PackedA::new(),
                    )
                },
                |(mut acc, mut scratch, mut dyp), (dyi, xi)| {
                    let (tmp, bp) = scratch.split_at_mut(KC * NR);
                    conv::dw_image(&sp, &rm, &mut dyp, dyi, xi, &mut acc, tmp, bp);
                    (acc, scratch, dyp)
                },
            )
            .map(|(acc, ..)| acc)
            .reduce(
                || vec![0.0f32; cout * ckk],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            );
        self.w.g.iter_mut().zip(&dw).for_each(|(a, b)| *a += b);
    }

    /// Retorna dX e acumula dW.
    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        self.backward_w(dy);

        let x = &self.x;
        let sp = self.spec(x.h, x.w);
        let (ckk, hw) = (sp.ckk(), sp.hw());
        let (cout, cin) = (self.cout, self.cin);
        let xstride = cin * x.h * x.w;
        let rm = RowMap::new(&sp);

        self.wtpack.pack_t(&self.w.v, ckk, cout);
        let wtpack = &self.wtpack;
        let mut dx = Tensor::zeros(x.n, cin, x.h, x.w);
        dx.d.par_chunks_mut(xstride)
            .zip(dy.d.par_chunks(cout * hw))
            .for_each_init(
                || (vec![0.0f32; KC * NR], vec![0.0f32; MR * NR]),
                |(bp, tile), (dxi, dyi)| conv::dx_image(&sp, &rm, wtpack, dyi, dxi, bp, tile),
            );
        dx
    }
}

// ---------------------------------------------------------------- BatchNorm

pub struct BatchNorm2d {
    pub g: Param,
    pub b: Param,
    pub rm: Vec<f32>,
    pub rv: Vec<f32>,
    xhat: Tensor,
    invstd: Vec<f32>,
    mom: f32,
    eps: f32,
}

impl BatchNorm2d {
    pub fn new(c: usize) -> Self {
        BatchNorm2d {
            g: Param::filled(c, 1.0, false),
            b: Param::new(c, false),
            rm: vec![0.0; c],
            rv: vec![1.0; c],
            xhat: Tensor::zeros(0, 0, 0, 0),
            invstd: vec![0.0; c],
            mom: 0.1,
            eps: 1e-5,
        }
    }

    /// Zera gamma (truque do "zero-init residual" no ultimo BN de cada bloco).
    pub fn zero_gamma(&mut self) {
        self.g.v.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Normaliza no proprio tensor de entrada.
    ///
    /// A saida da convolucao nao e usada em lugar nenhum depois do BN (o
    /// backward da conv precisa de X, nao de Y), entao consumir o tensor evita
    /// uma alocacao e uma passada de leitura+escrita por camada.
    pub fn forward(&mut self, x: Tensor, train: bool) -> Tensor {
        self.run(x, train, false)
    }

    /// Como `forward`, com a ReLU aplicada no mesmo passe.
    pub fn forward_relu(&mut self, x: Tensor, train: bool) -> Tensor {
        self.run(x, train, true)
    }

    fn run(&mut self, x: Tensor, train: bool, relu: bool) -> Tensor {
        let (n, c, h, w) = (x.n, x.c, x.h, x.w);
        let hw = h * w;
        let m = (n * hw) as f32;
        let mut y = x;

        if train {
            let mut mean = vec![0.0f32; c];
            let mut var = vec![0.0f32; c];
            mean.par_iter_mut()
                .zip(var.par_iter_mut())
                .enumerate()
                .for_each(|(ci, (mu, va))| {
                    let (mut s, mut s2) = (0.0f64, 0.0f64);
                    for i in 0..n {
                        let base = i * c * hw + ci * hw;
                        for p in 0..hw {
                            let v = y.d[base + p] as f64;
                            s += v;
                            s2 += v * v;
                        }
                    }
                    let mu_ = s / m as f64;
                    *mu = mu_ as f32;
                    *va = (s2 / m as f64 - mu_ * mu_).max(0.0) as f32;
                });
            for ci in 0..c {
                self.invstd[ci] = 1.0 / (var[ci] + self.eps).sqrt();
                self.rm[ci] = (1.0 - self.mom) * self.rm[ci] + self.mom * mean[ci];
                // variancia sem vies para as estatisticas de inferencia
                let unb = var[ci] * m / (m - 1.0).max(1.0);
                self.rv[ci] = (1.0 - self.mom) * self.rv[ci] + self.mom * unb;
            }
            let (gv, bv, ist) = (&self.g.v, &self.b.v, &self.invstd);
            self.xhat.resize(n, c, h, w);
            self.xhat
                .d
                .par_chunks_mut(hw)
                .zip(y.d.par_chunks_mut(hw))
                .enumerate()
                .for_each(|(idx, (xh_, yc))| {
                    let ci = idx % c;
                    let (mu, is, g, b) = (mean[ci], ist[ci], gv[ci], bv[ci]);
                    for p in 0..hw {
                        let xh = (yc[p] - mu) * is;
                        xh_[p] = xh;
                        let v = g * xh + b;
                        yc[p] = if relu { v.max(0.0) } else { v };
                    }
                });
        } else {
            let (gv, bv, rm, rv, eps) = (&self.g.v, &self.b.v, &self.rm, &self.rv, self.eps);
            y.d.par_chunks_mut(hw).enumerate().for_each(|(idx, yc)| {
                let ci = idx % c;
                let is = 1.0 / (rv[ci] + eps).sqrt();
                let (sc, sh) = (gv[ci] * is, bv[ci] - gv[ci] * is * rm[ci]);
                for p in 0..hw {
                    let v = sc * yc[p] + sh;
                    yc[p] = if relu { v.max(0.0) } else { v };
                }
            });
        }
        y
    }

    pub fn backward(&mut self, dy: &Tensor) -> Tensor {
        let (n, c, h, w) = (dy.n, dy.c, dy.h, dy.w);
        let hw = h * w;
        let m = (n * hw) as f32;
        let mut dgamma = vec![0.0f32; c];
        let mut dbeta = vec![0.0f32; c];

        dgamma
            .par_iter_mut()
            .zip(dbeta.par_iter_mut())
            .enumerate()
            .for_each(|(ci, (dg, db))| {
                let (mut a, mut b) = (0.0f32, 0.0f32);
                for i in 0..n {
                    let base = i * c * hw + ci * hw;
                    for p in 0..hw {
                        a += dy.d[base + p] * self.xhat.d[base + p];
                        b += dy.d[base + p];
                    }
                }
                *dg = a;
                *db = b;
            });

        let mut dx = Tensor::zeros(n, c, h, w);
        let (gv, ist) = (&self.g.v, &self.invstd);
        let (dgr, dbr) = (&dgamma, &dbeta);
        dx.d.par_chunks_mut(hw)
            .zip(dy.d.par_chunks(hw))
            .zip(self.xhat.d.par_chunks(hw))
            .enumerate()
            .for_each(|(idx, ((dxc, dyc), xh))| {
                let ci = idx % c;
                let s = gv[ci] * ist[ci] / m;
                let (dg, db) = (dgr[ci], dbr[ci]);
                for p in 0..hw {
                    dxc[p] = s * (m * dyc[p] - db - xh[p] * dg);
                }
            });

        self.g.g.iter_mut().zip(&dgamma).for_each(|(a, b)| *a += b);
        self.b.g.iter_mut().zip(&dbeta).for_each(|(a, b)| *a += b);
        dx
    }
}

// ---------------------------------------------------------------- ReLU / add

/// dx *= (y > 0), onde y e a saida da ReLU.
pub fn relu_back_(d: &mut Tensor, y: &Tensor) {
    d.d.par_iter_mut()
        .zip(y.d.par_iter())
        .for_each(|(g, &v)| {
            if v <= 0.0 {
                *g = 0.0;
            }
        });
}

pub fn add_(a: &mut Tensor, b: &Tensor) {
    a.d.par_iter_mut().zip(b.d.par_iter()).for_each(|(x, y)| *x += y);
}

/// a = max(0, a + b): a soma residual e a ReLU num passe so.
pub fn add_relu_(a: &mut Tensor, b: &Tensor) {
    a.d.par_iter_mut()
        .zip(b.d.par_iter())
        .for_each(|(x, y)| *x = (*x + y).max(0.0));
}

// ---------------------------------------------------------------- pooling

/// Global average pooling: [n,c,h,w] -> [n,c,1,1]
pub fn gap_forward(x: &Tensor) -> Tensor {
    let hw = x.h * x.w;
    let mut y = Tensor::zeros(x.n, x.c, 1, 1);
    y.d.par_iter_mut().enumerate().for_each(|(i, o)| {
        let s: f32 = x.d[i * hw..(i + 1) * hw].iter().sum();
        *o = s / hw as f32;
    });
    y
}

pub fn gap_backward(dy: &Tensor, h: usize, w: usize) -> Tensor {
    let hw = h * w;
    let mut dx = Tensor::zeros(dy.n, dy.c, h, w);
    dx.d.par_chunks_mut(hw).enumerate().for_each(|(i, ch)| {
        let v = dy.d[i] / hw as f32;
        ch.iter_mut().for_each(|x| *x = v);
    });
    dx
}

// ---------------------------------------------------------------- Linear

pub struct Linear {
    pub w: Param, // [out, in]
    pub b: Param,
    pub fin: usize,
    pub fout: usize,
    x: Vec<f32>,
    n: usize,
}

impl Linear {
    pub fn new(fin: usize, fout: usize, rng: &mut Rng) -> Self {
        let mut w = Param::kaiming(fin, fin * fout, rng);
        // escala menor no classificador ajuda a estabilizar o inicio
        w.v.iter_mut().for_each(|x| *x *= 0.5);
        Linear {
            w,
            b: Param::new(fout, false),
            fin,
            fout,
            x: Vec::new(),
            n: 0,
        }
    }

    pub fn forward(&mut self, x: &Tensor, train: bool) -> Vec<f32> {
        let n = x.n;
        let mut y = vec![0.0f32; n * self.fout];
        for i in 0..n {
            for o in 0..self.fout {
                let wr = &self.w.v[o * self.fin..(o + 1) * self.fin];
                let xr = &x.d[i * self.fin..(i + 1) * self.fin];
                let mut s = self.b.v[o];
                for j in 0..self.fin {
                    s += wr[j] * xr[j];
                }
                y[i * self.fout + o] = s;
            }
        }
        if train {
            self.x = x.d.clone();
            self.n = n;
        }
        y
    }

    /// dy: [n, fout] -> dx: [n, fin, 1, 1]
    pub fn backward(&mut self, dy: &[f32]) -> Tensor {
        let n = self.n;
        for i in 0..n {
            for o in 0..self.fout {
                let g = dy[i * self.fout + o];
                if g == 0.0 {
                    continue;
                }
                self.b.g[o] += g;
                let wg = &mut self.w.g[o * self.fin..(o + 1) * self.fin];
                let xr = &self.x[i * self.fin..(i + 1) * self.fin];
                for j in 0..self.fin {
                    wg[j] += g * xr[j];
                }
            }
        }
        let mut dx = Tensor::zeros(n, self.fin, 1, 1);
        for i in 0..n {
            let dxr = &mut dx.d[i * self.fin..(i + 1) * self.fin];
            for o in 0..self.fout {
                let g = dy[i * self.fout + o];
                if g == 0.0 {
                    continue;
                }
                let wr = &self.w.v[o * self.fin..(o + 1) * self.fin];
                for j in 0..self.fin {
                    dxr[j] += g * wr[j];
                }
            }
        }
        dx
    }
}

// ---------------------------------------------------------------- loss

/// Softmax + cross-entropy com label smoothing. Retorna (loss media, acertos)
/// e escreve o gradiente ja dividido pelo batch em `dlogits`.
pub fn softmax_ce(
    logits: &[f32],
    labels: &[u32],
    nclass: usize,
    smooth: f32,
    dlogits: &mut [f32],
) -> (f32, usize) {
    let n = labels.len();
    let inv = 1.0 / n as f32;
    let mut loss = 0.0f32;
    let mut correct = 0usize;
    for i in 0..n {
        let z = &logits[i * nclass..(i + 1) * nclass];
        let d = &mut dlogits[i * nclass..(i + 1) * nclass];
        let mut mx = f32::NEG_INFINITY;
        let mut arg = 0usize;
        for (j, &v) in z.iter().enumerate() {
            if v > mx {
                mx = v;
                arg = j;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..nclass {
            let e = (z[j] - mx).exp();
            d[j] = e;
            sum += e;
        }
        let lse = mx + sum.ln();
        let y = labels[i] as usize;
        if arg == y {
            correct += 1;
        }
        // alvo suavizado
        let on = 1.0 - smooth;
        let off = smooth / nclass as f32;
        loss += on * (lse - z[y]);
        if smooth > 0.0 {
            let mut s = 0.0;
            for j in 0..nclass {
                s += lse - z[j];
            }
            loss += off * s;
        }
        for j in 0..nclass {
            let t = if j == y { on + off } else { off };
            d[j] = (d[j] / sum - t) * inv;
        }
    }
    (loss * inv, correct)
}
