//! LayerNorm, SwiGLU, LayerScale, depthwise 3x3 e SeqPool sobre tokens.

use crate::gemm::scratch_vec;
use crate::linear::Linear;
use crate::nn::Param;
use crate::rng::Rng;
use crate::tok::Tok;
use rayon::prelude::*;

// ---------------------------------------------------------------- LayerNorm

/// Normaliza cada token sobre os seus `c` canais.
///
/// As somas acumulam em f64 pelo mesmo motivo do `BatchNorm2d`: com `c` na casa
/// das centenas e ativacoes de magnitude irregular, a soma de quadrados em f32
/// perde digitos justamente onde a variancia e pequena.
pub struct LayerNorm {
    pub g: Param,
    pub b: Param,
    pub c: usize,
    eps: f32,
    xhat: Vec<f32>,
    invstd: Vec<f32>,
}

impl LayerNorm {
    pub fn new(c: usize) -> LayerNorm {
        LayerNorm {
            g: Param::filled(c, 1.0, false),
            b: Param::new(c, false),
            c,
            eps: 1e-5,
            xhat: Vec::new(),
            invstd: Vec::new(),
        }
    }

    pub fn forward(&mut self, x: &Tok, train: bool) -> Tok {
        let (c, eps) = (self.c, self.eps);
        let rows = x.rows();
        let mut y = Tok::from_vec(unsafe { scratch_vec(x.d.len()) }, x.n, x.t, x.c);
        if train {
            self.xhat.resize(rows * c, 0.0);
            self.invstd.resize(rows, 0.0);
        }
        let (gv, bv) = (&self.g.v, &self.b.v);

        if train {
            y.d.par_chunks_mut(c)
                .zip(x.d.par_chunks(c))
                .zip(self.xhat.par_chunks_mut(c))
                .zip(self.invstd.par_iter_mut())
                .for_each(|(((yr, xr), xh), is)| {
                    let (mut s, mut s2) = (0.0f64, 0.0f64);
                    for &v in xr {
                        s += v as f64;
                        s2 += (v as f64) * (v as f64);
                    }
                    let m = c as f64;
                    let mu = s / m;
                    let var = (s2 / m - mu * mu).max(0.0) as f32;
                    let i = 1.0 / (var + eps).sqrt();
                    *is = i;
                    for j in 0..c {
                        let h = (xr[j] - mu as f32) * i;
                        xh[j] = h;
                        yr[j] = gv[j] * h + bv[j];
                    }
                });
        } else {
            y.d.par_chunks_mut(c).zip(x.d.par_chunks(c)).for_each(|(yr, xr)| {
                let (mut s, mut s2) = (0.0f64, 0.0f64);
                for &v in xr {
                    s += v as f64;
                    s2 += (v as f64) * (v as f64);
                }
                let m = c as f64;
                let mu = s / m;
                let var = (s2 / m - mu * mu).max(0.0) as f32;
                let i = 1.0 / (var + eps).sqrt();
                for j in 0..c {
                    yr[j] = gv[j] * ((xr[j] - mu as f32) * i) + bv[j];
                }
            });
        }
        y
    }

    pub fn backward(&mut self, dy: &Tok) -> Tok {
        let c = self.c;
        let mut dx = Tok::from_vec(unsafe { scratch_vec(dy.d.len()) }, dy.n, dy.t, dy.c);
        let gv = &self.g.v;

        // dgamma e dbeta: reducao por canal com acumulador privado por worker.
        let (dg, db) =
            dy.d.par_chunks(c)
                .zip(self.xhat.par_chunks(c))
                .fold(
                    || (vec![0.0f32; c], vec![0.0f32; c]),
                    |(mut a, mut b), (dyr, xh)| {
                        for j in 0..c {
                            a[j] += dyr[j] * xh[j];
                            b[j] += dyr[j];
                        }
                        (a, b)
                    },
                )
                .reduce(
                    || (vec![0.0f32; c], vec![0.0f32; c]),
                    |(mut a, mut b), (x, y)| {
                        a.iter_mut().zip(x).for_each(|(p, q)| *p += q);
                        b.iter_mut().zip(y).for_each(|(p, q)| *p += q);
                        (a, b)
                    },
                );
        self.g.g.iter_mut().zip(&dg).for_each(|(a, b)| *a += b);
        self.b.g.iter_mut().zip(&db).for_each(|(a, b)| *a += b);

        dx.d.par_chunks_mut(c)
            .zip(dy.d.par_chunks(c))
            .zip(self.xhat.par_chunks(c))
            .zip(self.invstd.par_iter())
            .for_each(|(((dxr, dyr), xh), &is)| {
                let (mut m1, mut m2) = (0.0f32, 0.0f32);
                for j in 0..c {
                    let d = dyr[j] * gv[j];
                    m1 += d;
                    m2 += d * xh[j];
                }
                let inv = 1.0 / c as f32;
                for j in 0..c {
                    let d = dyr[j] * gv[j];
                    dxr[j] = is * (d - m1 * inv - xh[j] * m2 * inv);
                }
            });
        dx
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.g);
        f(&mut self.b);
    }
}

// ---------------------------------------------------------------- LayerScale

/// y = lambda * x, com lambda por canal inicializado perto de zero.
///
/// E o mesmo truque do `zero_gamma()` do bloco residual convolucional: o ramo
/// comeca mudo e a rede parte da identidade, o que estabiliza o inicio.
pub struct LayerScale {
    pub l: Param,
    c: usize,
    x: Vec<f32>,
}

impl LayerScale {
    pub fn new(c: usize, init: f32) -> LayerScale {
        LayerScale {
            l: Param::filled(c, init, false),
            c,
            x: Vec::new(),
        }
    }

    /// Escala no proprio tensor recebido: a saida do ramo nao e usada em mais
    /// nenhum lugar, entao consumi-la evita uma alocacao e uma passada.
    pub fn forward(&mut self, mut x: Tok, train: bool) -> Tok {
        if train {
            self.x.clear();
            self.x.extend_from_slice(&x.d);
        }
        let lv = &self.l.v;
        x.d.par_chunks_mut(self.c).for_each(|r| {
            for (j, v) in r.iter_mut().enumerate() {
                *v *= lv[j];
            }
        });
        x
    }

    pub fn backward(&mut self, mut dy: Tok) -> Tok {
        let c = self.c;
        let dl =
            dy.d.par_chunks(c)
                .zip(self.x.par_chunks(c))
                .fold(
                    || vec![0.0f32; c],
                    |mut a, (dyr, xr)| {
                        for j in 0..c {
                            a[j] += dyr[j] * xr[j];
                        }
                        a
                    },
                )
                .reduce(
                    || vec![0.0f32; c],
                    |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(p, q)| *p += q);
                        a
                    },
                );
        self.l.g.iter_mut().zip(&dl).for_each(|(a, b)| *a += b);

        let lv = &self.l.v;
        dy.d.par_chunks_mut(c).for_each(|r| {
            for (j, v) in r.iter_mut().enumerate() {
                *v *= lv[j];
            }
        });
        dy
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.l);
    }
}

// ---------------------------------------------------------------- SwiGLU

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// y = (silu(x W1) * (x W2)) W3, com dimensao interna 4c/3.
///
/// Mesma contagem de parametros e de operacoes de um MLP de razao 2, e
/// consistentemente melhor. W1 e W2 sao uma projecao so, de `c` para `2h`: um
/// GEMM em vez de dois, e o empacotamento do peso acontece uma vez.
pub struct SwiGlu {
    pub wg: Linear,
    pub wd: Linear,
    pub h: usize,
    ab: Vec<f32>,
    rows: usize,
}

impl SwiGlu {
    pub fn new(c: usize, h: usize, depth: usize, rng: &mut Rng) -> SwiGlu {
        SwiGlu {
            wg: Linear::new(c, 2 * h, true, 0.02, rng),
            wd: Linear::scaled(h, c, true, depth, rng),
            h,
            ab: Vec::new(),
            rows: 0,
        }
    }

    pub fn forward(&mut self, x: &Tok, train: bool) -> Tok {
        let rows = x.rows();
        let h = self.h;
        let ab = self.wg.forward(&x.d, rows, train);
        let mut u = unsafe { scratch_vec(rows * h) };
        u.par_chunks_mut(h)
            .zip(ab.par_chunks(2 * h))
            .for_each(|(ur, abr)| {
                for j in 0..h {
                    let a = abr[j];
                    ur[j] = a * sigmoid(a) * abr[h + j];
                }
            });
        if train {
            self.ab = ab;
            self.rows = rows;
        }
        Tok::from_vec(self.wd.forward(&u, rows, train), x.n, x.t, x.c)
    }

    pub fn backward(&mut self, dy: &Tok) -> Tok {
        let (rows, h) = (self.rows, self.h);
        let du = self.wd.backward(&dy.d);
        let mut dab = unsafe { scratch_vec(rows * 2 * h) };
        dab.par_chunks_mut(2 * h)
            .zip(du.par_chunks(h))
            .zip(self.ab.par_chunks(2 * h))
            .for_each(|((dr, dur), abr)| {
                for j in 0..h {
                    let (a, b) = (abr[j], abr[h + j]);
                    let s = sigmoid(a);
                    let sa = a * s;
                    // d/da [a*sigmoid(a)] = s * (1 + a * (1 - s))
                    dr[j] = dur[j] * b * s * (1.0 + a * (1.0 - s));
                    dr[h + j] = dur[j] * sa;
                }
            });
        Tok::from_vec(self.wg.backward(&dab), dy.n, dy.t, dy.c)
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        self.wg.for_each_param(f);
        self.wd.for_each_param(f);
    }
}

// ---------------------------------------------------------------- depthwise

/// Convolucao 3x3 depthwise sobre a grade de tokens, em layout channel-last.
///
/// Com o canal na dimensao mais rapida, `y[p] += w[k] * x[p + off[k]]` percorre
/// `c` valores contiguos: o compilador vetoriza os nove taps sem im2col, sem
/// transposicao e sem empacotamento. Custa 0.15 MMAC por camada, o que na
/// pratica e nada, e devolve ao bloco a localidade que a atencao nao tem.
pub struct DwConv3 {
    /// [9, c]
    pub w: Param,
    pub b: Param,
    pub c: usize,
    pub th: usize,
    pub tw: usize,
    x: Vec<f32>,
    n: usize,
}

impl DwConv3 {
    pub fn new(c: usize, th: usize, tw: usize, rng: &mut Rng) -> DwConv3 {
        DwConv3 {
            w: Param::trunc_normal(9 * c, 0.1, false, rng),
            b: Param::new(c, false),
            c,
            th,
            tw,
            x: Vec::new(),
            n: 0,
        }
    }

    pub fn forward(&mut self, x: &Tok, train: bool) -> Tok {
        let (c, th, tw) = (self.c, self.th, self.tw);
        let mut y = x.zero_like();
        let (wv, bv) = (&self.w.v, &self.b.v);
        y.d.par_chunks_mut(th * tw * c)
            .zip(x.d.par_chunks(th * tw * c))
            .for_each(|(yi, xi)| {
                for oy in 0..th {
                    for ox in 0..tw {
                        let o = (oy * tw + ox) * c;
                        yi[o..o + c].copy_from_slice(bv);
                        for ky in 0..3 {
                            let sy = oy as isize + ky as isize - 1;
                            if sy < 0 || sy >= th as isize {
                                continue;
                            }
                            for kx in 0..3 {
                                let sx = ox as isize + kx as isize - 1;
                                if sx < 0 || sx >= tw as isize {
                                    continue;
                                }
                                let wr = &wv[(ky * 3 + kx) * c..(ky * 3 + kx + 1) * c];
                                let s = (sy as usize * tw + sx as usize) * c;
                                let src = &xi[s..s + c];
                                let dst = &mut yi[o..o + c];
                                for j in 0..c {
                                    dst[j] += wr[j] * src[j];
                                }
                            }
                        }
                    }
                }
            });
        if train {
            self.x.clear();
            self.x.extend_from_slice(&x.d);
            self.n = x.n;
        }
        y
    }

    pub fn backward(&mut self, dy: &Tok) -> Tok {
        let (c, th, tw) = (self.c, self.th, self.tw);
        let img = th * tw * c;
        let mut dx = dy.zero_like();

        let (dw, db) =
            dy.d.par_chunks(img)
                .zip(self.x.par_chunks(img))
                .fold(
                    || (vec![0.0f32; 9 * c], vec![0.0f32; c]),
                    |(mut aw, mut ab), (dyi, xi)| {
                        for oy in 0..th {
                            for ox in 0..tw {
                                let o = (oy * tw + ox) * c;
                                let d = &dyi[o..o + c];
                                for j in 0..c {
                                    ab[j] += d[j];
                                }
                                for ky in 0..3 {
                                    let sy = oy as isize + ky as isize - 1;
                                    if sy < 0 || sy >= th as isize {
                                        continue;
                                    }
                                    for kx in 0..3 {
                                        let sx = ox as isize + kx as isize - 1;
                                        if sx < 0 || sx >= tw as isize {
                                            continue;
                                        }
                                        let s = (sy as usize * tw + sx as usize) * c;
                                        let src = &xi[s..s + c];
                                        let acc = &mut aw[(ky * 3 + kx) * c..(ky * 3 + kx + 1) * c];
                                        for j in 0..c {
                                            acc[j] += d[j] * src[j];
                                        }
                                    }
                                }
                            }
                        }
                        (aw, ab)
                    },
                )
                .reduce(
                    || (vec![0.0f32; 9 * c], vec![0.0f32; c]),
                    |(mut a, mut b), (x, y)| {
                        a.iter_mut().zip(x).for_each(|(p, q)| *p += q);
                        b.iter_mut().zip(y).for_each(|(p, q)| *p += q);
                        (a, b)
                    },
                );
        self.w.g.iter_mut().zip(&dw).for_each(|(a, b)| *a += b);
        self.b.g.iter_mut().zip(&db).for_each(|(a, b)| *a += b);

        let wv = &self.w.v;
        dx.d.par_chunks_mut(img)
            .zip(dy.d.par_chunks(img))
            .for_each(|(dxi, dyi)| {
                for oy in 0..th {
                    for ox in 0..tw {
                        let o = (oy * tw + ox) * c;
                        let d = &dyi[o..o + c];
                        for ky in 0..3 {
                            let sy = oy as isize + ky as isize - 1;
                            if sy < 0 || sy >= th as isize {
                                continue;
                            }
                            for kx in 0..3 {
                                let sx = ox as isize + kx as isize - 1;
                                if sx < 0 || sx >= tw as isize {
                                    continue;
                                }
                                let wr = &wv[(ky * 3 + kx) * c..(ky * 3 + kx + 1) * c];
                                let s = (sy as usize * tw + sx as usize) * c;
                                let dst = &mut dxi[s..s + c];
                                for j in 0..c {
                                    dst[j] += wr[j] * d[j];
                                }
                            }
                        }
                    }
                }
            });
        dx
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.w);
        f(&mut self.b);
    }
}

// ---------------------------------------------------------------- SeqPool

/// Reducao por atencao: w = softmax(x . p), z = sum_t w_t x_t.
///
/// Substitui a media global do modelo convolucional. Com um vetor de `c` pesos
/// ela aprende quais tokens importam, e evita o token [CLS], que num conjunto de
/// cinquenta mil imagens e treinado por pouca coisa.
pub struct SeqPool {
    pub p: Param,
    pub c: usize,
    w: Vec<f32>,
    x: Vec<f32>,
    n: usize,
    t: usize,
}

impl SeqPool {
    pub fn new(c: usize, rng: &mut Rng) -> SeqPool {
        SeqPool {
            p: Param::trunc_normal(c, 0.02, true, rng),
            c,
            w: Vec::new(),
            x: Vec::new(),
            n: 0,
            t: 0,
        }
    }

    /// Retorna [n, c].
    pub fn forward(&mut self, x: &Tok, train: bool) -> Vec<f32> {
        let (n, t, c) = (x.n, x.t, x.c);
        let mut z = vec![0.0f32; n * c];
        let mut w = vec![0.0f32; n * t];
        let pv = &self.p.v;
        z.par_chunks_mut(c)
            .zip(w.par_chunks_mut(t))
            .zip(x.d.par_chunks(t * c))
            .for_each(|((zi, wi), xi)| {
                let mut mx = f32::NEG_INFINITY;
                for j in 0..t {
                    let mut s = 0.0;
                    let r = &xi[j * c..(j + 1) * c];
                    for k in 0..c {
                        s += r[k] * pv[k];
                    }
                    wi[j] = s;
                    if s > mx {
                        mx = s;
                    }
                }
                let mut sum = 0.0;
                for v in wi.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                for j in 0..t {
                    wi[j] *= inv;
                    let r = &xi[j * c..(j + 1) * c];
                    for k in 0..c {
                        zi[k] += wi[j] * r[k];
                    }
                }
            });
        if train {
            self.w = w;
            self.x.clear();
            self.x.extend_from_slice(&x.d);
            self.n = n;
            self.t = t;
        }
        z
    }

    pub fn backward(&mut self, dz: &[f32]) -> Tok {
        let (n, t, c) = (self.n, self.t, self.c);
        let mut dx = Tok::zeros(n, t, c);
        let pv = &self.p.v;
        let dp =
            dx.d.par_chunks_mut(t * c)
                .zip(dz.par_chunks(c))
                .zip(self.w.par_chunks(t))
                .zip(self.x.par_chunks(t * c))
                .fold(
                    || vec![0.0f32; c],
                    |mut adp, (((dxi, dzi), wi), xi)| {
                        // dw_j = <dz, x_j>, e o caminho direto dx_j += w_j * dz
                        let mut dw = vec![0.0f32; t];
                        let mut dot = 0.0f32;
                        for j in 0..t {
                            let xr = &xi[j * c..(j + 1) * c];
                            let dr = &mut dxi[j * c..(j + 1) * c];
                            let mut s = 0.0;
                            for k in 0..c {
                                s += dzi[k] * xr[k];
                                dr[k] += wi[j] * dzi[k];
                            }
                            dw[j] = s;
                            dot += s * wi[j];
                        }
                        // softmax: ds_j = w_j * (dw_j - sum_l w_l dw_l)
                        for j in 0..t {
                            let ds = wi[j] * (dw[j] - dot);
                            let xr = &xi[j * c..(j + 1) * c];
                            let dr = &mut dxi[j * c..(j + 1) * c];
                            for k in 0..c {
                                adp[k] += ds * xr[k];
                                dr[k] += ds * pv[k];
                            }
                        }
                        adp
                    },
                )
                .reduce(
                    || vec![0.0f32; c],
                    |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(p, q)| *p += q);
                        a
                    },
                );
        self.p.g.iter_mut().zip(&dp).for_each(|(a, b)| *a += b);
        dx
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.p);
    }
}
