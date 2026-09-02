//! Atencao multi-cabeca sobre tokens.
//!
//! Duas variantes com o mesmo nucleo. A auto-atencao roda sobre os 64 tokens da
//! grade 8x8. A atencao cruzada le uma memoria de alta resolucao de 256 tokens
//! construida a partir do mapa 16x16 do tronco convolucional: como o custo e
//! `Tq * Tk * D`, 64 queries em 256 chaves custam quatro vezes uma atencao
//! 64x64, e nao dezesseis, que e o que custaria mover o estagio inteiro para
//! 16x16. O transformer enxerga resolucao 16x16 a preco de 8x8.
//!
//! Com T=64 e dh=64, Q, K e V de uma cabeca somam 48 KB e a matriz de scores
//! 16 KB: tudo cabe na L2 e quase tudo na L1. O beneficio do flash attention sai
//! de graca pelo tamanho do problema, sem precisar do algoritmo.

use crate::gemm::{gemm_small, scratch_vec, PackedA, PackedB};
use crate::linear::Linear;
use crate::nn::Param;
use crate::rng::Rng;
use crate::tok::Tok;
use rayon::prelude::*;

/// Tabela (i, j) -> indice do vies posicional relativo, montada uma vez.
///
/// Para a auto-atencao as duas grades sao iguais e o deslocamento vai de
/// `-(th-1)` a `th-1`. Para a atencao cruzada a grade da memoria tem o dobro da
/// resolucao, entao o deslocamento e medido em meio token: `2*qy - my`.
fn rel_index(th: usize, tw: usize, mh: usize, mw: usize) -> (Vec<u32>, usize) {
    let (sy, sx) = if mh == th {
        (2 * th - 1, 2 * tw - 1)
    } else {
        (2 * (th - 1) + mh, 2 * (tw - 1) + mw)
    };
    let scale = if mh == th { 1isize } else { 2 };
    let (oy, ox) = if mh == th {
        (th as isize - 1, tw as isize - 1)
    } else {
        (mh as isize - 1, mw as isize - 1)
    };
    let mut idx = Vec::with_capacity(th * tw * mh * mw);
    for qy in 0..th as isize {
        for qx in 0..tw as isize {
            for my in 0..mh as isize {
                for mx in 0..mw as isize {
                    let dy = scale * qy - my + oy;
                    let dx = scale * qx - mx + ox;
                    idx.push((dy * sx as isize + dx) as u32);
                }
            }
        }
    }
    (idx, sy * sx)
}

/// Parametros compartilhados pelas duas variantes.
struct Core {
    h: usize,
    dh: usize,
    tq: usize,
    tk: usize,
    relidx: Vec<u32>,
    nrel: usize,
    /// mascara a diagonal (LSA): cada token e obrigado a olhar para os outros
    mask_diag: bool,
}

/// Buffers de uma tarefa (uma imagem). Q, K e V de uma cabeca somam 48 KB com
/// dh=64: uma vez reunidos em memoria contigua, a sequencia inteira roda dentro
/// da L2 e quase toda dentro da L1.
#[derive(Default)]
struct Work {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    s: Vec<f32>,
    o: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    ap: PackedA,
    bp: PackedB,
}

impl Work {
    fn size(&mut self, tq: usize, tk: usize, dh: usize) {
        self.q.resize(tq * dh, 0.0);
        self.k.resize(tk * dh, 0.0);
        self.v.resize(tk * dh, 0.0);
        self.s.resize(tq * tk, 0.0);
        self.o.resize(tq * dh, 0.0);
        self.dq.resize(tq * dh, 0.0);
        self.dk.resize(tk * dh, 0.0);
        self.dv.resize(tk * dh, 0.0);
    }
}

/// Copia `rows` linhas de `src` (passo `stride`, deslocamento `off`) para um
/// bloco contiguo de `dh` colunas.
#[inline]
fn gather(src: &[f32], base: usize, stride: usize, off: usize, rows: usize, dh: usize, dst: &mut [f32]) {
    for r in 0..rows {
        dst[r * dh..(r + 1) * dh].copy_from_slice(&src[base + r * stride + off..][..dh]);
    }
}

#[inline]
fn scatter_add(
    dst: &mut [f32],
    base: usize,
    stride: usize,
    off: usize,
    rows: usize,
    dh: usize,
    src: &[f32],
) {
    for r in 0..rows {
        let d = &mut dst[base + r * stride + off..][..dh];
        for z in 0..dh {
            d[z] += src[r * dh + z];
        }
    }
}

impl Core {
    /// s = tau * <q,k> + rel; p = softmax(s); o = p * v
    ///
    /// Os dois produtos passam pelo micro-kernel 6x16, e nao por lacos escalares:
    /// medido, o caminho escalar custava um quarto do passo de treino inteiro.
    fn forward(
        &self,
        q: &[f32],
        qstride: usize,
        qoff: usize,
        kv: &[f32],
        kvstride: usize,
        koff: usize,
        voff: usize,
        tau: &[f32],
        rel: &[f32],
        p: &mut [f32],
        o: &mut [f32],
    ) {
        let (h, dh, tq, tk) = (self.h, self.dh, self.tq, self.tk);
        let c = h * dh;
        let mask = self.mask_diag;
        let (relidx, nrel) = (&self.relidx, self.nrel);

        o.par_chunks_mut(tq * c)
            .zip(p.par_chunks_mut(h * tq * tk))
            .enumerate()
            .for_each_init(Work::default, |w, (i, (oi, pi))| {
                w.size(tq, tk, dh);
                let qbase = i * tq * qstride;
                let kbase = i * tk * kvstride;
                for hd in 0..h {
                    let (hq, hk, hv) = (qoff + hd * dh, koff + hd * dh, voff + hd * dh);
                    gather(q, qbase, qstride, hq, tq, dh, &mut w.q);
                    gather(kv, kbase, kvstride, hk, tk, dh, &mut w.k);
                    gather(kv, kbase, kvstride, hv, tk, dh, &mut w.v);

                    // S = Q K^T
                    gemm_small(
                        tq, tk, dh, &w.q, false, &w.k, true, &mut w.s, false, &mut w.ap, &mut w.bp,
                    );

                    let (t, rb) = (tau[hd], &rel[hd * nrel..(hd + 1) * nrel]);
                    let pr = &mut pi[hd * tq * tk..(hd + 1) * tq * tk];
                    for qi in 0..tq {
                        let sr = &mut w.s[qi * tk..(qi + 1) * tk];
                        let dr = &mut pr[qi * tk..(qi + 1) * tk];
                        let ridx = &relidx[qi * tk..(qi + 1) * tk];
                        let mut mx = f32::NEG_INFINITY;
                        for kj in 0..tk {
                            let val = t * sr[kj] + rb[ridx[kj] as usize];
                            dr[kj] = if mask && kj == qi { f32::NEG_INFINITY } else { val };
                            if dr[kj] > mx {
                                mx = dr[kj];
                            }
                        }
                        let mut sum = 0.0f32;
                        for kj in 0..tk {
                            let e = if dr[kj] == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (dr[kj] - mx).exp()
                            };
                            dr[kj] = e;
                            sum += e;
                        }
                        let inv = 1.0 / sum;
                        dr.iter_mut().for_each(|v| *v *= inv);
                    }

                    // O = P V
                    gemm_small(
                        tq, dh, tk, pr, false, &w.v, false, &mut w.o, false, &mut w.ap, &mut w.bp,
                    );
                    for qi in 0..tq {
                        oi[qi * c + hd * dh..qi * c + (hd + 1) * dh]
                            .copy_from_slice(&w.o[qi * dh..(qi + 1) * dh]);
                    }
                }
            });
    }

    /// Acumula dq em `dq` e dk/dv em `dkv`, alem de dtau e drel.
    fn backward(
        &self,
        q: &[f32],
        qstride: usize,
        qoff: usize,
        kv: &[f32],
        kvstride: usize,
        koff: usize,
        voff: usize,
        tau: &[f32],
        p: &[f32],
        dobuf: &[f32],
        dq: &mut [f32],
        dkv: &mut [f32],
        dtau: &mut [f32],
        drel: &mut [f32],
    ) {
        let (h, dh, tq, tk) = (self.h, self.dh, self.tq, self.tk);
        let c = h * dh;
        let (relidx, nrel) = (&self.relidx, self.nrel);

        let (ta, ra) = dq
            .par_chunks_mut(tq * qstride)
            .zip(dkv.par_chunks_mut(tk * kvstride))
            .zip(dobuf.par_chunks(tq * c))
            .zip(p.par_chunks(h * tq * tk))
            .enumerate()
            .fold(
                || (vec![0.0f32; h], vec![0.0f32; h * nrel], Work::default()),
                |(mut at, mut ar, mut w), (i, (((dqi, dkvi), doi), pi))| {
                    w.size(tq, tk, dh);
                    let qbase = i * tq * qstride;
                    let kbase = i * tk * kvstride;
                    for hd in 0..h {
                        let (hq, hk, hv) = (qoff + hd * dh, koff + hd * dh, voff + hd * dh);
                        gather(q, qbase, qstride, hq, tq, dh, &mut w.q);
                        gather(kv, kbase, kvstride, hk, tk, dh, &mut w.k);
                        gather(kv, kbase, kvstride, hv, tk, dh, &mut w.v);
                        for qi in 0..tq {
                            w.o[qi * dh..(qi + 1) * dh]
                                .copy_from_slice(&doi[qi * c + hd * dh..qi * c + (hd + 1) * dh]);
                        }
                        let pr = &pi[hd * tq * tk..(hd + 1) * tq * tk];

                        // dV = P^T dO
                        gemm_small(
                            tk, dh, tq, pr, true, &w.o, false, &mut w.dv, false, &mut w.ap, &mut w.bp,
                        );
                        scatter_add(dkvi, 0, kvstride, hv, tk, dh, &w.dv);

                        // dP = dO V^T, e o softmax por linha vira dS
                        gemm_small(
                            tq, tk, dh, &w.o, false, &w.v, true, &mut w.s, false, &mut w.ap, &mut w.bp,
                        );
                        for qi in 0..tq {
                            let sr = &mut w.s[qi * tk..(qi + 1) * tk];
                            let prr = &pr[qi * tk..(qi + 1) * tk];
                            let mut dot = 0.0f32;
                            for kj in 0..tk {
                                dot += sr[kj] * prr[kj];
                            }
                            for kj in 0..tk {
                                sr[kj] = prr[kj] * (sr[kj] - dot);
                            }
                        }

                        // dtau precisa do produto bruto <q,k>; recalcula-lo por
                        // GEMM sai mais barato que guardar mais um [n,h,tq,tk].
                        let t = tau[hd];
                        {
                            let mut sd = std::mem::take(&mut w.dq);
                            sd.resize(tq * tk, 0.0);
                            gemm_small(
                                tq, tk, dh, &w.q, false, &w.k, true, &mut sd, false, &mut w.ap,
                                &mut w.bp,
                            );
                            let mut acc = 0.0f32;
                            for z in 0..tq * tk {
                                acc += w.s[z] * sd[z];
                            }
                            at[hd] += acc;
                            w.dq = sd;
                        }
                        for qi in 0..tq {
                            let sr = &w.s[qi * tk..(qi + 1) * tk];
                            let ridx = &relidx[qi * tk..(qi + 1) * tk];
                            for kj in 0..tk {
                                ar[hd * nrel + ridx[kj] as usize] += sr[kj];
                            }
                        }
                        w.s.iter_mut().for_each(|v| *v *= t);

                        // dQ = dS K, dK = dS^T Q
                        w.dq.resize(tq * dh, 0.0);
                        gemm_small(
                            tq, dh, tk, &w.s, false, &w.k, false, &mut w.dq, false, &mut w.ap, &mut w.bp,
                        );
                        scatter_add(dqi, 0, qstride, hq, tq, dh, &w.dq);
                        gemm_small(
                            tk, dh, tq, &w.s, true, &w.q, false, &mut w.dk, false, &mut w.ap, &mut w.bp,
                        );
                        scatter_add(dkvi, 0, kvstride, hk, tk, dh, &w.dk);
                    }
                    (at, ar, w)
                },
            )
            .map(|(a, b, _)| (a, b))
            .reduce(
                || (vec![0.0f32; h], vec![0.0f32; h * nrel]),
                |(mut a, mut b), (x, y)| {
                    a.iter_mut().zip(x).for_each(|(p, q)| *p += q);
                    b.iter_mut().zip(y).for_each(|(p, q)| *p += q);
                    (a, b)
                },
            );
        dtau.iter_mut().zip(&ta).for_each(|(a, b)| *a += b);
        drel.iter_mut().zip(&ra).for_each(|(a, b)| *a += b);
    }
}

// ---------------------------------------------------------------- auto-atencao

pub struct SelfAttn {
    pub qkv: Linear,
    pub proj: Linear,
    /// temperatura aprendida por cabeca (LSA), inicializada em 1/sqrt(dh)
    pub tau: Param,
    /// vies posicional relativo 2D, [h, nrel]
    pub rel: Param,
    core: Core,
    c: usize,
    ab: Vec<f32>,
    p: Vec<f32>,
    n: usize,
}

impl SelfAttn {
    pub fn new(c: usize, h: usize, th: usize, tw: usize, depth: usize, rng: &mut Rng) -> SelfAttn {
        let dh = c / h;
        let (relidx, nrel) = rel_index(th, tw, th, tw);
        SelfAttn {
            qkv: Linear::new(c, 3 * c, true, 0.02, rng),
            proj: Linear::scaled(c, c, true, depth, rng),
            tau: Param::filled(h, 1.0 / (dh as f32).sqrt(), false),
            rel: Param::new(h * nrel, false),
            core: Core {
                h,
                dh,
                tq: th * tw,
                tk: th * tw,
                relidx,
                nrel,
                mask_diag: true,
            },
            c,
            ab: Vec::new(),
            p: Vec::new(),
            n: 0,
        }
    }

    pub fn forward(&mut self, x: &Tok, train: bool) -> Tok {
        let (n, t, c) = (x.n, x.t, self.c);
        let h = self.core.h;
        let ab = self.qkv.forward(&x.d, n * t, train);
        let mut p = unsafe { scratch_vec(n * h * t * t) };
        let mut o = unsafe { scratch_vec(n * t * c) };
        self.core.forward(
            &ab,
            3 * c,
            0,
            &ab,
            3 * c,
            c,
            2 * c,
            &self.tau.v,
            &self.rel.v,
            &mut p,
            &mut o,
        );
        let y = self.proj.forward(&o, n * t, train);
        if train {
            self.ab = ab;
            self.p = p;
            self.n = n;
        }
        Tok::from_vec(y, n, t, c)
    }

    pub fn backward(&mut self, dy: &Tok) -> Tok {
        let (n, t, c) = (self.n, self.core.tq, self.c);
        let dobuf = self.proj.backward(&dy.d);
        let mut dab = vec![0.0f32; n * t * 3 * c];
        // dq e dkv apontam para o mesmo buffer: a projecao QKV e fundida, entao
        // o gradiente das tres partes se acumula na mesma matriz.
        let mut dkv = vec![0.0f32; n * t * 3 * c];
        self.core.backward(
            &self.ab,
            3 * c,
            0,
            &self.ab,
            3 * c,
            c,
            2 * c,
            &self.tau.v,
            &self.p,
            &dobuf,
            &mut dab,
            &mut dkv,
            &mut self.tau.g,
            &mut self.rel.g,
        );
        dab.par_iter_mut().zip(dkv.par_iter()).for_each(|(a, b)| *a += b);
        Tok::from_vec(self.qkv.backward(&dab), n, t, c)
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        self.qkv.for_each_param(f);
        self.proj.for_each_param(f);
        f(&mut self.tau);
        f(&mut self.rel);
    }
}

// ---------------------------------------------------------------- cruzada

/// Le a memoria de alta resolucao. K e V vem prontos de fora, num tensor
/// [n, m, 2c] produzido uma unica vez por uma conv 1x1 sobre o mapa 16x16 e
/// compartilhado por todos os blocos leitores.
pub struct CrossAttn {
    pub wq: Linear,
    pub proj: Linear,
    pub tau: Param,
    pub rel: Param,
    core: Core,
    c: usize,
    q: Vec<f32>,
    p: Vec<f32>,
    n: usize,
}

impl CrossAttn {
    pub fn new(
        c: usize,
        h: usize,
        th: usize,
        tw: usize,
        mh: usize,
        mw: usize,
        depth: usize,
        rng: &mut Rng,
    ) -> CrossAttn {
        let dh = c / h;
        let (relidx, nrel) = rel_index(th, tw, mh, mw);
        CrossAttn {
            wq: Linear::new(c, c, true, 0.02, rng),
            proj: Linear::scaled(c, c, true, depth, rng),
            tau: Param::filled(h, 1.0 / (dh as f32).sqrt(), false),
            rel: Param::new(h * nrel, false),
            core: Core {
                h,
                dh,
                tq: th * tw,
                tk: mh * mw,
                relidx,
                nrel,
                mask_diag: false,
            },
            c,
            q: Vec::new(),
            p: Vec::new(),
            n: 0,
        }
    }

    /// `mem` e [n, tk, 2c]: K nas primeiras c colunas, V nas ultimas c.
    pub fn forward(&mut self, x: &Tok, mem: &[f32], train: bool) -> Tok {
        let (n, t, c) = (x.n, x.t, self.c);
        let (h, tk) = (self.core.h, self.core.tk);
        let q = self.wq.forward(&x.d, n * t, train);
        let mut p = unsafe { scratch_vec(n * h * t * tk) };
        let mut o = unsafe { scratch_vec(n * t * c) };
        self.core.forward(
            &q,
            c,
            0,
            mem,
            2 * c,
            0,
            c,
            &self.tau.v,
            &self.rel.v,
            &mut p,
            &mut o,
        );
        let y = self.proj.forward(&o, n * t, train);
        if train {
            self.q = q;
            self.p = p;
            self.n = n;
        }
        Tok::from_vec(y, n, t, c)
    }

    /// Retorna dx e acumula o gradiente da memoria em `dmem`.
    pub fn backward(&mut self, dy: &Tok, mem: &[f32], dmem: &mut [f32]) -> Tok {
        let (n, t, c) = (self.n, self.core.tq, self.c);
        let dobuf = self.proj.backward(&dy.d);
        let mut dq = vec![0.0f32; n * t * c];
        self.core.backward(
            &self.q,
            c,
            0,
            mem,
            2 * c,
            0,
            c,
            &self.tau.v,
            &self.p,
            &dobuf,
            &mut dq,
            dmem,
            &mut self.tau.g,
            &mut self.rel.g,
        );
        Tok::from_vec(self.wq.backward(&dq), n, t, c)
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        self.wq.for_each_param(f);
        self.proj.for_each_param(f);
        f(&mut self.tau);
        f(&mut self.rel);
    }
}
