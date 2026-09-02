//! Camada densa sobre o GEMM.
//!
//! Com os tokens em [n, t, c] o lote inteiro e uma matriz [n*t, c], e cada uma
//! das tres derivadas da camada e um unico produto grande: M na casa dos
//! milhares, sem borda, sem `RowMap` e sem divisao inteira no laco interno. E
//! uma forma mais amigavel ao micro-kernel 6x16 do que qualquer convolucao
//! deste repositorio.

use crate::gemm::{matmul, matmul_dw, scratch_vec};
use crate::nn::Param;
use crate::rng::Rng;
use rayon::prelude::*;

pub struct Linear {
    /// [fout, fin] row-major
    pub w: Param,
    pub b: Option<Param>,
    pub fin: usize,
    pub fout: usize,
    x: Vec<f32>,
    rows: usize,
}

impl Linear {
    pub fn new(fin: usize, fout: usize, bias: bool, std: f32, rng: &mut Rng) -> Linear {
        Linear {
            w: Param::trunc_normal(fin * fout, std, true, rng).shaped(fout, fin),
            b: if bias { Some(Param::new(fout, false)) } else { None },
            fin,
            fout,
            x: Vec::new(),
            rows: 0,
        }
    }

    /// Inicializacao dos ramos que saem de um residual: escala reduzida por
    /// profundidade, o analogo do `zero_gamma` do bloco convolucional.
    pub fn scaled(fin: usize, fout: usize, bias: bool, depth: usize, rng: &mut Rng) -> Linear {
        let std = 0.02 / (2.0 * depth as f32).sqrt();
        Linear::new(fin, fout, bias, std, rng)
    }

    /// y[rows, fout] = x[rows, fin] * W^T + b
    pub fn forward(&mut self, x: &[f32], rows: usize, train: bool) -> Vec<f32> {
        let mut y = unsafe { scratch_vec(rows * self.fout) };
        matmul(
            rows, self.fout, self.fin, x, false, &self.w.v, true, &mut y, false,
        );
        if let Some(b) = &self.b {
            let bv = &b.v;
            y.par_chunks_mut(self.fout).for_each(|r| {
                for (o, v) in r.iter_mut().enumerate() {
                    *v += bv[o];
                }
            });
        }
        if train {
            self.x.clear();
            self.x.extend_from_slice(&x[..rows * self.fin]);
            self.rows = rows;
        }
        y
    }

    /// Acumula dW e db, retorna dx[rows, fin].
    pub fn backward(&mut self, dy: &[f32]) -> Vec<f32> {
        let rows = self.rows;
        // dW[fout, fin] += dY^T[fout, rows] * X[rows, fin]
        matmul_dw(self.fout, self.fin, rows, dy, &self.x, &mut self.w.g);
        if let Some(b) = &mut self.b {
            let fout = self.fout;
            let acc = dy
                .par_chunks(fout)
                .fold(
                    || vec![0.0f32; fout],
                    |mut a, r| {
                        a.iter_mut().zip(r).for_each(|(x, y)| *x += y);
                        a
                    },
                )
                .reduce(
                    || vec![0.0f32; fout],
                    |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                        a
                    },
                );
            b.g.iter_mut().zip(&acc).for_each(|(x, y)| *x += y);
        }
        // dX[rows, fin] = dY[rows, fout] * W[fout, fin]
        let mut dx = unsafe { scratch_vec(rows * self.fin) };
        matmul(
            rows, self.fin, self.fout, dy, false, &self.w.v, false, &mut dx, false,
        );
        dx
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.w);
        if let Some(b) = &mut self.b {
            f(b);
        }
    }
}
