//! Tensor de tokens em layout [n, t, c] row-major.
//!
//! O canal e a dimensao mais rapida, e isso decide tres coisas de uma vez. As
//! matrizes de peso viram um GEMM unico de `[n*t, c]`, sem bordas e sem
//! `RowMap`. A LayerNorm e o softmax operam sobre trechos contiguos. E a
//! convolucao depthwise vira um laco de 9 taps que vetoriza sobre `c`, sem
//! im2col e sem transposicao: dezesseis canais por instrucao.

use crate::nn::Tensor;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Tok {
    pub d: Vec<f32>,
    /// imagens no lote
    pub n: usize,
    /// tokens por imagem
    pub t: usize,
    /// canais por token
    pub c: usize,
}

impl Tok {
    pub fn zeros(n: usize, t: usize, c: usize) -> Tok {
        Tok {
            d: vec![0.0; n * t * c],
            n,
            t,
            c,
        }
    }

    /// Linhas da matriz [n*t, c] que o GEMM enxerga.
    #[inline]
    pub fn rows(&self) -> usize {
        self.n * self.t
    }

    /// Envolve um vetor pronto sem copiar.
    ///
    /// As camadas densas devolvem `Vec<f32>` no layout exato do `Tok`, e
    /// alocar um tensor zerado so para copiar por cima custava a memset mais a
    /// copia em cada uma das dezenas de camadas do modelo.
    pub fn from_vec(d: Vec<f32>, n: usize, t: usize, c: usize) -> Tok {
        debug_assert_eq!(d.len(), n * t * c);
        Tok { d, n, t, c }
    }

    pub fn zero_like(&self) -> Tok {
        Tok::zeros(self.n, self.t, self.c)
    }

    /// Converte [n, c, h, w] em [n, h*w, c].
    pub fn from_tensor(x: &Tensor) -> Tok {
        let (n, c, hw) = (x.n, x.c, x.h * x.w);
        let mut y = Tok::zeros(n, hw, c);
        y.d.par_chunks_mut(hw * c)
            .zip(x.d.par_chunks(c * hw))
            .for_each(|(dst, src)| {
                for ci in 0..c {
                    let s = &src[ci * hw..(ci + 1) * hw];
                    for p in 0..hw {
                        dst[p * c + ci] = s[p];
                    }
                }
            });
        y
    }

    /// Inversa de `from_tensor`, para o gradiente voltar ao tronco convolucional.
    pub fn to_tensor(&self, h: usize, w: usize) -> Tensor {
        debug_assert_eq!(h * w, self.t);
        let (c, hw) = (self.c, self.t);
        let mut y = Tensor::zeros(self.n, c, h, w);
        y.d.par_chunks_mut(c * hw)
            .zip(self.d.par_chunks(hw * c))
            .for_each(|(dst, src)| {
                for ci in 0..c {
                    let d = &mut dst[ci * hw..(ci + 1) * hw];
                    for p in 0..hw {
                        d[p] = src[p * c + ci];
                    }
                }
            });
        y
    }
}

/// a += b
pub fn add_(a: &mut Tok, b: &Tok) {
    a.d.par_iter_mut().zip(b.d.par_iter()).for_each(|(x, y)| *x += y);
}
