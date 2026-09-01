//! Convolucao com im2col fundido no empacotamento do GEMM.
//!
//! A matriz im2col nunca e materializada: o painel B de kc x 16 e montado direto
//! a partir da imagem, cabe em L1 e e reusado por todos os paineis de A. Isso tira
//! do caminho ~2.4 MB de trafego por convolucao por imagem.

use crate::gemm::{micro_kernel, transpose_to_panel, PackedA, KC, MR, NR};

#[derive(Clone, Copy)]
pub struct ConvSpec {
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub h: usize,
    pub w: usize,
    pub oh: usize,
    pub ow: usize,
}

impl ConvSpec {
    #[inline]
    pub fn ckk(&self) -> usize {
        self.cin * self.k * self.k
    }
    #[inline]
    pub fn hw(&self) -> usize {
        self.oh * self.ow
    }
    /// (linha, coluna) do pixel de saida de indice linear `o`.
    #[inline]
    fn pix(&self, o: usize) -> (usize, usize) {
        let oy = o / self.ow;
        (oy, o - oy * self.ow)
    }
}

/// Decomposicao (canal, kh, kw) de cada linha da matriz im2col.
///
/// Calcular isso na hora custa quatro divisoes inteiras por linha, e o laco
/// interno visita a mesma linha uma vez por painel B. Montar a tabela uma vez
/// por camada (ckk entradas, cabe em L1) tira ~300 mil divisoes por imagem.
pub struct RowMap {
    d: Vec<[u32; 3]>,
}

impl RowMap {
    pub fn new(sp: &ConvSpec) -> RowMap {
        let mut d = Vec::with_capacity(sp.ckk());
        for ci in 0..sp.cin {
            for kh in 0..sp.k {
                for kw in 0..sp.k {
                    d.push([ci as u32, kh as u32, kw as u32]);
                }
            }
        }
        RowMap { d }
    }

    #[inline]
    fn get(&self, r: usize) -> (usize, usize, usize) {
        let v = self.d[r];
        (v[0] as usize, v[1] as usize, v[2] as usize)
    }
}

/// dst[0..n] recebe a entrada dos pixels de saida a partir de (oy, ox) no
/// deslocamento (ci, kh, kw); zero fora da borda.
///
/// Percorre uma linha de saida por vez, entao com stride 1 cada trecho vira uma
/// copia contigua. O chamador passa (oy, ox) ja resolvidos porque todas as
/// chamadas de um mesmo painel partem do mesmo pixel.
#[inline]
fn gather_span(
    x: &[f32],
    ci: usize,
    kh: usize,
    kw: usize,
    sp: &ConvSpec,
    oy0: usize,
    ox0: usize,
    n: usize,
    dst: &mut [f32],
) {
    let (h, w, ow, stride, pad) = (sp.h, sp.w, sp.ow, sp.stride, sp.pad);
    let base = ci * h * w;
    let (mut oy, mut ox, mut t) = (oy0, ox0, 0);
    while t < n {
        let len = (n - t).min(ow - ox);
        let iy = oy * stride + kh;
        if iy < pad || iy >= h + pad {
            dst[t..t + len].fill(0.0);
        } else {
            let rb = base + (iy - pad) * w;
            if stride == 1 {
                let lo = pad.saturating_sub(kw + ox).min(len);
                let hi = (w + pad).saturating_sub(kw + ox).min(len);
                dst[t..t + lo].fill(0.0);
                if hi > lo {
                    let s = rb + ox + kw + lo - pad;
                    dst[t + lo..t + hi].copy_from_slice(&x[s..s + hi - lo]);
                }
                dst[t + hi..t + len].fill(0.0);
            } else {
                for u in 0..len {
                    let ix = (ox + u) * stride + kw;
                    dst[t + u] = if ix < pad || ix >= w + pad {
                        0.0
                    } else {
                        x[rb + ix - pad]
                    };
                }
            }
        }
        t += len;
        oy += 1;
        ox = 0;
    }
}

/// Inverso do `gather_span`: acumula src de volta na imagem (col2im fundido).
#[inline]
fn scatter_span(
    dx: &mut [f32],
    ci: usize,
    kh: usize,
    kw: usize,
    sp: &ConvSpec,
    oy0: usize,
    ox0: usize,
    n: usize,
    src: &[f32],
) {
    let (h, w, ow, stride, pad) = (sp.h, sp.w, sp.ow, sp.stride, sp.pad);
    let base = ci * h * w;
    let (mut oy, mut ox, mut t) = (oy0, ox0, 0);
    while t < n {
        let len = (n - t).min(ow - ox);
        let iy = oy * stride + kh;
        if iy >= pad && iy < h + pad {
            let rb = base + (iy - pad) * w;
            if stride == 1 {
                // o trecho vira um acumulo contiguo, vetorizavel
                let lo = pad.saturating_sub(kw + ox).min(len);
                let hi = (w + pad).saturating_sub(kw + ox).min(len);
                if hi > lo {
                    let s0 = rb + ox + kw + lo - pad;
                    for (d, v) in dx[s0..s0 + hi - lo].iter_mut().zip(&src[t + lo..t + hi]) {
                        *d += v;
                    }
                }
            } else {
                for u in 0..len {
                    let ix = (ox + u) * stride + kw;
                    if ix >= pad && ix < w + pad {
                        dx[rb + ix - pad] += src[t + u];
                    }
                }
            }
        }
        t += len;
        oy += 1;
        ox = 0;
    }
}

/// Y[cout x hw] = W[cout x ckk] * im2col(X).
pub fn forward_image(
    sp: &ConvSpec,
    rm: &RowMap,
    wpack: &PackedA,
    x: &[f32],
    y: &mut [f32],
    bp: &mut [f32],
) {
    let (ckk, hw) = (sp.ckk(), sp.hw());
    let mpan = (sp.cout + MR - 1) / MR;
    let mut q = 0;
    let mut pc = 0;
    while pc < ckk {
        let kc = KC.min(ckk - pc);
        let mut j = 0;
        while j < hw {
            let nr = NR.min(hw - j);
            let (oy, ox) = sp.pix(j);
            for p in 0..kc {
                let (ci, kh, kw) = rm.get(pc + p);
                let dst = &mut bp[p * NR..(p + 1) * NR];
                dst[nr..].fill(0.0);
                gather_span(x, ci, kh, kw, sp, oy, ox, nr, dst);
            }
            for i in 0..mpan {
                let mr = MR.min(sp.cout - i * MR);
                let off = i * MR * hw + j;
                micro_kernel(kc, wpack.panel(q, i, kc), bp, &mut y[off..], hw, mr, nr, q > 0);
            }
            j += NR;
        }
        pc += KC;
        q += 1;
    }
}

/// dX = col2im(W^T[ckk x cout] * dY[cout x hw]), com o col2im fundido na saida
/// do micro-kernel (o dcol nunca existe na memoria).
pub fn dx_image(
    sp: &ConvSpec,
    rm: &RowMap,
    wtpack: &PackedA,
    dy: &[f32],
    dx: &mut [f32],
    bp: &mut [f32],
    tile: &mut [f32],
) {
    let (ckk, hw) = (sp.ckk(), sp.hw());
    let mpan = (ckk + MR - 1) / MR;
    dx.fill(0.0);
    let mut q = 0;
    let mut pc = 0;
    while pc < sp.cout {
        let kc = KC.min(sp.cout - pc);
        let mut j = 0;
        while j < hw {
            let nr = NR.min(hw - j);
            let (oy, ox) = sp.pix(j);
            for p in 0..kc {
                let src = &dy[(pc + p) * hw + j..(pc + p) * hw + j + nr];
                let dst = &mut bp[p * NR..(p + 1) * NR];
                dst[..nr].copy_from_slice(src);
                dst[nr..].fill(0.0);
            }
            for i in 0..mpan {
                let mr = MR.min(ckk - i * MR);
                micro_kernel(kc, wtpack.panel(q, i, kc), bp, tile, NR, mr, nr, false);
                for r in 0..mr {
                    let (ci, kh, kw) = rm.get(i * MR + r);
                    scatter_span(dx, ci, kh, kw, sp, oy, ox, nr, &tile[r * NR..]);
                }
            }
            j += NR;
        }
        pc += KC;
        q += 1;
    }
}

/// Empacota im2col(X)^T: linhas = pixels de saida pc..pc+kc, colunas = j..j+nr
/// da dimensao ckk.
///
/// O painel esta transposto em relacao ao forward, o que num laco ingenuo vira
/// um gather escalar com uma condicional de borda por elemento. Em vez disso
/// cada uma das NR linhas e lida de forma contigua para `tmp` e o bloco inteiro
/// e transposto em registradores.
fn pack_colt(
    sp: &ConvSpec,
    rm: &RowMap,
    x: &[f32],
    pc: usize,
    kc: usize,
    j: usize,
    nr: usize,
    tmp: &mut [f32],
    bp: &mut [f32],
) {
    let (oy, ox) = sp.pix(pc);
    for t in 0..nr {
        let (ci, kh, kw) = rm.get(j + t);
        gather_span(x, ci, kh, kw, sp, oy, ox, kc, &mut tmp[t * KC..t * KC + kc]);
    }
    for t in nr..NR {
        tmp[t * KC..t * KC + kc].fill(0.0);
    }
    transpose_to_panel(tmp, KC, kc, bp);
}

/// dW[cout x ckk] += dY[cout x hw] * im2col(X)^T.
///
/// dY entra empacotado: cada painel e reusado ckk/NR vezes (72x na camada mais
/// larga), entao empacotar se paga de sobra, e o micro-kernel deixa de ler seis
/// fluxos separados por `hw` floats -- que, com hw potencia de dois, caem todos
/// no mesmo conjunto de L1.
pub fn dw_image(
    sp: &ConvSpec,
    rm: &RowMap,
    dypack: &mut PackedA,
    dy: &[f32],
    x: &[f32],
    dw: &mut [f32],
    tmp: &mut [f32],
    bp: &mut [f32],
) {
    let (ckk, hw) = (sp.ckk(), sp.hw());
    let mpan = (sp.cout + MR - 1) / MR;
    dypack.pack(dy, sp.cout, hw);
    let mut q = 0;
    let mut pc = 0;
    while pc < hw {
        let kc = KC.min(hw - pc);
        let mut j = 0;
        while j < ckk {
            let nr = NR.min(ckk - j);
            pack_colt(sp, rm, x, pc, kc, j, nr, tmp, bp);
            for i in 0..mpan {
                let mr = MR.min(sp.cout - i * MR);
                micro_kernel(
                    kc,
                    dypack.panel(q, i, kc),
                    bp,
                    &mut dw[i * MR * ckk + j..],
                    ckk,
                    mr,
                    nr,
                    true,
                );
            }
            j += NR;
        }
        pc += KC;
        q += 1;
    }
}
