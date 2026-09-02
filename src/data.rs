//! Loader do CIFAR-100 (binary version), taxonomia e augmentation.

use crate::rng::Rng;
use rayon::prelude::*;
use std::fs;
use std::path::Path;

pub const IMG: usize = 32;
pub const CH: usize = 3;
pub const PIX: usize = CH * IMG * IMG;
pub const NCLASS: usize = 100;

const MEAN: [f32; 3] = [0.5071, 0.4865, 0.4409];
const STD: [f32; 3] = [0.2673, 0.2564, 0.2762];

pub struct Dataset {
    /// bytes brutos em layout CHW, u8
    pub images: Vec<u8>,
    /// classe fina, 0..99 (byte 1 do registro)
    pub labels: Vec<u8>,
    /// superclasse, 0..19 (byte 0 do registro, que o loader antigo descartava)
    pub coarse: Vec<u8>,
    pub len: usize,
}

impl Dataset {
    fn from_bin(path: &Path) -> std::io::Result<Dataset> {
        let raw = fs::read(path)?;
        let rec = 2 + PIX; // coarse label, fine label, 3072 bytes
        assert!(
            raw.len() % rec == 0,
            "arquivo {:?} com tamanho inesperado ({} bytes)",
            path,
            raw.len()
        );
        let len = raw.len() / rec;
        let mut images = vec![0u8; len * PIX];
        let mut labels = vec![0u8; len];
        let mut coarse = vec![0u8; len];
        for i in 0..len {
            let o = i * rec;
            coarse[i] = raw[o];
            labels[i] = raw[o + 1];
            images[i * PIX..(i + 1) * PIX].copy_from_slice(&raw[o + 2..o + rec]);
        }
        Ok(Dataset {
            images,
            labels,
            coarse,
            len,
        })
    }
}

pub fn load(dir: &str) -> std::io::Result<(Dataset, Dataset)> {
    let d = Path::new(dir);
    let train = Dataset::from_bin(&d.join("train.bin"))?;
    let test = Dataset::from_bin(&d.join("test.bin"))?;
    Ok((train, test))
}

/// Arvore de 20 superclasses com 5 classes finas cada, derivada do proprio
/// arquivo e nao codificada a mao.
pub struct Taxo {
    /// superclasse de cada classe fina
    pub parent: Vec<u32>,
    /// classes finas de cada superclasse
    pub groups: Vec<Vec<u32>>,
    pub ncoarse: usize,
}

pub fn taxonomy(ds: &Dataset) -> Taxo {
    let mut parent = vec![u32::MAX; NCLASS];
    for i in 0..ds.len {
        let (f, c) = (ds.labels[i] as usize, ds.coarse[i] as u32);
        if parent[f] == u32::MAX {
            parent[f] = c;
        } else {
            assert_eq!(parent[f], c, "classe fina {f} com duas superclasses");
        }
    }
    assert!(parent.iter().all(|&p| p != u32::MAX), "classe fina sem pai");
    let ncoarse = *parent.iter().max().unwrap() as usize + 1;
    let mut groups = vec![Vec::new(); ncoarse];
    for (f, &p) in parent.iter().enumerate() {
        groups[p as usize].push(f as u32);
    }
    Taxo {
        parent,
        groups,
        ncoarse,
    }
}

#[inline]
fn norm(v: u8, c: usize) -> f32 {
    (v as f32 / 255.0 - MEAN[c]) / STD[c]
}

// ---------------------------------------------------------------- RandAugment

const NOPS: usize = 13;

#[inline]
fn px(im: &[u8], c: usize, y: isize, x: isize) -> u8 {
    if y < 0 || x < 0 || y >= IMG as isize || x >= IMG as isize {
        128
    } else {
        im[c * IMG * IMG + y as usize * IMG + x as usize]
    }
}

/// Reamostragem afim inversa com interpolacao bilinear.
fn affine(im: &mut [u8], a: f32, b: f32, tx: f32, c_: f32, d: f32, ty: f32) {
    let src = im.to_vec();
    let (cy, cx) = (IMG as f32 / 2.0 - 0.5, IMG as f32 / 2.0 - 0.5);
    for ch in 0..CH {
        for y in 0..IMG {
            for x in 0..IMG {
                let (fy, fx) = (y as f32 - cy, x as f32 - cx);
                let sx = a * fx + b * fy + tx + cx;
                let sy = c_ * fx + d * fy + ty + cy;
                let (x0, y0) = (sx.floor(), sy.floor());
                let (wx, wy) = (sx - x0, sy - y0);
                let (x0, y0) = (x0 as isize, y0 as isize);
                let p = |dy: isize, dx: isize| px(&src, ch, y0 + dy, x0 + dx) as f32;
                let v = p(0, 0) * (1.0 - wx) * (1.0 - wy)
                    + p(0, 1) * wx * (1.0 - wy)
                    + p(1, 0) * (1.0 - wx) * wy
                    + p(1, 1) * wx * wy;
                im[ch * IMG * IMG + y * IMG + x] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn blend(im: &mut [u8], other: &[u8], f: f32) {
    for i in 0..im.len() {
        let v = other[i] as f32 + f * (im[i] as f32 - other[i] as f32);
        im[i] = v.clamp(0.0, 255.0) as u8;
    }
}

fn grayscale(im: &[u8]) -> Vec<u8> {
    let mut g = vec![0u8; im.len()];
    for p in 0..IMG * IMG {
        let v = 0.299 * im[p] as f32
            + 0.587 * im[IMG * IMG + p] as f32
            + 0.114 * im[2 * IMG * IMG + p] as f32;
        let v = v.clamp(0.0, 255.0) as u8;
        for c in 0..CH {
            g[c * IMG * IMG + p] = v;
        }
    }
    g
}

/// Uma operacao do RandAugment. `v` em [0,1] e a magnitude ja normalizada.
fn apply_op(im: &mut [u8], op: usize, v: f32, rng: &mut Rng) {
    let sign = if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 };
    match op {
        0 => affine(im, 1.0, sign * 0.3 * v, 0.0, 0.0, 1.0, 0.0), // shear X
        1 => affine(im, 1.0, 0.0, 0.0, sign * 0.3 * v, 1.0, 0.0), // shear Y
        2 => affine(im, 1.0, 0.0, sign * 10.0 * v, 0.0, 1.0, 0.0), // translate X
        3 => affine(im, 1.0, 0.0, 0.0, 0.0, 1.0, sign * 10.0 * v), // translate Y
        4 => {
            let t = sign * 30.0 * v * std::f32::consts::PI / 180.0;
            affine(im, t.cos(), -t.sin(), 0.0, t.sin(), t.cos(), 0.0)
        }
        5 => {
            // posterize: 8 -> 4 bits
            let bits = 8 - (4.0 * v) as u32;
            let mask = !((1u16 << (8 - bits)) - 1) as u8;
            im.iter_mut().for_each(|p| *p &= mask);
        }
        6 => {
            // solarize
            let th = (255.0 * (1.0 - v)) as u8;
            im.iter_mut().for_each(|p| {
                if *p >= th {
                    *p = 255 - *p;
                }
            });
        }
        7 => {
            // autocontraste por canal
            for c in 0..CH {
                let s = &mut im[c * IMG * IMG..(c + 1) * IMG * IMG];
                let (lo, hi) = s.iter().fold((255u8, 0u8), |(l, h), &p| (l.min(p), h.max(p)));
                if hi > lo {
                    let k = 255.0 / (hi - lo) as f32;
                    s.iter_mut()
                        .for_each(|p| *p = (((*p - lo) as f32) * k).clamp(0.0, 255.0) as u8);
                }
            }
        }
        8 => {
            // equalizacao de histograma por canal
            for c in 0..CH {
                let s = &mut im[c * IMG * IMG..(c + 1) * IMG * IMG];
                let mut hist = [0u32; 256];
                s.iter().for_each(|&p| hist[p as usize] += 1);
                let mut cdf = [0u32; 256];
                let mut acc = 0;
                for i in 0..256 {
                    acc += hist[i];
                    cdf[i] = acc;
                }
                let lo = cdf.iter().find(|&&v| v > 0).cloned().unwrap_or(0);
                let den = (s.len() as u32).saturating_sub(lo).max(1);
                s.iter_mut()
                    .for_each(|p| *p = (((cdf[*p as usize] - lo) * 255) / den) as u8);
            }
        }
        9 => {
            // brilho
            let f = 1.0 + sign * 0.9 * v;
            let z = vec![0u8; im.len()];
            blend(im, &z, f);
        }
        10 => {
            // saturacao
            let f = 1.0 + sign * 0.9 * v;
            let g = grayscale(im);
            blend(im, &g, f);
        }
        11 => {
            // contraste
            let f = 1.0 + sign * 0.9 * v;
            let g = grayscale(im);
            let mean =
                (g[..IMG * IMG].iter().map(|&p| p as u32).sum::<u32>() / (IMG * IMG) as u32) as u8;
            let m = vec![mean; im.len()];
            blend(im, &m, f);
        }
        _ => {
            // nitidez: mistura com um borrado 3x3
            let f = 1.0 + sign * 0.9 * v;
            let src = im.to_vec();
            let mut bl = src.clone();
            for c in 0..CH {
                for y in 1..IMG - 1 {
                    for x in 1..IMG - 1 {
                        let mut s = 0.0f32;
                        for dy in 0..3isize {
                            for dx in 0..3isize {
                                let w = if dy == 1 && dx == 1 { 5.0 } else { 1.0 / 8.0 * 3.0 };
                                s += w * px(&src, c, y as isize + dy - 1, x as isize + dx - 1) as f32;
                            }
                        }
                        bl[c * IMG * IMG + y * IMG + x] = (s / 8.0).clamp(0.0, 255.0) as u8;
                    }
                }
            }
            blend(im, &bl, f);
        }
    }
}

/// Parametros de augmentation, variaveis ao longo do treino.
#[derive(Clone, Copy)]
pub struct Aug {
    pub pad: usize,
    /// magnitude do RandAugment em [0,1]; 0 desliga
    pub ra_mag: f32,
    pub ra_n: usize,
    /// probabilidade do random erasing
    pub erase: f32,
    pub erase_side: usize,
}

impl Default for Aug {
    fn default() -> Aug {
        Aug {
            pad: 4,
            ra_mag: 0.0,
            ra_n: 2,
            erase: 0.0,
            erase_side: 8,
        }
    }
}

/// Monta um batch: RandAugment sobre os bytes, depois recorte, espelho,
/// normalizacao e random erasing.
pub fn make_batch_train(
    ds: &Dataset,
    idx: &[usize],
    aug: &Aug,
    seed: u64,
    out: &mut [f32],
    labels: &mut [u32],
    coarse: &mut [u32],
) {
    for (b, &i) in idx.iter().enumerate() {
        labels[b] = ds.labels[i] as u32;
        coarse[b] = ds.coarse[i] as u32;
    }

    out.par_chunks_mut(PIX)
        .zip(idx.par_iter())
        .enumerate()
        .for_each(|(bi, (dst, &i))| {
            let mut rng = Rng::new(seed ^ (bi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut work = [0u8; PIX];
            work.copy_from_slice(&ds.images[i * PIX..(i + 1) * PIX]);

            if aug.ra_mag > 0.0 {
                for _ in 0..aug.ra_n {
                    let op = rng.below(NOPS);
                    // mstd 0.5: a magnitude sorteia em torno do valor nominal
                    let v = (aug.ra_mag * (1.0 + 0.5 * rng.normal())).clamp(0.0, 1.0);
                    apply_op(&mut work, op, v, &mut rng);
                }
            }

            let pad = aug.pad;
            let (dy, dx) = if pad > 0 {
                (
                    rng.below(2 * pad + 1) as isize - pad as isize,
                    rng.below(2 * pad + 1) as isize - pad as isize,
                )
            } else {
                (0, 0)
            };
            let flip = rng.next_u32() & 1 == 1;
            for c in 0..CH {
                for y in 0..IMG {
                    let sy = y as isize + dy;
                    let row = &mut dst[c * IMG * IMG + y * IMG..c * IMG * IMG + (y + 1) * IMG];
                    if sy < 0 || sy >= IMG as isize {
                        row.iter_mut().for_each(|v| *v = -MEAN[c] / STD[c]);
                        continue;
                    }
                    let sbase = c * IMG * IMG + sy as usize * IMG;
                    for x in 0..IMG {
                        let sx = (if flip { IMG - 1 - x } else { x }) as isize + dx;
                        row[x] = if sx < 0 || sx >= IMG as isize {
                            -MEAN[c] / STD[c]
                        } else {
                            norm(work[sbase + sx as usize], c)
                        };
                    }
                }
            }

            // random erasing: ruido normalizado, nao zero
            if aug.erase > 0.0 && rng.uniform() < aug.erase {
                let (cy, cx) = (rng.below(IMG) as isize, rng.below(IMG) as isize);
                let h = aug.erase_side as isize / 2;
                for c in 0..CH {
                    for y in (cy - h).max(0)..(cy + h).min(IMG as isize) {
                        for x in (cx - h).max(0)..(cx + h).min(IMG as isize) {
                            dst[c * IMG * IMG + y as usize * IMG + x as usize] = rng.normal();
                        }
                    }
                }
            }
        });
}

/// Mistura o lote consigo mesmo permutado. Retorna (permutacao, lambda por
/// imagem); `lam = 1` significa que aquela imagem nao foi misturada.
pub fn mix_batch(
    x: &mut [f32],
    bs: usize,
    alpha_mix: f32,
    alpha_cut: f32,
    switch: f32,
    rng: &mut Rng,
) -> (Vec<usize>, Vec<f32>) {
    let mut perm: Vec<usize> = (0..bs).collect();
    rng.shuffle(&mut perm);
    let mut lam = vec![1.0f32; bs];
    if alpha_mix <= 0.0 && alpha_cut <= 0.0 {
        return (perm, lam);
    }
    let cut = alpha_cut > 0.0 && (alpha_mix <= 0.0 || rng.uniform() < switch);
    // Beta(a,a) aproximada por uma media de uniformes: basta um lambda com
    // massa nas pontas, e evita implementar a gama.
    let a = if cut { alpha_cut } else { alpha_mix };
    let u = rng.uniform();
    let l = if a >= 1.0 {
        u
    } else {
        // a < 1 concentra nas pontas
        if u < 0.5 {
            (2.0 * u).powf(1.0 / a) * 0.5
        } else {
            1.0 - (2.0 * (1.0 - u)).powf(1.0 / a) * 0.5
        }
    };

    if cut {
        let r = (1.0 - l).sqrt();
        let (bh, bw) = (
            ((IMG as f32 * r) as usize).min(IMG),
            ((IMG as f32 * r) as usize).min(IMG),
        );
        let (cy, cx) = (rng.below(IMG), rng.below(IMG));
        let (y0, y1) = (cy.saturating_sub(bh / 2), (cy + bh / 2).min(IMG));
        let (x0, x1) = (cx.saturating_sub(bw / 2), (cx + bw / 2).min(IMG));
        let area = ((y1 - y0) * (x1 - x0)) as f32 / (IMG * IMG) as f32;
        let src: Vec<f32> = x[..bs * PIX].to_vec();
        for b in 0..bs {
            for c in 0..CH {
                for y in y0..y1 {
                    for xx in x0..x1 {
                        let o = c * IMG * IMG + y * IMG + xx;
                        x[b * PIX + o] = src[perm[b] * PIX + o];
                    }
                }
            }
            lam[b] = 1.0 - area;
        }
    } else {
        let src: Vec<f32> = x[..bs * PIX].to_vec();
        for b in 0..bs {
            let (p, o) = (perm[b] * PIX, b * PIX);
            for i in 0..PIX {
                x[o + i] = l * src[o + i] + (1.0 - l) * src[p + i];
            }
            lam[b] = l;
        }
    }
    (perm, lam)
}

/// Batch de avaliacao: apenas normalizacao. `flip` espelha (TTA).
pub fn make_batch_eval(
    ds: &Dataset,
    start: usize,
    bs: usize,
    flip: bool,
    out: &mut [f32],
    labels: &mut [u32],
) {
    for b in 0..bs {
        labels[b] = ds.labels[start + b] as u32;
    }
    out[..bs * PIX]
        .par_chunks_mut(PIX)
        .enumerate()
        .for_each(|(b, dst)| {
            let src = &ds.images[(start + b) * PIX..(start + b + 1) * PIX];
            for c in 0..CH {
                for y in 0..IMG {
                    for x in 0..IMG {
                        let sx = if flip { IMG - 1 - x } else { x };
                        dst[c * IMG * IMG + y * IMG + x] = norm(src[c * IMG * IMG + y * IMG + sx], c);
                    }
                }
            }
        });
}
