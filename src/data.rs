//! Loader do CIFAR-100 (binary version) + augmentation.

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
    pub labels: Vec<u8>,
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
        for i in 0..len {
            let o = i * rec;
            labels[i] = raw[o + 1]; // fine label (0..99)
            images[i * PIX..(i + 1) * PIX].copy_from_slice(&raw[o + 2..o + rec]);
        }
        Ok(Dataset {
            images,
            labels,
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

#[inline]
fn norm(v: u8, c: usize) -> f32 {
    (v as f32 / 255.0 - MEAN[c]) / STD[c]
}

/// Monta um batch aplicando random crop (pad 4), flip horizontal e cutout.
pub fn make_batch_train(
    ds: &Dataset,
    idx: &[usize],
    pad: usize,
    cutout: usize,
    seed: u64,
    out: &mut [f32],
    labels: &mut [u32],
) {
    labels
        .iter_mut()
        .zip(idx)
        .for_each(|(l, &i)| *l = ds.labels[i] as u32);

    out.par_chunks_mut(PIX)
        .zip(idx.par_iter())
        .enumerate()
        .for_each(|(bi, (dst, &i))| {
            let mut rng = Rng::new(seed ^ (bi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let src = &ds.images[i * PIX..(i + 1) * PIX];
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
                        for v in row.iter_mut() {
                            *v = -MEAN[c] / STD[c];
                        }
                        continue;
                    }
                    let sbase = c * IMG * IMG + sy as usize * IMG;
                    for x in 0..IMG {
                        let sx = (if flip { IMG - 1 - x } else { x }) as isize + dx;
                        row[x] = if sx < 0 || sx >= IMG as isize {
                            -MEAN[c] / STD[c]
                        } else {
                            norm(src[sbase + sx as usize], c)
                        };
                    }
                }
            }
            if cutout > 0 {
                let cy = rng.below(IMG) as isize;
                let cx = rng.below(IMG) as isize;
                let h = cutout as isize / 2;
                for c in 0..CH {
                    for y in (cy - h).max(0)..(cy + h).min(IMG as isize) {
                        for x in (cx - h).max(0)..(cx + h).min(IMG as isize) {
                            dst[c * IMG * IMG + y as usize * IMG + x as usize] = 0.0;
                        }
                    }
                }
            }
        });
}

/// Batch de avaliacao: apenas normalizacao.
pub fn make_batch_eval(ds: &Dataset, start: usize, bs: usize, out: &mut [f32], labels: &mut [u32]) {
    for b in 0..bs {
        labels[b] = ds.labels[start + b] as u32;
    }
    out[..bs * PIX]
        .par_chunks_mut(PIX)
        .enumerate()
        .for_each(|(b, dst)| {
            let src = &ds.images[(start + b) * PIX..(start + b + 1) * PIX];
            for c in 0..CH {
                for p in 0..IMG * IMG {
                    dst[c * IMG * IMG + p] = norm(src[c * IMG * IMG + p], c);
                }
            }
        });
}
