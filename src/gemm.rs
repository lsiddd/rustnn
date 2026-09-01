//! Nucleo de multiplicacao de matrizes: micro-kernel AVX2/FMA 6x16 e empacotamento
//! do operando A. O operando B e produzido direto pelas rotinas de convolucao
//! (`conv.rs`), sem materializar a matriz im2col.

pub const MR: usize = 6;
pub const NR: usize = 16;
pub const KC: usize = 256;

#[cfg(target_arch = "x86_64")]
static HAS_AVX2: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
});

/// A empacotado em paineis de kc x MR, um grupo por bloco de KC da dimensao K.
pub struct PackedA {
    d: Vec<f32>,
    mpad: usize,
}

impl PackedA {
    pub fn new() -> PackedA {
        PackedA {
            d: Vec::new(),
            mpad: 0,
        }
    }

    fn prepare(&mut self, m: usize, k: usize) -> (usize, usize) {
        let mpan = (m + MR - 1) / MR;
        self.mpad = mpan * MR;
        let nb = (k + KC - 1) / KC;
        self.d.resize(nb * KC * self.mpad, 0.0);
        (mpan, nb)
    }

    /// `a` e [m x k] row-major.
    pub fn pack(&mut self, a: &[f32], m: usize, k: usize) {
        let (mpan, nb) = self.prepare(m, k);
        for q in 0..nb {
            let pc = q * KC;
            let kc = KC.min(k - pc);
            let blk = q * KC * self.mpad;
            for i in 0..mpan {
                let panel = &mut self.d[blk + i * kc * MR..blk + (i + 1) * kc * MR];
                for r in 0..MR {
                    let row = i * MR + r;
                    if row >= m {
                        for p in 0..kc {
                            panel[p * MR + r] = 0.0;
                        }
                        continue;
                    }
                    let src = &a[row * k + pc..row * k + pc + kc];
                    for p in 0..kc {
                        panel[p * MR + r] = src[p];
                    }
                }
            }
        }
    }

    /// Empacota a transposta: `a` e [k x m] row-major, trata como [m x k].
    pub fn pack_t(&mut self, a: &[f32], m: usize, k: usize) {
        let (mpan, nb) = self.prepare(m, k);
        for q in 0..nb {
            let pc = q * KC;
            let kc = KC.min(k - pc);
            let blk = q * KC * self.mpad;
            for i in 0..mpan {
                let panel = &mut self.d[blk + i * kc * MR..blk + (i + 1) * kc * MR];
                for p in 0..kc {
                    let src = &a[(pc + p) * m..];
                    for r in 0..MR {
                        let row = i * MR + r;
                        panel[p * MR + r] = if row < m { src[row] } else { 0.0 };
                    }
                }
            }
        }
    }

    #[inline]
    pub fn panel(&self, q: usize, i: usize, kc: usize) -> &[f32] {
        let blk = q * KC * self.mpad;
        &self.d[blk + i * kc * MR..blk + (i + 1) * kc * MR]
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_avx2(
    kc: usize,
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    mr: usize,
    nr: usize,
    accumulate: bool,
) {
    use std::arch::x86_64::*;
    // 6x16 acumuladores = 12 ymm, + 2 para B e 1 para o broadcast de A: cabe nos 16.
    let mut c0 = [_mm256_setzero_ps(); MR];
    let mut c1 = [_mm256_setzero_ps(); MR];
    let (mut apk, mut bpk) = (ap.as_ptr(), bp.as_ptr());
    for _ in 0..kc {
        let b0 = _mm256_loadu_ps(bpk);
        let b1 = _mm256_loadu_ps(bpk.add(8));
        for i in 0..MR {
            let a = _mm256_broadcast_ss(&*apk.add(i));
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
        apk = apk.add(MR);
        bpk = bpk.add(NR);
    }
    if mr == MR && nr == NR {
        let cp = c.as_mut_ptr();
        for i in 0..MR {
            let d = cp.add(i * ldc);
            if accumulate {
                _mm256_storeu_ps(d, _mm256_add_ps(_mm256_loadu_ps(d), c0[i]));
                _mm256_storeu_ps(d.add(8), _mm256_add_ps(_mm256_loadu_ps(d.add(8)), c1[i]));
            } else {
                _mm256_storeu_ps(d, c0[i]);
                _mm256_storeu_ps(d.add(8), c1[i]);
            }
        }
    } else {
        let mut tmp = [0.0f32; MR * NR];
        for i in 0..MR {
            _mm256_storeu_ps(tmp.as_mut_ptr().add(i * NR), c0[i]);
            _mm256_storeu_ps(tmp.as_mut_ptr().add(i * NR + 8), c1[i]);
        }
        for i in 0..mr {
            let row = &mut c[i * ldc..i * ldc + nr];
            if accumulate {
                for j in 0..nr {
                    row[j] += tmp[i * NR + j];
                }
            } else {
                row[..nr].copy_from_slice(&tmp[i * NR..i * NR + nr]);
            }
        }
    }
}

#[inline(always)]
fn micro_scalar(
    kc: usize,
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    mr: usize,
    nr: usize,
    accumulate: bool,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..kc {
        let bv = &bp[p * NR..p * NR + NR];
        let av = &ap[p * MR..p * MR + MR];
        for i in 0..MR {
            let a = av[i];
            let ai = &mut acc[i];
            for j in 0..NR {
                ai[j] += a * bv[j];
            }
        }
    }
    for i in 0..mr {
        let row = &mut c[i * ldc..i * ldc + nr];
        if accumulate {
            for j in 0..nr {
                row[j] += acc[i][j];
            }
        } else {
            row[..nr].copy_from_slice(&acc[i][..nr]);
        }
    }
}

/// C[mr x nr] (+)= Apanel[MR x kc] * Bpanel[kc x NR].
#[inline]
pub fn micro_kernel(
    kc: usize,
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    mr: usize,
    nr: usize,
    accumulate: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2 {
            unsafe { micro_avx2(kc, ap, bp, c, ldc, mr, nr, accumulate) };
            return;
        }
    }
    micro_scalar(kc, ap, bp, c, ldc, mr, nr, accumulate);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose8x8(src: *const f32, lds: usize, dst: *mut f32, ldd: usize) {
    use std::arch::x86_64::*;
    let r = |i: usize| _mm256_loadu_ps(src.add(i * lds));
    let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
    let (r4, r5, r6, r7) = (r(4), r(5), r(6), r(7));
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);
    let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    _mm256_storeu_ps(dst, _mm256_permute2f128_ps(u0, u4, 0x20));
    _mm256_storeu_ps(dst.add(ldd), _mm256_permute2f128_ps(u1, u5, 0x20));
    _mm256_storeu_ps(dst.add(2 * ldd), _mm256_permute2f128_ps(u2, u6, 0x20));
    _mm256_storeu_ps(dst.add(3 * ldd), _mm256_permute2f128_ps(u3, u7, 0x20));
    _mm256_storeu_ps(dst.add(4 * ldd), _mm256_permute2f128_ps(u0, u4, 0x31));
    _mm256_storeu_ps(dst.add(5 * ldd), _mm256_permute2f128_ps(u1, u5, 0x31));
    _mm256_storeu_ps(dst.add(6 * ldd), _mm256_permute2f128_ps(u2, u6, 0x31));
    _mm256_storeu_ps(dst.add(7 * ldd), _mm256_permute2f128_ps(u3, u7, 0x31));
}

/// bp[p*NR + t] = src[t*lds + p], para p em 0..kc e t em 0..NR.
///
/// Serve para montar um painel B a partir de NR linhas ja lidas de forma
/// contigua: em vez de um gather escalar por elemento, sao copias de linha
/// inteira mais uma transposicao 8x8 em registradores (~0.5 instrucao por
/// elemento contra ~9 do caminho escalar).
pub fn transpose_to_panel(src: &[f32], lds: usize, kc: usize, bp: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX2 {
            let n8 = kc & !7;
            unsafe {
                let (sp, dp) = (src.as_ptr(), bp.as_mut_ptr());
                let mut p = 0;
                while p < n8 {
                    transpose8x8(sp.add(p), lds, dp.add(p * NR), NR);
                    transpose8x8(sp.add(8 * lds + p), lds, dp.add(p * NR + 8), NR);
                    p += 8;
                }
            }
            for p in n8..kc {
                for t in 0..NR {
                    bp[p * NR + t] = src[t * lds + p];
                }
            }
            return;
        }
    }
    for p in 0..kc {
        for t in 0..NR {
            bp[p * NR + t] = src[t * lds + p];
        }
    }
}
