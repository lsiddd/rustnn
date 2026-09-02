//! Nucleo de multiplicacao de matrizes: micro-kernel AVX2/FMA 6x16 e
//! empacotamento dos dois operandos.
//!
//! A convolucao (`conv.rs`) produz o painel B direto da imagem, sem materializar
//! a matriz im2col. O transformer (`linear.rs`, `attn.rs`) e a ortogonalizacao do
//! Muon (`optim.rs`) usam o driver `matmul` do fim deste arquivo, que empacota os
//! dois lados e paraleliza sobre paineis de linha de C.

use rayon::prelude::*;

pub const MR: usize = 6;
pub const NR: usize = 16;
pub const KC: usize = 256;
/// Elementos a partir dos quais o empacotamento compensa ser paralelo.
const PAR_MIN: usize = 1 << 18;

#[cfg(target_arch = "x86_64")]
static HAS_AVX2: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));

/// A empacotado em paineis de kc x MR, um grupo por bloco de KC da dimensao K.
#[derive(Default)]
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
        let mpan = m.div_ceil(MR);
        self.mpad = mpan * MR;
        let nb = k.div_ceil(KC);
        let need = nb * KC * self.mpad;
        if self.d.len() < need {
            self.d.resize(need, 0.0);
        }
        (mpan, nb)
    }

    /// `a` e [m x k] row-major.
    ///
    /// O empacotamento e paralelo acima de um limiar. Nas formas do transformer
    /// o operando A e a matriz de ativacoes inteira -- dezenas de MB -- e
    /// deixa-lo serial punha treze threads esperando uma. Abaixo do limiar
    /// (matrizes por cabeca da atencao, ja dentro de uma regiao paralela) o
    /// caminho serial evita aninhar rayon a toa.
    pub fn pack(&mut self, a: &[f32], m: usize, k: usize) {
        let (mpan, nb) = self.prepare(m, k);
        let mpad = self.mpad;
        for q in 0..nb {
            let pc = q * KC;
            let kc = KC.min(k - pc);
            let blk = q * KC * mpad;
            let dst = &mut self.d[blk..blk + kc * mpan * MR];
            let body = |i: usize, panel: &mut [f32]| {
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
            };
            if m * kc >= PAR_MIN {
                dst.par_chunks_mut(kc * MR)
                    .enumerate()
                    .for_each(|(i, p)| body(i, p));
            } else {
                dst.chunks_mut(kc * MR).enumerate().for_each(|(i, p)| body(i, p));
            }
        }
    }

    /// Empacota a transposta: `a` e [k x m] row-major, trata como [m x k].
    pub fn pack_t(&mut self, a: &[f32], m: usize, k: usize) {
        let (mpan, nb) = self.prepare(m, k);
        let mpad = self.mpad;
        for q in 0..nb {
            let pc = q * KC;
            let kc = KC.min(k - pc);
            let blk = q * KC * mpad;
            let dst = &mut self.d[blk..blk + kc * mpan * MR];
            let body = |i: usize, panel: &mut [f32]| {
                for p in 0..kc {
                    let src = &a[(pc + p) * m..];
                    for r in 0..MR {
                        let row = i * MR + r;
                        panel[p * MR + r] = if row < m { src[row] } else { 0.0 };
                    }
                }
            };
            if m * kc >= PAR_MIN {
                dst.par_chunks_mut(kc * MR)
                    .enumerate()
                    .for_each(|(i, p)| body(i, p));
            } else {
                dst.chunks_mut(kc * MR).enumerate().for_each(|(i, p)| body(i, p));
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

// ---------------------------------------------------------------- operando B

/// B empacotado em paineis de kc x NR, um grupo por bloco de KC da dimensao K.
///
/// A convolucao monta o painel B direto da imagem e nunca precisa desta
/// estrutura. O transformer precisa: os dois operandos sao matrizes densas de
/// verdade, e o painel de B e reusado por todos os paineis de A (M/6 vezes, o
/// que da milhares de reusos com o lote inteiro numa matriz so).
#[derive(Default)]
pub struct PackedB {
    d: Vec<f32>,
    npad: usize,
}

impl PackedB {
    pub fn new() -> PackedB {
        PackedB {
            d: Vec::new(),
            npad: 0,
        }
    }

    fn prepare(&mut self, n: usize, k: usize) -> (usize, usize) {
        let npan = n.div_ceil(NR);
        self.npad = npan * NR;
        let nb = k.div_ceil(KC);
        let need = nb * KC * self.npad;
        if self.d.len() < need {
            self.d.resize(need, 0.0);
        }
        (npan, nb)
    }

    /// `b` e [k x n] row-major: cada painel sai de copias de linha contiguas.
    pub fn pack(&mut self, b: &[f32], n: usize, k: usize) {
        let (npan, nb) = self.prepare(n, k);
        let npad = self.npad;
        for q in 0..nb {
            let pc = q * KC;
            let kc = KC.min(k - pc);
            let blk = q * KC * npad;
            let dst = &mut self.d[blk..blk + kc * npan * NR];
            let body = |j: usize, panel: &mut [f32]| {
                let nr = NR.min(n - j * NR);
                for p in 0..kc {
                    let src = &b[(pc + p) * n + j * NR..(pc + p) * n + j * NR + nr];
                    let d = &mut panel[p * NR..(p + 1) * NR];
                    d[..nr].copy_from_slice(src);
                    d[nr..].fill(0.0);
                }
            };
            if n * kc >= PAR_MIN {
                dst.par_chunks_mut(kc * NR)
                    .enumerate()
                    .for_each(|(j, p)| body(j, p));
            } else {
                dst.chunks_mut(kc * NR).enumerate().for_each(|(j, p)| body(j, p));
            }
        }
    }

    /// Empacota a transposta: `b` e [n x k] row-major, o produto e A * B^T.
    ///
    /// E o caso do peso no forward (`Y = X W^T`) e de K^T na atencao. As NR
    /// linhas ficam contiguas, entao a transposicao 8x8 em registrador de
    /// `transpose_to_panel` se aplica direto.
    pub fn pack_t(&mut self, b: &[f32], n: usize, k: usize) {
        let (npan, nb) = self.prepare(n, k);
        let npad = self.npad;
        for q in 0..nb {
            let pc = q * KC;
            let kc = KC.min(k - pc);
            let blk = q * KC * npad;
            let dst = &mut self.d[blk..blk + kc * npan * NR];
            let body = |j: usize, panel: &mut [f32]| {
                let nr = NR.min(n - j * NR);
                if nr == NR {
                    transpose_to_panel(&b[(j * NR) * k + pc..], k, kc, panel);
                } else {
                    for p in 0..kc {
                        for t in 0..NR {
                            panel[p * NR + t] = if t < nr { b[(j * NR + t) * k + pc + p] } else { 0.0 };
                        }
                    }
                }
            };
            if n * kc >= PAR_MIN {
                dst.par_chunks_mut(kc * NR)
                    .enumerate()
                    .for_each(|(j, p)| body(j, p));
            } else {
                dst.chunks_mut(kc * NR).enumerate().for_each(|(j, p)| body(j, p));
            }
        }
    }

    #[inline]
    pub fn panel(&self, q: usize, j: usize, kc: usize) -> &[f32] {
        let blk = q * KC * self.npad;
        &self.d[blk + j * kc * NR..blk + (j + 1) * kc * NR]
    }
}

// ---------------------------------------------------------------- driver

thread_local! {
    /// Buffers de empacotamento reusados entre chamadas.
    ///
    /// Sao os alocamentos grandes do transformer (dezenas de MB com o lote
    /// inteiro numa matriz), e realoca-los por camada dominaria o tempo. Sao
    /// `thread_local` porque a atencao chama `matmul` de dentro de uma regiao
    /// paralela, com uma matriz pequena por (imagem, cabeca).
    static SCRATCH: std::cell::RefCell<(PackedA, PackedB)> =
        std::cell::RefCell::new((PackedA::new(), PackedB::new()));
}

/// C[m x n] = op(A[m x k]) * op(B), com op = transposta quando a flag esta ligada.
///
/// `at = false`: A e [m x k]. `at = true`: A e [k x m].
/// `bt = false`: B e [k x n]. `bt = true`: B e [n x k].
/// `acc = true` soma sobre o C existente em vez de sobrescrever.
///
/// As quatro combinacoes cobrem os tres produtos de uma camada densa: o forward
/// e `A * B^T`, o gradiente da entrada e `A * B`, e o dos pesos e `A^T * B`.
///
/// O paralelismo e sobre paineis de MR linhas de C. Com o lote inteiro numa
/// matriz isso da milhares de tarefas independentes, contra as dezenas de
/// regioes com barreira que a convolucao produz.
pub fn matmul(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    at: bool,
    b: &[f32],
    bt: bool,
    c: &mut [f32],
    acc: bool,
) {
    debug_assert!(a.len() >= m * k && c.len() >= m * n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !acc {
            c[..m * n].fill(0.0);
        }
        return;
    }
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let (ap, bp) = &mut *s;
        if at {
            ap.pack_t(a, m, k);
        } else {
            ap.pack(a, m, k);
        }
        if bt {
            bp.pack_t(b, n, k);
        } else {
            bp.pack(b, n, k);
        }
        let (ap, bp) = (&*ap, &*bp);
        let npan = n.div_ceil(NR);
        let nb = k.div_ceil(KC);
        c[..m * n].par_chunks_mut(MR * n).enumerate().for_each(|(i, cb)| {
            let mr = MR.min(m - i * MR);
            // j por fora e q por dentro: o tile 6x16 de C fica quente
            // enquanto a dimensao K inteira e percorrida.
            for j in 0..npan {
                let nr = NR.min(n - j * NR);
                for q in 0..nb {
                    let kc = KC.min(k - q * KC);
                    micro_kernel(
                        kc,
                        ap.panel(q, i, kc),
                        bp.panel(q, j, kc),
                        &mut cb[j * NR..],
                        n,
                        mr,
                        nr,
                        q > 0 || acc,
                    );
                }
            }
        });
    });
}

/// GEMM sequencial sobre o micro-kernel, com o scratch vindo do chamador.
///
/// E o caminho da atencao, onde `matmul` nao serve: as matrizes sao pequenas
/// (uma por imagem e cabeca) e o laco de fora ja e paralelo, entao empacotar num
/// buffer thread-local e paralelizar de novo por dentro so custaria barreira.
pub fn gemm_small(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    at: bool,
    b: &[f32],
    bt: bool,
    c: &mut [f32],
    acc: bool,
    ap: &mut PackedA,
    bp: &mut PackedB,
) {
    if at {
        ap.pack_t(a, m, k);
    } else {
        ap.pack(a, m, k);
    }
    if bt {
        bp.pack_t(b, n, k);
    } else {
        bp.pack(b, n, k);
    }
    let npan = n.div_ceil(NR);
    let nb = k.div_ceil(KC);
    let mpan = m.div_ceil(MR);
    for i in 0..mpan {
        let mr = MR.min(m - i * MR);
        for j in 0..npan {
            let nr = NR.min(n - j * NR);
            for q in 0..nb {
                let kc = KC.min(k - q * KC);
                micro_kernel(
                    kc,
                    ap.panel(q, i, kc),
                    bp.panel(q, j, kc),
                    &mut c[i * MR * n + j * NR..],
                    n,
                    mr,
                    nr,
                    q > 0 || acc,
                );
            }
        }
    }
}

/// C[m x n] += A^T[m x k] * B[k x n], paralelo sobre K.
///
/// E a forma do gradiente dos pesos de uma camada densa, onde K e o lote inteiro
/// (dezenas de milhares de linhas) e M, N sao pequenos. Pelo driver geral, o
/// operando A^T sozinho ocuparia dezenas de MB de empacotamento e a divisao por
/// paineis de M daria poucas tarefas mal balanceadas. Aqui cada worker fatia K,
/// acumula num C privado de poucas centenas de KB e a reducao vem no fim: e a
/// mesma estrutura que `Conv2d::backward_w` ja usa, pelo mesmo motivo.
pub fn matmul_dw(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    // Um dos dois operandos entra transposto, e o empacotamento transposto le em
    // trechos de MR floats separados por uma linha inteira: aproveita 24 dos 64
    // bytes de cada linha de cache. Vale escolher o menor lado. Na projecao QKV
    // isso troca uma passada transposta de 50 MB por uma de 17 MB; o resultado
    // sai transposto e a transposicao final custa uma matriz de pesos.
    let flip = n < m;
    let (mm, nn, aa, bb) = if flip { (n, m, b, a) } else { (m, n, a, b) };
    let rows = KC.min(k).max(1);
    let acc = aa[..k * mm]
        .par_chunks(rows * mm)
        .zip(bb[..k * nn].par_chunks(rows * nn))
        .fold(
            || (vec![0.0f32; mm * nn], PackedA::default(), PackedB::default()),
            |(mut acc, mut ap, mut bp), (ai, bi)| {
                let kk = ai.len() / mm;
                gemm_small(mm, nn, kk, ai, true, bi, false, &mut acc, true, &mut ap, &mut bp);
                (acc, ap, bp)
            },
        )
        .map(|(a, ..)| a)
        .reduce(
            || vec![0.0f32; mm * nn],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    if flip {
        c[..m * n].par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v += acc[j * m + i];
            }
        });
    } else {
        c[..m * n]
            .par_iter_mut()
            .zip(acc.par_iter())
            .for_each(|(x, y)| *x += y);
    }
}

/// Vetor sem inicializar, para buffers que o GEMM sobrescreve por inteiro.
///
/// Um `vec![0.0; n]` de dezenas de MB por camada custa a memset inteira a toa. O
/// alocador de `pool.rs` devolve blocos sujos, entao nem o truque de pagina
/// zerada do `calloc` se aplica: medido em rodadas alternadas, zerar custa 87
/// contra 68 img/s, 21% do passo inteiro.
///
/// # Safety
///
/// O chamador tem que escrever os `n` elementos antes de ler qualquer um deles.
/// Na pratica isso significa passar o vetor como destino de um `matmul` com
/// `acc = false`, que escreve todo o bloco `[m, n]`, ou preenche-lo na mao.
#[inline]
#[allow(clippy::uninit_vec)] // o contrato acima e o que torna isso valido
pub unsafe fn scratch_vec(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(n);
    v.set_len(n);
    v
}
