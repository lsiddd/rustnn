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

/// Reduz a profundidade de bits. `v` vem clampado em [0,1] por
/// `make_batch_train`, entao `bits` fica em [4,8] e o deslocamento nao estoura.
fn posterize(im: &mut [u8], v: f32) {
    let bits = 8 - (4.0 * v) as u32;
    let mask = !((1u16 << (8 - bits)) - 1) as u8;
    im.iter_mut().for_each(|p| *p &= mask);
}

fn solarize(im: &mut [u8], v: f32) {
    let th = (255.0 * (1.0 - v)) as u8;
    im.iter_mut().for_each(|p| {
        if *p >= th {
            *p = 255 - *p;
        }
    });
}

/// Estica cada canal para ocupar os 256 niveis.
fn autocontrast(im: &mut [u8]) {
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

/// Equalizacao de histograma por canal.
fn equalize(im: &mut [u8]) {
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
        let lo = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
        let den = (s.len() as u32).saturating_sub(lo).max(1);
        s.iter_mut()
            .for_each(|p| *p = (((cdf[*p as usize] - lo) * 255) / den) as u8);
    }
}

fn brightness(im: &mut [u8], f: f32) {
    let black = vec![0u8; im.len()];
    blend(im, &black, f);
}

fn saturation(im: &mut [u8], f: f32) {
    let gray = grayscale(im);
    blend(im, &gray, f);
}

fn contrast(im: &mut [u8], f: f32) {
    let gray = grayscale(im);
    let mean = (gray[..IMG * IMG].iter().map(|&p| p as u32).sum::<u32>() / (IMG * IMG) as u32) as u8;
    let flat = vec![mean; im.len()];
    blend(im, &flat, f);
}

/// Um tap do borrado 3x3: centro com peso 5, vizinhos com 3/8, tudo sobre 8.
#[inline]
fn blur_tap(src: &[u8], c: usize, y: usize, x: usize) -> u8 {
    let mut s = 0.0f32;
    for dy in 0..3isize {
        for dx in 0..3isize {
            let w = if dy == 1 && dx == 1 { 5.0 } else { 1.0 / 8.0 * 3.0 };
            s += w * px(src, c, y as isize + dy - 1, x as isize + dx - 1) as f32;
        }
    }
    (s / 8.0).clamp(0.0, 255.0) as u8
}

/// Mistura com um borrado 3x3: `f > 1` afia, `f < 1` suaviza.
fn sharpness(im: &mut [u8], f: f32) {
    let src = im.to_vec();
    let mut bl = src.clone();
    for c in 0..CH {
        for y in 1..IMG - 1 {
            for x in 1..IMG - 1 {
                bl[c * IMG * IMG + y * IMG + x] = blur_tap(&src, c, y, x);
            }
        }
    }
    blend(im, &bl, f);
}

/// Uma operacao do RandAugment. `v` em [0,1] e a magnitude ja normalizada.
///
/// O sinal e sorteado sempre, inclusive para as operacoes que o ignoram: a
/// sequencia do gerador faz parte do resultado reproduzivel por semente.
fn apply_op(im: &mut [u8], op: usize, v: f32, rng: &mut Rng) {
    let sign = if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 };
    let f = 1.0 + sign * 0.9 * v;
    match op {
        0 => affine(im, 1.0, sign * 0.3 * v, 0.0, 0.0, 1.0, 0.0), // shear X
        1 => affine(im, 1.0, 0.0, 0.0, sign * 0.3 * v, 1.0, 0.0), // shear Y
        2 => affine(im, 1.0, 0.0, sign * 10.0 * v, 0.0, 1.0, 0.0), // translate X
        3 => affine(im, 1.0, 0.0, 0.0, 0.0, 1.0, sign * 10.0 * v), // translate Y
        4 => {
            let t = sign * 30.0 * v * std::f32::consts::PI / 180.0;
            affine(im, t.cos(), -t.sin(), 0.0, t.sin(), t.cos(), 0.0)
        }
        5 => posterize(im, v),
        6 => solarize(im, v),
        7 => autocontrast(im),
        8 => equalize(im),
        9 => brightness(im, f),
        10 => saturation(im, f),
        11 => contrast(im, f),
        _ => sharpness(im, f),
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

/// Aplica `ra_n` operacoes sorteadas do RandAugment sobre os bytes crus.
fn randaugment(work: &mut [u8], aug: &Aug, rng: &mut Rng) {
    if aug.ra_mag <= 0.0 {
        return;
    }
    for _ in 0..aug.ra_n {
        let op = rng.below(NOPS);
        // mstd 0.5: a magnitude sorteia em torno do valor nominal
        let v = (aug.ra_mag * (1.0 + 0.5 * rng.normal())).clamp(0.0, 1.0);
        apply_op(work, op, v, rng);
    }
}

/// Deslocamento do recorte dentro da moldura de `pad` pixels.
fn crop_offset(pad: usize, rng: &mut Rng) -> (isize, isize) {
    if pad == 0 {
        return (0, 0);
    }
    (
        rng.below(2 * pad + 1) as isize - pad as isize,
        rng.below(2 * pad + 1) as isize - pad as isize,
    )
}

/// Recorta com deslocamento, espelha e normaliza para dentro de `dst`.
///
/// O que cai fora da imagem recebe `-mean/std`, que e o zero da escala
/// normalizada: preencher com zero cru poria um cinza deslocado na borda.
/// Com `dy = dx = 0` nenhum pixel cai fora, e e assim que a avaliacao usa.
fn crop_flip_normalize(src: &[u8], dst: &mut [f32], dy: isize, dx: isize, flip: bool) {
    for c in 0..CH {
        let border = -MEAN[c] / STD[c];
        for y in 0..IMG {
            let sy = y as isize + dy;
            let row = &mut dst[c * IMG * IMG + y * IMG..c * IMG * IMG + (y + 1) * IMG];
            if (0..IMG as isize).contains(&sy) {
                normalize_row(src, c, sy as usize, dx, flip, border, row);
            } else {
                row.fill(border);
            }
        }
    }
}

/// Uma linha ja sabidamente dentro da imagem na vertical. Fica separada so
/// para o laco de cima caber na cabeca; `inline` garante que o codigo gerado
/// seja o mesmo de antes.
#[inline]
fn normalize_row(src: &[u8], c: usize, sy: usize, dx: isize, flip: bool, border: f32, row: &mut [f32]) {
    let sbase = c * IMG * IMG + sy * IMG;
    for x in 0..IMG {
        let sx = (if flip { IMG - 1 - x } else { x }) as isize + dx;
        row[x] = if (0..IMG as isize).contains(&sx) {
            norm(src[sbase + sx as usize], c)
        } else {
            border
        };
    }
}

/// Random erasing com ruido normalizado, nao com zero: zero e um valor valido
/// nessa escala e a rede aprenderia a ignorar o retangulo.
fn random_erase(dst: &mut [f32], aug: &Aug, rng: &mut Rng) {
    if aug.erase <= 0.0 || rng.uniform() >= aug.erase {
        return;
    }
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

/// Monta um batch: RandAugment sobre os bytes, depois recorte, espelho,
/// normalizacao e random erasing.
///
/// Cada imagem tem seu proprio gerador derivado da semente e do indice no lote,
/// entao o resultado nao depende de como o rayon reparte o trabalho.
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

            randaugment(&mut work, aug, &mut rng);
            let (dy, dx) = crop_offset(aug.pad, &mut rng);
            let flip = rng.next_u32() & 1 == 1;
            crop_flip_normalize(&work, dst, dy, dx, flip);
            random_erase(dst, aug, &mut rng);
        });
}

/// Lambda com massa nas pontas, no lugar de uma Beta(a,a) de verdade.
///
/// A mistura so precisa de um lambda que caia perto de 0 ou de 1 na maior parte
/// das vezes; implementar a funcao gama para isso nao se paga.
fn beta_like(a: f32, rng: &mut Rng) -> f32 {
    let u = rng.uniform();
    if a >= 1.0 {
        return u;
    }
    // a < 1 concentra nas pontas
    if u < 0.5 {
        (2.0 * u).powf(1.0 / a) * 0.5
    } else {
        1.0 - (2.0 * (1.0 - u)).powf(1.0 / a) * 0.5
    }
}

/// Cola um retangulo da imagem permutada sobre a original. O lambda vira a
/// fracao de area que sobrou, e nao o valor sorteado, porque o recorte foi
/// aparado nas bordas.
fn cutmix(x: &mut [f32], bs: usize, perm: &[usize], l: f32, rng: &mut Rng, lam: &mut [f32]) {
    let r = (1.0 - l).sqrt();
    let side = ((IMG as f32 * r) as usize).min(IMG);
    let (cy, cx) = (rng.below(IMG), rng.below(IMG));
    let (y0, y1) = (cy.saturating_sub(side / 2), (cy + side / 2).min(IMG));
    let (x0, x1) = (cx.saturating_sub(side / 2), (cx + side / 2).min(IMG));
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
}

/// Interpolacao pixel a pixel entre a imagem e a permutada.
fn mixup(x: &mut [f32], bs: usize, perm: &[usize], l: f32, lam: &mut [f32]) {
    let src: Vec<f32> = x[..bs * PIX].to_vec();
    for b in 0..bs {
        let (p, o) = (perm[b] * PIX, b * PIX);
        for i in 0..PIX {
            x[o + i] = l * src[o + i] + (1.0 - l) * src[p + i];
        }
        lam[b] = l;
    }
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
    let l = beta_like(if cut { alpha_cut } else { alpha_mix }, rng);
    if cut {
        cutmix(x, bs, &perm, l, rng, &mut lam);
    } else {
        mixup(x, bs, &perm, l, &mut lam);
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
            crop_flip_normalize(src, dst, 0, 0, flip);
        });
}

#[cfg(test)]
mod tests {
    use super::{beta_like, crop_flip_normalize, crop_offset, mix_batch, norm, CH, IMG, PIX};
    use crate::rng::Rng;

    /// Uma imagem em que cada pixel codifica sua posicao, para o espelho e o
    /// deslocamento serem verificaveis por igualdade.
    fn rampa() -> Vec<u8> {
        (0..PIX).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn beta_like_fica_no_intervalo() {
        let mut rng = Rng::new(5);
        for a in [0.2f32, 0.8, 1.0, 4.0] {
            for _ in 0..2000 {
                let l = beta_like(a, &mut rng);
                assert!((0.0..=1.0).contains(&l), "a={a} deu {l}");
            }
        }
    }

    #[test]
    fn crop_offset_respeita_a_moldura() {
        let mut rng = Rng::new(11);
        assert_eq!(crop_offset(0, &mut rng), (0, 0));
        for _ in 0..2000 {
            let (dy, dx) = crop_offset(4, &mut rng);
            assert!((-4..=4).contains(&dy) && (-4..=4).contains(&dx));
        }
    }

    /// Sem deslocamento nenhum pixel cai fora, entao a saida e exatamente a
    /// normalizacao da entrada. E o caso que a avaliacao usa.
    #[test]
    fn sem_deslocamento_e_so_normalizacao() {
        let src = rampa();
        let mut dst = vec![0.0f32; PIX];
        crop_flip_normalize(&src, &mut dst, 0, 0, false);
        for c in 0..CH {
            for i in 0..IMG * IMG {
                let o = c * IMG * IMG + i;
                assert_eq!(dst[o], norm(src[o], c));
            }
        }
    }

    #[test]
    fn flip_espelha_as_colunas() {
        let src = rampa();
        let (mut reto, mut espelhado) = (vec![0.0f32; PIX], vec![0.0f32; PIX]);
        crop_flip_normalize(&src, &mut reto, 0, 0, false);
        crop_flip_normalize(&src, &mut espelhado, 0, 0, true);
        for c in 0..CH {
            for y in 0..IMG {
                for x in 0..IMG {
                    let base = c * IMG * IMG + y * IMG;
                    assert_eq!(espelhado[base + x], reto[base + IMG - 1 - x]);
                }
            }
        }
        // Espelhar duas vezes volta ao original.
        let mut duplo = vec![0.0f32; PIX];
        crop_flip_normalize(&src, &mut duplo, 0, 0, true);
        assert_ne!(duplo, reto, "a rampa nao e simetrica, o flip tem de mudar algo");
    }

    /// Deslocar a imagem inteira para fora tem de deixar so a borda.
    #[test]
    fn deslocamento_grande_so_deixa_borda() {
        let src = rampa();
        let mut dst = vec![0.0f32; PIX];
        crop_flip_normalize(&src, &mut dst, IMG as isize, 0, false);
        for c in 0..CH {
            let borda = -super::MEAN[c] / super::STD[c];
            for i in 0..IMG * IMG {
                assert_eq!(dst[c * IMG * IMG + i], borda);
            }
        }
    }

    /// Com os dois alphas em zero a mistura tem de ser identidade: `lam = 1`
    /// para todos e nenhum pixel tocado.
    #[test]
    fn mix_desligado_e_identidade() {
        let bs = 4;
        let original: Vec<f32> = (0..bs * PIX).map(|i| i as f32 * 0.001).collect();
        let mut x = original.clone();
        let mut rng = Rng::new(2);
        let (perm, lam) = mix_batch(&mut x, bs, 0.0, 0.0, 0.5, &mut rng);
        assert_eq!(x, original);
        assert!(lam.iter().all(|&l| l == 1.0));
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..bs).collect::<Vec<_>>());
    }

    /// Ligada, a mistura devolve lambdas em [0,1] e nao sai da escala.
    #[test]
    fn mix_ligado_produz_lambda_valido() {
        let bs = 8;
        let mut rng = Rng::new(17);
        for (m, c) in [(0.8f32, 0.0f32), (0.0, 1.0), (0.8, 1.0)] {
            let mut x: Vec<f32> = (0..bs * PIX).map(|i| (i % 97) as f32 * 0.01).collect();
            let (_, lam) = mix_batch(&mut x, bs, m, c, 0.5, &mut rng);
            assert!(lam.iter().all(|&l| (0.0..=1.0).contains(&l)), "{lam:?}");
            assert!(x.iter().all(|v| v.is_finite()));
        }
    }
}
