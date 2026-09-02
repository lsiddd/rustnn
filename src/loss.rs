//! Perdas: alvos suaves com taxonomia, entropia cruzada, destilacao e hint.

use rayon::prelude::*;

/// Alvo suave de uma imagem, com a massa de suavizacao concentrada nos irmaos.
///
/// A suavizacao uniforme diz ao modelo que todo erro e igualmente errado, o que
/// e falso: a matriz de confusao do CIFAR-100 e dominada por trocas dentro da
/// mesma superclasse. Aqui `sib` da fracao de eps vai para as outras quatro
/// classes da superclasse e o resto se espalha pelas 95 restantes.
pub fn taxo_target(y: usize, groups: &[Vec<u32>], parent: &[u32], eps: f32, sib: f32, out: &mut [f32]) {
    let nf = out.len();
    let g = &groups[parent[y] as usize];
    let k = g.len();
    let off_far = if nf > k {
        eps * (1.0 - sib) / (nf - k) as f32
    } else {
        0.0
    };
    out.iter_mut().for_each(|v| *v = off_far);
    if k > 1 {
        let off_sib = eps * sib / (k - 1) as f32;
        for &c in g.iter() {
            out[c as usize] = off_sib;
        }
    }
    out[y] = 1.0 - eps;
}

pub fn uniform_target(y: usize, eps: f32, out: &mut [f32]) {
    let k = out.len() as f32;
    out.iter_mut().for_each(|v| *v = eps / k);
    out[y] += 1.0 - eps;
}

/// Entropia cruzada com alvos suaves. Escreve `scale * dz/dn` em `dz`.
///
/// Retorna (perda media, acertos contra `hard`).
pub fn ce_soft(
    logits: &[f32],
    target: &[f32],
    hard: &[u32],
    nclass: usize,
    scale: f32,
    dz: &mut [f32],
) -> (f32, usize) {
    let n = hard.len();
    let inv = 1.0 / n as f32;
    let out: Vec<(f32, usize)> = logits
        .par_chunks(nclass)
        .zip(target.par_chunks(nclass))
        .zip(dz.par_chunks_mut(nclass))
        .zip(hard.par_iter())
        .map(|(((z, t), d), &y)| {
            let mut mx = f32::NEG_INFINITY;
            let mut arg = 0usize;
            for (j, &v) in z.iter().enumerate() {
                if v > mx {
                    mx = v;
                    arg = j;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..nclass {
                let e = (z[j] - mx).exp();
                d[j] = e;
                sum += e;
            }
            let lse = mx + sum.ln();
            let mut l = 0.0f32;
            for j in 0..nclass {
                l += t[j] * (lse - z[j]);
                d[j] = (d[j] / sum - t[j]) * inv * scale;
            }
            (l, (arg == y as usize) as usize)
        })
        .collect();
    let loss: f32 = out.iter().map(|x| x.0).sum();
    let correct: usize = out.iter().map(|x| x.1).sum();
    (loss * inv, correct)
}

/// KL(professor || aluno) com temperatura, o termo de autodestilacao.
///
/// O professor entra destacado do grafo: sem isso o gradiente volta pelo alvo e
/// o treino colapsa em silencio, sem erro e sem NaN.
pub fn kl_distill(
    student: &[f32],
    teacher: &[f32],
    nclass: usize,
    temp: f32,
    scale: f32,
    dz: &mut [f32],
) -> f32 {
    let n = student.len() / nclass;
    let inv = 1.0 / n as f32;
    let it = 1.0 / temp;
    let parts: Vec<f32> = student
        .par_chunks(nclass)
        .zip(teacher.par_chunks(nclass))
        .zip(dz.par_chunks_mut(nclass))
        .map(|((zs, zt), d)| {
            let soft = |z: &[f32], o: &mut [f32]| {
                let mx = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0;
                for j in 0..nclass {
                    let e = ((z[j] - mx) * it).exp();
                    o[j] = e;
                    s += e;
                }
                let is = 1.0 / s;
                o.iter_mut().for_each(|v| *v *= is);
            };
            let mut ps = vec![0.0f32; nclass];
            let mut pt = vec![0.0f32; nclass];
            soft(zs, &mut ps);
            soft(zt, &mut pt);
            let mut l = 0.0f32;
            for j in 0..nclass {
                if pt[j] > 1e-9 {
                    l += pt[j] * (pt[j].max(1e-9).ln() - ps[j].max(1e-9).ln());
                }
                // d/dz_s de T^2 * KL = T * (ps - pt)
                d[j] += temp * (ps[j] - pt[j]) * inv * scale;
            }
            l * temp * temp
        })
        .collect();
    parts.iter().sum::<f32>() * inv
}

/// Erro quadratico entre a feature de uma saida rasa e a da profunda.
pub fn hint_mse(pred: &[f32], target: &[f32], n: usize, scale: f32, dz: &mut [f32]) -> f32 {
    let c = pred.len() / n;
    let inv = 1.0 / (n * c) as f32;
    let l: f32 = pred
        .par_chunks(c)
        .zip(target.par_chunks(c))
        .zip(dz.par_chunks_mut(c))
        .map(|((p, t), d)| {
            let mut s = 0.0f32;
            for j in 0..c {
                let e = p[j] - t[j];
                s += e * e;
                d[j] += 2.0 * e * inv * scale;
            }
            s
        })
        .sum();
    l * inv * scale
}
