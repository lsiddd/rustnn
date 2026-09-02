//! Muon com Newton-Schulz, AdamW, clip de norma e media exponencial.
//!
//! O Muon substitui o gradiente das matrizes 2D ocultas pela matriz ortogonal
//! mais proxima do momento, aproximada por iteracoes de um polinomio quintico em
//! `X X^T`. Ele entrega tipicamente 1,4 a 2x de eficiencia amostral, que e a
//! moeda em jogo quando o orcamento sao cem epocas.
//!
//! O metodo cai bem numa CPU por um motivo que nao vale numa GPU. Na GPU a
//! iteracao e uma sequencia de matmuls pequenos, limitados por latencia de
//! lancamento, e ela rouba tempo de um passo ja limitado por banda. Aqui o passo
//! e limitado por computacao, e a iteracao e GEMM denso puro nas formas que o
//! micro-kernel 6x16 prefere: com lote 256 e tres iteracoes, custa cerca de 8%
//! do passo. Tres iteracoes bastam; a ortogonalizacao exata nao e necessaria.

use crate::gemm::{gemm_small, PackedA, PackedB};
use crate::nn::Param;
use rayon::prelude::*;

/// Elementos a partir dos quais o laco elemento a elemento compensa ser paralelo.
const PAR_MIN: usize = 1 << 16;

/// Coeficientes do quintico do Muon, ajustados para convergir rapido perto de
/// zero em vez de convergir exatamente para 1.
const NS_A: f32 = 3.4445;
const NS_B: f32 = -4.7750;
const NS_C: f32 = 2.0315;

#[derive(Default)]
pub struct NsScratch {
    x: Vec<f32>,
    a: Vec<f32>,
    aa: Vec<f32>,
    t: Vec<f32>,
    ap: PackedA,
    bp: PackedB,
}

/// Ortogonaliza `g` [rows x cols] no lugar, por `steps` iteracoes.
pub fn newton_schulz(g: &mut [f32], rows: usize, cols: usize, steps: usize, s: &mut NsScratch) {
    let (m, n, tr) = if rows <= cols {
        (rows, cols, false)
    } else {
        (cols, rows, true)
    };
    s.x.resize(m * n, 0.0);
    if tr {
        for i in 0..rows {
            for j in 0..cols {
                s.x[j * rows + i] = g[i * cols + j];
            }
        }
    } else {
        s.x.copy_from_slice(g);
    }

    let norm = s.x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-7;
    let inv = 1.0 / norm;
    s.x.iter_mut().for_each(|v| *v *= inv);
    s.a.resize(m * m, 0.0);
    s.aa.resize(m * m, 0.0);
    s.t.resize(m * n, 0.0);

    for _ in 0..steps {
        // A = X X^T
        gemm_small(
            m, m, n, &s.x, false, &s.x, true, &mut s.a, false, &mut s.ap, &mut s.bp,
        );
        // AA = A A
        gemm_small(
            m, m, m, &s.a, false, &s.a, false, &mut s.aa, false, &mut s.ap, &mut s.bp,
        );
        // B = b A + c A A, reaproveitando o buffer de A
        for i in 0..m * m {
            s.a[i] = NS_B * s.a[i] + NS_C * s.aa[i];
        }
        // X <- a X + B X
        gemm_small(
            m, n, m, &s.a, false, &s.x, false, &mut s.t, false, &mut s.ap, &mut s.bp,
        );
        for i in 0..m * n {
            s.x[i] = NS_A * s.x[i] + s.t[i];
        }
    }

    if tr {
        for i in 0..rows {
            for j in 0..cols {
                g[i * cols + j] = s.x[j * rows + i];
            }
        }
    } else {
        g.copy_from_slice(&s.x);
    }
}

pub struct Optim {
    pub muon: bool,
    pub ns_steps: usize,
    pub mom: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub t: usize,
}

impl Optim {
    pub fn new(muon: bool, ns_steps: usize) -> Optim {
        Optim {
            muon,
            ns_steps,
            mom: 0.95,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            t: 0,
        }
    }

    /// Atualiza todos os parametros de uma vez.
    ///
    /// O Muon e paralelo **por parametro**, e nao por dentro de cada
    /// Newton-Schulz. As matrizes do modelo sao pequenas (a maior tem 442 mil
    /// elementos), e paralelizar cada produto punha o custo de despacho do rayon
    /// em cada uma das ~270 chamadas por passo. Com trinta matrizes e catorze
    /// threads, o balanceamento de fora ja e suficiente.
    pub fn step_all(&mut self, params: &mut [&mut Param], lr: f32, wd: f32) {
        let (mut muon, mut adam): (Vec<&mut &mut Param>, Vec<&mut &mut Param>) = params
            .iter_mut()
            .partition(|p| self.muon && p.rows > 0 && p.rows.min(p.cols) > 1);
        let (mom, ns) = (self.mom, self.ns_steps);
        muon.par_iter_mut().for_each_init(NsScratch::default, |sc, p| {
            let n = p.v.len();
            let wdp = if p.decay { wd } else { 0.0 };
            for i in 0..n {
                p.m[i] = mom * p.m[i] + p.g[i];
                p.g[i] += mom * p.m[i];
            }
            let (rows, cols) = (p.rows, p.cols);
            newton_schulz(&mut p.g, rows, cols, ns, sc);
            let scale = 0.2 * (rows.max(cols) as f32).sqrt();
            for i in 0..n {
                p.v[i] -= lr * (scale * p.g[i] + wdp * p.v[i]);
                p.g[i] = 0.0;
            }
        });
        for p in adam.iter_mut() {
            self.step_adam(p, lr, wd);
        }
    }

    fn step_adam(&self, p: &mut Param, lr: f32, wd: f32) {
        let wd = if p.decay { wd } else { 0.0 };
        let n = p.v.len();
        if p.v2.len() != n {
            p.v2 = vec![0.0; n];
        }
        let (b1, b2, eps) = (self.b1, self.b2, self.eps);
        let c1 = 1.0 - b1.powi(self.t as i32 + 1);
        let c2 = 1.0 - b2.powi(self.t as i32 + 1);
        let body = |v: &mut f32, g: &mut f32, m: &mut f32, v2: &mut f32| {
            *m = b1 * *m + (1.0 - b1) * *g;
            *v2 = b2 * *v2 + (1.0 - b2) * *g * *g;
            *v -= lr * ((*m / c1) / ((*v2 / c2).sqrt() + eps) + wd * *v);
            *g = 0.0;
        };
        if n >= PAR_MIN {
            p.v.par_iter_mut()
                .zip(p.g.par_iter_mut())
                .zip(p.m.par_iter_mut())
                .zip(p.v2.par_iter_mut())
                .for_each(|(((v, g), m), v2)| body(v, g, m, v2));
        } else {
            for i in 0..n {
                let (mut vv, mut gg, mut mm, mut v2v) = (p.v[i], p.g[i], p.m[i], p.v2[i]);
                body(&mut vv, &mut gg, &mut mm, &mut v2v);
                (p.v[i], p.g[i], p.m[i], p.v2[i]) = (vv, gg, mm, v2v);
            }
        }
    }
}

/// Norma global do gradiente; se passar de `max`, reescala tudo no lugar.
pub fn clip_(params: &mut Vec<&mut Param>, max: f32) -> f32 {
    let sq: f64 = params
        .par_iter()
        .map(|p| p.g.par_iter().map(|&g| (g as f64) * (g as f64)).sum::<f64>())
        .sum();
    let norm = sq.sqrt() as f32;
    if norm > max && norm > 0.0 {
        let s = max / norm;
        params
            .par_iter_mut()
            .for_each(|p| p.g.par_iter_mut().for_each(|g| *g *= s));
    }
    norm
}

/// Media exponencial dos pesos, mantida em paralelo ao treino.
pub struct Ema {
    pub d: f32,
    pub w: Vec<Vec<f32>>,
    ready: bool,
    t: usize,
}

impl Ema {
    pub fn new(d: f32) -> Ema {
        Ema {
            d,
            w: Vec::new(),
            ready: false,
            t: 0,
        }
    }

    pub fn update(&mut self, params: &[&mut Param]) {
        self.t += 1;
        if !self.ready {
            self.w = params.iter().map(|p| p.v.clone()).collect();
            self.ready = true;
            return;
        }
        // Correcao de vies: com decay 0.9998 a constante de tempo e de cinco mil
        // passos, e no comeco do treino a media ainda seria basicamente a
        // inicializacao. O teto cresce com t e some depois de algumas centenas
        // de passos.
        let d = self.d.min((1.0 + self.t as f32) / (10.0 + self.t as f32));
        self.w.par_iter_mut().zip(params.par_iter()).for_each(|(e, p)| {
            e.par_iter_mut()
                .zip(p.v.par_iter())
                .for_each(|(a, b)| *a = d * *a + (1.0 - d) * b);
        });
    }

    /// Troca os pesos do modelo pelos da media, devolvendo os originais.
    pub fn swap_in(&mut self, params: &mut Vec<&mut Param>) -> Vec<Vec<f32>> {
        let mut old = Vec::with_capacity(params.len());
        for (p, e) in params.iter_mut().zip(self.w.iter()) {
            old.push(p.v.clone());
            p.v.copy_from_slice(e);
        }
        old
    }

    pub fn swap_out(params: &mut Vec<&mut Param>, old: Vec<Vec<f32>>) {
        for (p, o) in params.iter_mut().zip(old) {
            p.v.copy_from_slice(&o);
        }
    }
}
