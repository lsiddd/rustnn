//! rustvit: hibrido hierarquico com memoria de alta resolucao.
//!
//! Convolucao onde a resolucao e alta e os tokens seriam muitos; atencao global
//! onde os tokens ja sao poucos. Sem janelas e sem mascara de deslocamento: a
//! 8x8 e 4x4 a janela seria o mapa inteiro de qualquer forma.
//!
//! Quatro coisas separam este modelo de um hibrido convencional. O mapa 16x16 do
//! tronco nao e descartado depois da tokenizacao: vira uma memoria enderecavel
//! que dois dos blocos leem por atencao cruzada. Duas saidas auxiliares dao
//! gradiente curto ao tronco e servem de aluno numa autodestilacao. A cabeca
//! usa a taxonomia de 20 superclasses que o arquivo do CIFAR-100 carrega no byte
//! 0 de cada registro. E os pesos 2D sao atualizados pelo Muon (`optim.rs`).

use crate::attn::{CrossAttn, SelfAttn};
use crate::linear::Linear;
use crate::model::BasicBlock;
use crate::nn::{gap_backward, gap_forward, BatchNorm2d, Conv2d, Param, Tensor};
use crate::norm::{DwConv3, LayerNorm, LayerScale, SeqPool, SwiGlu};
use crate::rng::Rng;
use crate::tok::{add_, Tok};
use rayon::prelude::*;
use std::io::{Read, Write};
use std::sync::Arc;

/// Descarta o ramo inteiro de uma imagem com probabilidade `p`.
///
/// A mascara e derivada de uma semente explicita, e nao de estado interno: o
/// gradcheck precisa que `L(t+ev)` e `L(t-ev)` sorteiem a mesma mascara, senao a
/// diferenca finita nao mede nada.
struct DropPath {
    p: f32,
    mask: Vec<f32>,
}

impl DropPath {
    fn new(p: f32) -> DropPath {
        DropPath { p, mask: Vec::new() }
    }

    fn forward(&mut self, mut x: Tok, train: bool, seed: u64) -> Tok {
        if !train || self.p <= 0.0 {
            return x;
        }
        let keep = 1.0 - self.p;
        let mut rng = Rng::new(seed);
        self.mask.clear();
        for _ in 0..x.n {
            self.mask
                .push(if rng.uniform() < keep { 1.0 / keep } else { 0.0 });
        }
        let m = &self.mask;
        x.d.par_chunks_mut(x.t * x.c).enumerate().for_each(|(i, r)| {
            let s = m[i];
            if s != 1.0 {
                r.iter_mut().for_each(|v| *v *= s);
            }
        });
        x
    }

    fn backward(&self, mut dy: Tok) -> Tok {
        if self.mask.is_empty() {
            return dy;
        }
        let m = &self.mask;
        dy.d.par_chunks_mut(dy.t * dy.c).enumerate().for_each(|(i, r)| {
            let s = m[i];
            if s != 1.0 {
                r.iter_mut().for_each(|v| *v *= s);
            }
        });
        dy
    }
}

// ---------------------------------------------------------------- bloco

/// pre-LN com tres ramos partindo da mesma normalizacao:
///
/// ```text
/// h = LN(x)
/// x = x + l1 * [ LSA(h) + gamma * DW3x3(h) ]
/// x = x + lm * CrossAttn(h -> memoria)        // so nos blocos leitores
/// x = x + l2 * SwiGLU(LN(x))
/// ```
pub struct Block {
    n1: LayerNorm,
    attn: SelfAttn,
    dw: DwConv3,
    /// peso por canal do ramo depthwise
    gamma: Param,
    ls1: LayerScale,
    cross: Option<CrossAttn>,
    lsm: Option<LayerScale>,
    n2: LayerNorm,
    mlp: SwiGlu,
    ls2: LayerScale,
    dp1: DropPath,
    dp2: DropPath,
    dp3: DropPath,
    c: usize,
    /// saida do ramo depthwise, para o gradiente de gamma
    dwo: Vec<f32>,
    idx: u64,
}

impl Block {
    pub fn new(
        c: usize,
        h: usize,
        th: usize,
        tw: usize,
        mem: Option<(usize, usize)>,
        depth: usize,
        drop: f32,
        idx: u64,
        ls_init: f32,
        rng: &mut Rng,
    ) -> Block {
        let hidden = (4 * c / 3).div_ceil(16) * 16;
        Block {
            n1: LayerNorm::new(c),
            attn: SelfAttn::new(c, h, th, tw, depth, rng),
            dw: DwConv3::new(c, th, tw, rng),
            gamma: Param::filled(c, 0.5, false),
            ls1: LayerScale::new(c, ls_init),
            cross: mem.map(|(mh, mw)| CrossAttn::new(c, h, th, tw, mh, mw, depth, rng)),
            lsm: mem.map(|_| LayerScale::new(c, ls_init)),
            n2: LayerNorm::new(c),
            mlp: SwiGlu::new(c, hidden, depth, rng),
            ls2: LayerScale::new(c, ls_init),
            dp1: DropPath::new(drop),
            dp2: DropPath::new(drop),
            dp3: DropPath::new(drop),
            c,
            dwo: Vec::new(),
            idx,
        }
    }

    pub fn forward(&mut self, x: Tok, mem: Option<&[f32]>, train: bool, seed: u64) -> Tok {
        let c = self.c;
        let h = self.n1.forward(&x, train);
        let mut a = self.attn.forward(&h, train);
        let d = self.dw.forward(&h, train);
        let gv = &self.gamma.v;
        a.d.par_chunks_mut(c).zip(d.d.par_chunks(c)).for_each(|(ar, dr)| {
            for j in 0..c {
                ar[j] += gv[j] * dr[j];
            }
        });
        if train {
            self.dwo = d.d;
        }
        let a = self
            .dp1
            .forward(self.ls1.forward(a, train), train, seed ^ self.idx);
        let mut y = x;
        add_(&mut y, &a);

        if let Some(cr) = &mut self.cross {
            let cm = cr.forward(&h, mem.expect("bloco leitor sem memoria"), train);
            let cm = self.dp2.forward(
                self.lsm.as_mut().unwrap().forward(cm, train),
                train,
                seed ^ self.idx ^ 0x5151,
            );
            add_(&mut y, &cm);
        }

        let h2 = self.n2.forward(&y, train);
        let m = self.mlp.forward(&h2, train);
        let m = self
            .dp3
            .forward(self.ls2.forward(m, train), train, seed ^ self.idx ^ 0xA2A2);
        add_(&mut y, &m);
        y
    }

    /// Retorna dx e acumula o gradiente da memoria em `dmem`.
    pub fn backward(&mut self, dz: Tok, mem: Option<&[f32]>, dmem: Option<&mut [f32]>) -> Tok {
        let c = self.c;
        let dm = self.ls2.backward(self.dp3.backward(dz.clone()));
        let dh2 = self.mlp.backward(&dm);
        let mut dy = dz;
        add_(&mut dy, &self.n2.backward(&dh2));

        let mut dh = Tok::zeros(dy.n, dy.t, dy.c);
        if let Some(cr) = &mut self.cross {
            let dc = self.lsm.as_mut().unwrap().backward(self.dp2.backward(dy.clone()));
            let dcx = cr.backward(&dc, mem.unwrap(), dmem.unwrap());
            add_(&mut dh, &dcx);
        }

        let da = self.ls1.backward(self.dp1.backward(dy.clone()));
        let dg =
            da.d.par_chunks(c)
                .zip(self.dwo.par_chunks(c))
                .fold(
                    || vec![0.0f32; c],
                    |mut acc, (ar, dr)| {
                        for j in 0..c {
                            acc[j] += ar[j] * dr[j];
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0.0f32; c],
                    |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(p, q)| *p += q);
                        a
                    },
                );
        self.gamma.g.iter_mut().zip(&dg).for_each(|(a, b)| *a += b);

        let mut ddw = da.zero_like();
        let gv = &self.gamma.v;
        ddw.d
            .par_chunks_mut(c)
            .zip(da.d.par_chunks(c))
            .for_each(|(dr, ar)| {
                for j in 0..c {
                    dr[j] = gv[j] * ar[j];
                }
            });
        add_(&mut dh, &self.attn.backward(&da));
        add_(&mut dh, &self.dw.backward(&ddw));

        add_(&mut dy, &self.n1.backward(&dh));
        dy
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        self.n1.for_each_param(f);
        self.attn.for_each_param(f);
        self.dw.for_each_param(f);
        f(&mut self.gamma);
        self.ls1.for_each_param(f);
        if let Some(c) = &mut self.cross {
            c.for_each_param(f);
        }
        if let Some(l) = &mut self.lsm {
            l.for_each_param(f);
        }
        self.n2.for_each_param(f);
        self.mlp.for_each_param(f);
        self.ls2.for_each_param(f);
    }
}

// ---------------------------------------------------------------- cabeca

/// Logits finos residuais sobre os grossos: `z_f[c] = z_g[pai(c)] + r[c]`.
///
/// O CIFAR-100 binario guarda a superclasse no byte 0 de cada registro e a
/// classe fina no byte 1. Sao 20 superclasses com exatamente 5 classes cada.
/// Com esta parametrizacao o modelo so precisa aprender o delta dentro da
/// superclasse, que e onde os erros do conjunto de fato se concentram.
pub struct TaxoHead {
    pub coarse: Linear,
    pub resid: Linear,
    pub parent: Vec<u32>,
    pub nc: usize,
    pub nf: usize,
    n: usize,
}

impl TaxoHead {
    pub fn new(c: usize, parent: Vec<u32>, nc: usize, rng: &mut Rng) -> TaxoHead {
        let nf = parent.len();
        TaxoHead {
            coarse: Linear::new(c, nc, true, 0.02, rng),
            resid: Linear::new(c, nf, true, 0.02, rng),
            parent,
            nc,
            nf,
            n: 0,
        }
    }

    /// Retorna (logits finos [n, nf], logits grossos [n, nc]).
    pub fn forward(&mut self, f: &[f32], n: usize, train: bool) -> (Vec<f32>, Vec<f32>) {
        let zc = self.coarse.forward(f, n, train);
        let mut zf = self.resid.forward(f, n, train);
        let (nf, nc, par) = (self.nf, self.nc, &self.parent);
        zf.par_chunks_mut(nf).zip(zc.par_chunks(nc)).for_each(|(fr, cr)| {
            for i in 0..nf {
                fr[i] += cr[par[i] as usize];
            }
        });
        self.n = n;
        (zf, zc)
    }

    pub fn backward(&mut self, dzf: &[f32], dzc: &[f32]) -> Vec<f32> {
        let (n, nf, nc, par) = (self.n, self.nf, self.nc, &self.parent);
        let mut dc = vec![0.0f32; n * nc];
        dc.par_chunks_mut(nc)
            .zip(dzf.par_chunks(nf))
            .zip(dzc.par_chunks(nc))
            .for_each(|((cr, fr), gr)| {
                cr.copy_from_slice(gr);
                for i in 0..nf {
                    cr[par[i] as usize] += fr[i];
                }
            });
        let mut df = self.resid.backward(dzf);
        let dfc = self.coarse.backward(&dc);
        df.par_iter_mut().zip(dfc.par_iter()).for_each(|(a, b)| *a += b);
        df
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        self.coarse.for_each_param(f);
        self.resid.for_each_param(f);
    }
}

/// Saida auxiliar: classifica e projeta a feature para a dimensao da saida
/// profunda, que e o alvo do termo de hint da autodestilacao.
pub struct Exit {
    pub fc: Linear,
    pub hint: Linear,
    n: usize,
}

impl Exit {
    pub fn new(cin: usize, nclass: usize, chint: usize, rng: &mut Rng) -> Exit {
        Exit {
            fc: Linear::new(cin, nclass, true, 0.02, rng),
            hint: Linear::new(cin, chint, false, 0.02, rng),
            n: 0,
        }
    }

    pub fn forward(&mut self, f: &[f32], n: usize, train: bool) -> (Vec<f32>, Vec<f32>) {
        self.n = n;
        (self.fc.forward(f, n, train), self.hint.forward(f, n, train))
    }

    pub fn backward(&mut self, dz: &[f32], dh: &[f32]) -> Vec<f32> {
        let mut d = self.fc.backward(dz);
        let dhh = self.hint.backward(dh);
        d.par_iter_mut().zip(dhh.par_iter()).for_each(|(a, b)| *a += b);
        d
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        self.fc.for_each_param(f);
        self.hint.for_each_param(f);
    }
}

// ---------------------------------------------------------------- rede

pub struct Cfg {
    pub width: usize,
    pub cmid: usize,
    pub dim: usize,
    pub heads: usize,
    pub depth3: usize,
    pub dim4: usize,
    pub heads4: usize,
    pub depth4: usize,
    pub mem_at: Vec<usize>,
    pub droppath: f32,
    pub img: usize,
}

impl Default for Cfg {
    fn default() -> Cfg {
        Cfg {
            width: 32,
            cmid: 96,
            dim: 256,
            heads: 4,
            depth3: 5,
            dim4: 384,
            heads4: 6,
            depth4: 2,
            mem_at: vec![1, 3],
            droppath: 0.1,
            img: 32,
        }
    }
}

/// Saidas de um forward: as tres cabecas e as tres features agrupadas.
pub struct Out {
    pub logits: Vec<f32>,
    pub coarse: Vec<f32>,
    pub e1: Vec<f32>,
    pub e2: Vec<f32>,
    pub h1: Vec<f32>,
    pub h2: Vec<f32>,
    pub feat: Vec<f32>,
}

pub struct RustViT {
    stem: Conv2d,
    stem_bn: BatchNorm2d,
    stem_out: Arc<Tensor>,
    c1: BasicBlock,
    c2: BasicBlock,
    c2_out: Arc<Tensor>,
    memconv: Conv2d,
    memln: LayerNorm,
    tokconv: Conv2d,
    tok_bn: BatchNorm2d,
    blocks3: Vec<Block>,
    mergeln: LayerNorm,
    merge: Linear,
    blocks4: Vec<Block>,
    headln: LayerNorm,
    pool: SeqPool,
    head: TaxoHead,
    e1: Exit,
    pool2: SeqPool,
    e2: Exit,
    cfg: Cfg,
    th: usize,
    mh: usize,
    mem: Vec<f32>,
    pub name: String,
    pub nclass: usize,
    pub drop_seed: u64,
    pub use_exits: bool,
}

impl RustViT {
    pub fn new(cfg: Cfg, parent: Vec<u32>, ncoarse: usize, seed: u64) -> RustViT {
        let mut rng = Rng::new(seed);
        let (w, cm, d, d4) = (cfg.width, cfg.cmid, cfg.dim, cfg.dim4);
        let mh = cfg.img / 2;
        let th = cfg.img / 4;
        let th4 = th / 2;
        let depth = cfg.depth3 + cfg.depth4;
        let nf = parent.len();

        let mut blocks3 = Vec::new();
        for i in 0..cfg.depth3 {
            let dp = cfg.droppath * i as f32 / (depth.max(2) - 1) as f32;
            blocks3.push(Block::new(
                d,
                cfg.heads,
                th,
                th,
                if cfg.mem_at.contains(&i) {
                    Some((mh, mh))
                } else {
                    None
                },
                depth,
                dp,
                i as u64 + 1,
                1e-4,
                &mut rng,
            ));
        }
        let mut blocks4 = Vec::new();
        for i in 0..cfg.depth4 {
            let dp = cfg.droppath * (cfg.depth3 + i) as f32 / (depth.max(2) - 1) as f32;
            blocks4.push(Block::new(
                d4,
                cfg.heads4,
                th4,
                th4,
                None,
                depth,
                dp,
                (cfg.depth3 + i) as u64 + 1,
                1e-4,
                &mut rng,
            ));
        }

        RustViT {
            stem: Conv2d::new(3, w, 3, 1, 1, &mut rng),
            stem_bn: BatchNorm2d::new(w),
            stem_out: Arc::new(Tensor::zeros(0, 0, 0, 0)),
            c1: BasicBlock::new(w, w, 1, &mut rng),
            c2: BasicBlock::new(w, cm, 2, &mut rng),
            c2_out: Arc::new(Tensor::zeros(0, 0, 0, 0)),
            memconv: Conv2d::new(cm, 2 * d, 1, 1, 0, &mut rng),
            memln: LayerNorm::new(2 * d),
            tokconv: Conv2d::new(cm, d, 3, 2, 1, &mut rng),
            tok_bn: BatchNorm2d::new(d),
            blocks3,
            mergeln: LayerNorm::new(4 * d),
            merge: Linear::new(4 * d, d4, true, 0.02, &mut rng),
            blocks4,
            headln: LayerNorm::new(d4),
            pool: SeqPool::new(d4, &mut rng),
            head: TaxoHead::new(d4, parent, ncoarse, &mut rng),
            e1: Exit::new(cm, nf, d4, &mut rng),
            pool2: SeqPool::new(d, &mut rng),
            e2: Exit::new(d, nf, d4, &mut rng),
            th,
            mh,
            mem: Vec::new(),
            name: format!("rustvit-d{}x{}-{}", cfg.dim, cfg.depth3, cfg.dim4),
            nclass: nf,
            cfg,
            drop_seed: 0,
            use_exits: true,
        }
    }

    pub fn forward(&mut self, x: &Arc<Tensor>, train: bool) -> Out {
        let n = x.n;
        let d = self.cfg.dim;
        // tronco convolucional: codigo do modelo residual, sem mudanca
        let h = Arc::new(self.stem_bn.forward_relu(self.stem.forward(x, train), train));
        if train {
            self.stem_out = h.clone();
        }
        let h = self.c1.forward(&h, train);
        let h = self.c2.forward(&h, train);
        if train {
            self.c2_out = h.clone();
        }

        // memoria de alta resolucao: K e V de 256 tokens, construidos uma vez
        let memt = Tok::from_tensor(&self.memconv.forward(&h, train));
        let mem = self.memln.forward(&memt, train).d;

        // tokens 8x8
        let t = Tok::from_tensor(&self.tok_bn.forward(self.tokconv.forward(&h, train), train));
        let mut z = t;
        let seed = self.drop_seed;
        for b in self.blocks3.iter_mut() {
            z = b.forward(z, Some(&mem), train, seed);
        }
        if train {
            self.mem = mem;
        }

        let (e2l, h2v) = if self.use_exits && train {
            let f = self.pool2.forward(&z, train);
            let (l, hh) = self.e2.forward(&f, n, train);
            (l, hh)
        } else {
            (Vec::new(), Vec::new())
        };

        // merge 2x2 de tokens: quatro vizinhos concatenados
        let z = self.token_merge(&z);
        let zn = self.mergeln.forward(&z, train);
        let th4 = self.th / 2;
        let mut z4 = Tok::from_vec(
            self.merge.forward(&zn.d, zn.rows(), train),
            n,
            th4 * th4,
            self.cfg.dim4,
        );
        for b in self.blocks4.iter_mut() {
            z4 = b.forward(z4, None, train, seed);
        }

        let zf = self.headln.forward(&z4, train);
        let feat = self.pool.forward(&zf, train);
        let (logits, coarse) = self.head.forward(&feat, n, train);

        let (e1l, h1v) = if self.use_exits && train {
            let g = gap_forward(&self.c2_out);
            let (l, hh) = self.e1.forward(&g.d, n, train);
            (l, hh)
        } else {
            (Vec::new(), Vec::new())
        };
        let _ = d;

        Out {
            logits,
            coarse,
            e1: e1l,
            e2: e2l,
            h1: h1v,
            h2: h2v,
            feat,
        }
    }

    /// [n, th*tw, c] -> [n, (th/2)*(tw/2), 4c], vizinhos 2x2 concatenados.
    fn token_merge(&self, z: &Tok) -> Tok {
        let (n, c, th) = (z.n, z.c, self.th);
        let t2 = th / 2;
        let mut y = Tok::zeros(n, t2 * t2, 4 * c);
        y.d.par_chunks_mut(t2 * t2 * 4 * c)
            .zip(z.d.par_chunks(th * th * c))
            .for_each(|(yi, zi)| {
                for oy in 0..t2 {
                    for ox in 0..t2 {
                        let o = (oy * t2 + ox) * 4 * c;
                        for (q, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].iter().enumerate() {
                            let s = ((2 * oy + dy) * th + 2 * ox + dx) * c;
                            yi[o + q * c..o + (q + 1) * c].copy_from_slice(&zi[s..s + c]);
                        }
                    }
                }
            });
        y
    }

    fn token_merge_back(&self, dy: &Tok, c: usize) -> Tok {
        let (n, th) = (dy.n, self.th);
        let t2 = th / 2;
        let mut dz = Tok::zeros(n, th * th, c);
        dz.d.par_chunks_mut(th * th * c)
            .zip(dy.d.par_chunks(t2 * t2 * 4 * c))
            .for_each(|(zi, yi)| {
                for oy in 0..t2 {
                    for ox in 0..t2 {
                        let o = (oy * t2 + ox) * 4 * c;
                        for (q, (dy_, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].iter().enumerate() {
                            let s = ((2 * oy + dy_) * th + 2 * ox + dx) * c;
                            zi[s..s + c].copy_from_slice(&yi[o + q * c..o + (q + 1) * c]);
                        }
                    }
                }
            });
        dz
    }

    /// `d` traz os gradientes das tres saidas e dos dois termos de hint.
    pub fn backward(&mut self, d: &Grads) {
        let n = self.c2_out.n;
        let d3 = self.cfg.dim;
        let df = self.head.backward(&d.logits, &d.coarse);
        let mut dfeat = df;
        if !d.hfinal.is_empty() {
            dfeat
                .par_iter_mut()
                .zip(d.hfinal.par_iter())
                .for_each(|(a, b)| *a += b);
        }
        let dzf = self.pool.backward(&dfeat);
        let mut dz4 = self.headln.backward(&dzf);
        for b in self.blocks4.iter_mut().rev() {
            dz4 = b.backward(dz4, None, None);
        }
        let dmg = Tok::from_vec(self.merge.backward(&dz4.d), n, self.th * self.th / 4, 4 * d3);
        let dzn = self.mergeln.backward(&dmg);
        let mut dz = self.token_merge_back(&dzn, d3);

        if self.use_exits && !d.e2.is_empty() {
            let df2 = self.e2.backward(&d.e2, &d.h2);
            add_(&mut dz, &self.pool2.backward(&df2));
        }

        let mut dmem = vec![0.0f32; self.mem.len()];
        let mem = std::mem::take(&mut self.mem);
        for b in self.blocks3.iter_mut().rev() {
            dz = b.backward(dz, Some(&mem), Some(&mut dmem));
        }
        self.mem = mem;

        let dt = dz.to_tensor(self.th, self.th);
        let dtc = self.tokconv.backward(&self.tok_bn.backward(&dt));

        let mut dmemt = Tok::zeros(n, self.mh * self.mh, 2 * d3);
        dmemt.d.copy_from_slice(&dmem);
        let dmemn = self.memln.backward(&dmemt);
        let dmc = self.memconv.backward(&dmemn.to_tensor(self.mh, self.mh));

        let mut dc2 = dtc;
        dc2.d
            .par_iter_mut()
            .zip(dmc.d.par_iter())
            .for_each(|(a, b)| *a += b);
        if self.use_exits && !d.e1.is_empty() {
            let dg = self.e1.backward(&d.e1, &d.h1);
            let mut gt = Tensor::zeros(n, self.cfg.cmid, 1, 1);
            gt.d.copy_from_slice(&dg);
            let dgap = gap_backward(&gt, self.mh, self.mh);
            dc2.d
                .par_iter_mut()
                .zip(dgap.d.par_iter())
                .for_each(|(a, b)| *a += b);
        }

        let dh = self.c2.backward(dc2);
        let mut dh = self.c1.backward(dh);
        crate::nn::relu_back_(&mut dh, &self.stem_out);
        let dh = self.stem_bn.backward(&dh);
        self.stem.backward_w(&dh);
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.stem.w);
        f(&mut self.stem_bn.g);
        f(&mut self.stem_bn.b);
        self.c1.for_each_param(f);
        self.c2.for_each_param(f);
        f(&mut self.memconv.w);
        self.memln.for_each_param(f);
        f(&mut self.tokconv.w);
        f(&mut self.tok_bn.g);
        f(&mut self.tok_bn.b);
        for b in self.blocks3.iter_mut() {
            b.for_each_param(f);
        }
        self.mergeln.for_each_param(f);
        self.merge.for_each_param(f);
        for b in self.blocks4.iter_mut() {
            b.for_each_param(f);
        }
        self.headln.for_each_param(f);
        self.pool.for_each_param(f);
        self.head.for_each_param(f);
        if self.use_exits {
            self.e1.for_each_param(f);
            self.pool2.for_each_param(f);
            self.e2.for_each_param(f);
        }
    }

    fn for_each_buffer(&mut self, f: &mut dyn FnMut(&mut Vec<f32>)) {
        f(&mut self.stem_bn.rm);
        f(&mut self.stem_bn.rv);
        self.c1.buffers(f);
        self.c2.buffers(f);
        f(&mut self.tok_bn.rm);
        f(&mut self.tok_bn.rv);
    }

    pub fn num_params(&mut self) -> usize {
        let mut n = 0;
        self.for_each_param(&mut |p| n += p.v.len());
        n
    }

    /// MACs de forward por imagem.
    pub fn macs(&self) -> u64 {
        let (w, cm, d, d4) = (self.cfg.width, self.cfg.cmid, self.cfg.dim, self.cfg.dim4);
        let img = self.cfg.img;
        let (r1, r2, t, t4) = (img * img, img * img / 4, self.th * self.th, self.th * self.th / 4);
        let mut m = (3 * 9 * w * r1) as u64; // stem
        m += 2 * (w * 9 * w * r1) as u64; // C1
        m += (w * 9 * cm * r2 + cm * 9 * cm * r2 + w * cm * r2) as u64; // C2
        m += (cm * 2 * d * r2) as u64; // memoria
        m += (cm * 9 * d * t) as u64; // tokens
        let hid3 = (4 * d / 3).div_ceil(16) * 16;
        let blk3 = (t * d * 3 * d + 2 * t * t * d + t * d * d + 3 * t * d * hid3 + 9 * d * t) as u64;
        m += blk3 * self.cfg.depth3 as u64;
        m += (2 * t * r2 * d * self.cfg.mem_at.len()) as u64; // atencao cruzada
        m += (t4 * 4 * d * d4) as u64; // merge
        let hid4 = (4 * d4 / 3).div_ceil(16) * 16;
        let blk4 =
            (t4 * d4 * 3 * d4 + 2 * t4 * t4 * d4 + t4 * d4 * d4 + 3 * t4 * d4 * hid4 + 9 * d4 * t4)
                as u64;
        m += blk4 * self.cfg.depth4 as u64;
        m += (d4 * (self.nclass + self.head.nc)) as u64;
        m
    }

    pub fn save(&mut self, path: &str) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut err = Ok(());
        let mut w = |v: &Vec<f32>, err: &mut std::io::Result<()>| {
            if err.is_ok() {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                *err = f.write_all(&bytes);
            }
        };
        let mut params: Vec<Vec<f32>> = Vec::new();
        self.for_each_param(&mut |p| params.push(p.v.clone()));
        let mut bufs: Vec<Vec<f32>> = Vec::new();
        self.for_each_buffer(&mut |b| bufs.push(b.clone()));
        for v in params.iter().chain(bufs.iter()) {
            w(v, &mut err);
        }
        err?;
        f.flush()
    }

    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        let mut buf = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut buf)?;
        let mut off = 0usize;
        let mut rd = |v: &mut Vec<f32>| {
            for x in v.iter_mut() {
                *x = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                off += 4;
            }
        };
        self.for_each_param(&mut |p| rd(&mut p.v));
        self.for_each_buffer(&mut |b| rd(b));
        Ok(())
    }
}

/// Gradientes que a perda multi-saida devolve ao modelo.
#[derive(Default)]
pub struct Grads {
    pub logits: Vec<f32>,
    pub coarse: Vec<f32>,
    pub e1: Vec<f32>,
    pub e2: Vec<f32>,
    pub h1: Vec<f32>,
    pub h2: Vec<f32>,
    pub hfinal: Vec<f32>,
}
