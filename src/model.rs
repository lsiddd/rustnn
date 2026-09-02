//! ResNet estilo CIFAR: stem 3x3 + 3 estagios de blocos residuais + GAP + FC.

use crate::nn::*;
use crate::rng::Rng;
use std::io::{Read, Write};
use std::sync::Arc;

pub struct BasicBlock {
    pub c1: Conv2d,
    b1: BatchNorm2d,
    pub c2: Conv2d,
    b2: BatchNorm2d,
    pub down: Option<(Conv2d, BatchNorm2d)>,
    h1: Arc<Tensor>,  // saida da 1a ReLU
    out: Arc<Tensor>, // saida da 2a ReLU
}

impl BasicBlock {
    pub fn new(cin: usize, cout: usize, stride: usize, rng: &mut Rng) -> Self {
        let mut b2 = BatchNorm2d::new(cout);
        b2.zero_gamma(); // o bloco comeca como identidade
        BasicBlock {
            c1: Conv2d::new(cin, cout, 3, stride, 1, rng),
            b1: BatchNorm2d::new(cout),
            c2: Conv2d::new(cout, cout, 3, 1, 1, rng),
            b2,
            down: if stride != 1 || cin != cout {
                Some((Conv2d::new(cin, cout, 1, stride, 0, rng), BatchNorm2d::new(cout)))
            } else {
                None
            },
            h1: Arc::new(Tensor::zeros(0, 0, 0, 0)),
            out: Arc::new(Tensor::zeros(0, 0, 0, 0)),
        }
    }

    pub fn forward(&mut self, x: &Arc<Tensor>, train: bool) -> Arc<Tensor> {
        let h = Arc::new(self.b1.forward_relu(self.c1.forward(x, train), train));
        let mut y = self.b2.forward(self.c2.forward(&h, train), train);
        match &mut self.down {
            Some((c, b)) => {
                let s = b.forward(c.forward(x, train), train);
                add_relu_(&mut y, &s);
            }
            None => add_relu_(&mut y, x),
        }
        let y = Arc::new(y);
        if train {
            self.h1 = h;
            self.out = y.clone();
        }
        y
    }

    pub fn backward(&mut self, mut d: Tensor) -> Tensor {
        relu_back_(&mut d, &self.out);

        let mut dmain = self.c2.backward(&self.b2.backward(&d));
        relu_back_(&mut dmain, &self.h1);
        let mut dx = self.c1.backward(&self.b1.backward(&dmain));

        match &mut self.down {
            Some((c, b)) => {
                let ds = c.backward(&b.backward(&d));
                add_(&mut dx, &ds);
            }
            None => add_(&mut dx, &d),
        }
        dx
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.c1.w);
        f(&mut self.b1.g);
        f(&mut self.b1.b);
        f(&mut self.c2.w);
        f(&mut self.b2.g);
        f(&mut self.b2.b);
        if let Some((c, b)) = &mut self.down {
            f(&mut c.w);
            f(&mut b.g);
            f(&mut b.b);
        }
    }

    pub fn buffers(&mut self, f: &mut dyn FnMut(&mut Vec<f32>)) {
        f(&mut self.b1.rm);
        f(&mut self.b1.rv);
        f(&mut self.b2.rm);
        f(&mut self.b2.rv);
        if let Some((_, b)) = &mut self.down {
            f(&mut b.rm);
            f(&mut b.rv);
        }
    }
}

pub struct ResNet {
    pub stem: Conv2d,
    stem_bn: BatchNorm2d,
    stem_out: Arc<Tensor>,
    pub blocks: Vec<BasicBlock>,
    pub fc: Linear,
    feat_hw: (usize, usize),
    pub name: String,
}

impl ResNet {
    /// `depth_n` blocos por estagio (profundidade total = 6n+2), largura base `width`.
    pub fn new(depth_n: usize, width: usize, nclass: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let widths = [width, width * 2, width * 4];
        let mut blocks = Vec::new();
        let mut cin = width;
        for (s, &w) in widths.iter().enumerate() {
            for b in 0..depth_n {
                let stride = if s > 0 && b == 0 { 2 } else { 1 };
                blocks.push(BasicBlock::new(cin, w, stride, &mut rng));
                cin = w;
            }
        }
        ResNet {
            stem: Conv2d::new(3, width, 3, 1, 1, &mut rng),
            stem_bn: BatchNorm2d::new(width),
            stem_out: Arc::new(Tensor::zeros(0, 0, 0, 0)),
            blocks,
            fc: Linear::new(width * 4, nclass, &mut rng),
            feat_hw: (8, 8),
            name: format!("resnet{}-w{}", 6 * depth_n + 2, width),
        }
    }

    pub fn forward(&mut self, x: &Arc<Tensor>, train: bool) -> Vec<f32> {
        let mut h = Arc::new(self.stem_bn.forward_relu(self.stem.forward(x, train), train));
        if train {
            self.stem_out = h.clone();
        }
        for b in self.blocks.iter_mut() {
            h = b.forward(&h, train);
        }
        self.feat_hw = (h.h, h.w);
        let pooled = gap_forward(&h);
        self.fc.forward(&pooled, train)
    }

    pub fn backward(&mut self, dlogits: &[f32]) {
        let dpool = self.fc.backward(dlogits);
        let mut d = gap_backward(&dpool, self.feat_hw.0, self.feat_hw.1);
        for b in self.blocks.iter_mut().rev() {
            d = b.backward(d);
        }
        relu_back_(&mut d, &self.stem_out);
        let d = self.stem_bn.backward(&d);
        self.stem.backward_w(&d); // dX aqui seria o gradiente da imagem: descartado
    }

    pub fn for_each_param(&mut self, f: &mut dyn FnMut(&mut Param)) {
        f(&mut self.stem.w);
        f(&mut self.stem_bn.g);
        f(&mut self.stem_bn.b);
        for b in self.blocks.iter_mut() {
            b.for_each_param(f);
        }
        f(&mut self.fc.w);
        f(&mut self.fc.b);
    }

    fn for_each_buffer(&mut self, f: &mut dyn FnMut(&mut Vec<f32>)) {
        f(&mut self.stem_bn.rm);
        f(&mut self.stem_bn.rv);
        for b in self.blocks.iter_mut() {
            b.buffers(f);
        }
    }

    /// MACs de forward por imagem (2 flops cada); treino ~ 3x isso.
    pub fn flops(&self, h: usize, w: usize) -> u64 {
        let mut f = 0u64;
        let mut hw = (h, w);
        let cin0 = 3;
        let mut c = self.stem.cout;
        f += 2 * (cin0 * 9 * c * hw.0 * hw.1) as u64;
        for b in &self.blocks {
            let s = b.c1.stride;
            let (oh, ow) = (hw.0 / s, hw.1 / s);
            f += 2 * (b.c1.cin * 9 * b.c1.cout * oh * ow) as u64;
            f += 2 * (b.c2.cin * 9 * b.c2.cout * oh * ow) as u64;
            if let Some((cd, _)) = &b.down {
                f += 2 * (cd.cin * cd.cout * oh * ow) as u64;
            }
            hw = (oh, ow);
            c = b.c2.cout;
        }
        f += 2 * (c * self.fc.fout) as u64;
        f
    }

    pub fn num_params(&mut self) -> usize {
        let mut n = 0;
        self.for_each_param(&mut |p| n += p.v.len());
        n
    }

    pub fn step(&mut self, lr: f32, mom: f32, wd: f32) {
        self.for_each_param(&mut |p| p.step(lr, mom, wd));
    }

    pub fn save(&mut self, path: &str) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut w = |v: &Vec<f32>| -> std::io::Result<()> {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&bytes)
        };
        let mut err = Ok(());
        self.for_each_param(&mut |p| {
            if err.is_ok() {
                err = w(&p.v);
            }
        });
        self.for_each_buffer(&mut |b| {
            if err.is_ok() {
                err = w(b);
            }
        });
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
