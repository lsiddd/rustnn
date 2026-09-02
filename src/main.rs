//! rustnn: redes convolucionais e um transformer hibrido para CIFAR-100, do
//! zero em Rust, sem BLAS nem biblioteca de ML.
//!
//! `main.rs` so despacha. A linha de comando esta em `cli.rs`, o treino em
//! `train.rs`, a avaliacao em `eval.rs` e a cronometragem em `bench.rs`. A rede
//! mora em `vit.rs`, o GEMM em `gemm.rs` e as verificacoes de gradiente em
//! `check.rs`.

// Este e um codigo de kernel numerico: laco por indice e a linguagem do dominio
// (`for j in 0..n { d[j] = ... }` le como a formula) e as assinaturas largas sao
// os parametros de um GEMM, nao acidente de projeto.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

mod attn;
mod bench;
mod ckpt;
mod cli;
mod conv;
mod data;
mod eval;
mod gemm;
mod linear;
mod loss;
mod model;
mod nn;
mod norm;
mod optim;
mod pool;
mod rng;
mod tok;
mod train;
mod vit;

/// Reciclar os blocos grandes vale mais que qualquer outra otimizacao de memoria
/// deste modelo: ver `pool.rs`.
#[global_allocator]
static ALLOC: pool::Recycling = pool::Recycling;

use model::ResNet;
use nn::{softmax_ce, Param, Tensor};
use rng::{rand_vec, Rng};
use std::sync::Arc;
use vit::{Cfg, Grads, Out, RustViT};

include!("check.rs");

fn set_threads(n: usize) {
    if n > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .unwrap();
    }
}

fn main() {
    let cfg = cli::parse();
    if cfg.gemmbench {
        set_threads(cfg.threads);
        return bench::gemm();
    }
    if cfg.gradcheck {
        let ok = layercheck(cfg.seed) & vitcheck(cfg.seed) & gradcheck(cfg.seed);
        std::process::exit(if ok { 0 } else { 1 });
    }
    // O fingerprint so e reprodutivel numa thread: a ordem da arvore de reducao
    // do rayon depende de work stealing, e soma de f32 nao e associativa.
    set_threads(if cfg.fingerprint > 0 { 1 } else { cfg.threads });

    let (train, test) = match data::load(&cfg.data) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "erro lendo o dataset em '{}': {e}\n\
                 baixe com: curl -LO https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz && tar xzf cifar-100-binary.tar.gz",
                cfg.data
            );
            std::process::exit(1);
        }
    };
    let taxo = data::taxonomy(&train);
    let ntrain = if cfg.subset > 0 {
        cfg.subset.min(train.len)
    } else {
        train.len
    };

    if cfg.fingerprint > 0 {
        return train::fingerprint(&cfg, &train, &taxo, ntrain);
    }
    if cfg.arch == "resnet" {
        return train::run_resnet(&cfg, &train, &test, ntrain);
    }
    train::run_vit(&cfg, &train, &test, &taxo, ntrain);
}
