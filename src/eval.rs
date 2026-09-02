//! Avaliacao. As duas arquiteturas compartilham o laco de batching; o que muda
//! entre elas e so como um lote vira logits.

use crate::data::{self, IMG, NCLASS};
use crate::model::ResNet;
use crate::nn::{softmax_ce, Tensor};
use crate::vit::RustViT;
use std::sync::Arc;

/// Percorre o conjunto em lotes e acumula perda e acertos.
///
/// `logits` recebe o tensor ja preenchido com o lote cru, o offset no conjunto
/// e o tamanho do lote. Recebe `x` e `labels` por referencia mutavel porque o
/// TTA precisa reescrever o lote com a versao espelhada.
fn eval_loop(
    ds: &data::Dataset,
    batch: usize,
    mut logits: impl FnMut(&mut Arc<Tensor>, &mut [u32], usize, usize) -> Vec<f32>,
) -> (f32, f32) {
    let mut x = Arc::new(Tensor::zeros(batch, 3, IMG, IMG));
    let mut labels = vec![0u32; batch];
    let mut dl = vec![0.0f32; batch * NCLASS];
    let (mut sum, mut correct, mut seen, mut s) = (0.0f32, 0usize, 0usize, 0usize);
    while s < ds.len {
        let bs = batch.min(ds.len - s);
        {
            let xm = Arc::make_mut(&mut x);
            xm.resize(bs, 3, IMG, IMG);
            data::make_batch_eval(ds, s, bs, false, &mut xm.d, &mut labels);
        }
        let lg = logits(&mut x, &mut labels, s, bs);
        let (l, c) = softmax_ce(&lg, &labels[..bs], NCLASS, 0.0, &mut dl[..bs * NCLASS]);
        sum += l * bs as f32;
        correct += c;
        seen += bs;
        s += bs;
    }
    (sum / seen as f32, correct as f32 / seen as f32)
}

pub fn resnet(net: &mut ResNet, ds: &data::Dataset, batch: usize) -> (f32, f32) {
    eval_loop(ds, batch, |x, _, _, _| net.forward(x, false))
}

/// As saidas auxiliares so existem no treino: avaliar com elas ligadas mediria
/// um modelo que nao e o que se usa.
pub fn vit(net: &mut RustViT, ds: &data::Dataset, batch: usize, tta: bool) -> (f32, f32) {
    let exits = net.use_exits;
    net.use_exits = false;
    let r = eval_loop(ds, batch, |x, labels, s, bs| {
        let mut lg = net.forward(x, false).logits;
        if tta {
            {
                let xm = Arc::make_mut(x);
                data::make_batch_eval(ds, s, bs, true, &mut xm.d, labels);
            }
            let mirrored = net.forward(x, false).logits;
            lg.iter_mut()
                .zip(&mirrored)
                .for_each(|(a, b)| *a = 0.5 * (*a + b));
        }
        lg
    });
    net.use_exits = exits;
    r
}
