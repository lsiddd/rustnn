//! Alocador global que recicla os blocos grandes.
//!
//! Cada camada aloca a propria saida, e com lote 256 sao dezenas de blocos de 20
//! a 50 MB por passo. Acima de 32 MB o alocador do sistema serve tudo por `mmap`
//! e devolve por `munmap`, entao **cada** buffer novo custa uma falta de pagina
//! por pagina no primeiro toque: 12 800 faltas num bloco de 50 MB. Medindo, essa
//! era a maior fatia do tempo que nao e nem GEMM nem convolucao.
//!
//! O modelo repete exatamente as mesmas formas em todo passo, entao um cache por
//! tamanho e alinhamento exatos converge para o conjunto de buffers vivos ao
//! mesmo tempo e nunca mais toca no sistema. Nada aqui muda um unico bit do
//! resultado.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const MIN_CACHE: usize = 1 << 20;
const SLOTS: usize = 96;
/// Teto do cache. Estourou, o bloco volta para o sistema.
const MAX_BYTES: usize = 6 << 30;

#[derive(Clone, Copy)]
struct Slot {
    ptr: usize,
    size: usize,
    align: usize,
}

const EMPTY: Slot = Slot {
    ptr: 0,
    size: 0,
    align: 0,
};

/// Array de tamanho fixo, e nao `Vec`: um `push` dentro do lock reentraria no
/// alocador segurando o proprio mutex.
static CACHE: Mutex<[Slot; SLOTS]> = Mutex::new([EMPTY; SLOTS]);
static HELD: AtomicUsize = AtomicUsize::new(0);
/// 0 = nao decidido, 1 = ligado, 2 = desligado por RUSTNN_NOPOOL.
static ON: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn enabled() -> bool {
    match ON.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let v = if std::env::var_os("RUSTNN_NOPOOL").is_some() {
                2
            } else {
                1
            };
            ON.store(v, Ordering::Relaxed);
            v == 1
        }
    }
}

pub struct Recycling;

unsafe impl GlobalAlloc for Recycling {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.size() >= MIN_CACHE && enabled() {
            if let Ok(mut c) = CACHE.try_lock() {
                for s in c.iter_mut() {
                    if s.ptr != 0 && s.size == l.size() && s.align == l.align() {
                        let p = s.ptr;
                        *s = EMPTY;
                        HELD.fetch_sub(l.size(), Ordering::Relaxed);
                        return p as *mut u8;
                    }
                }
            }
        }
        System.alloc(l)
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if l.size() >= MIN_CACHE && enabled() && HELD.load(Ordering::Relaxed) + l.size() <= MAX_BYTES {
            if let Ok(mut c) = CACHE.try_lock() {
                for s in c.iter_mut() {
                    if s.ptr == 0 {
                        *s = Slot {
                            ptr: p as usize,
                            size: l.size(),
                            align: l.align(),
                        };
                        HELD.fetch_add(l.size(), Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
        System.dealloc(p, l)
    }
}
