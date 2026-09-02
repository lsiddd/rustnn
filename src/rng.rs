//! xoshiro256++ minimalista.

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut sm = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            s: [sm(), sm(), sm(), sm()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Inteiro uniforme em [0, n).
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Normal padrao (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// Vetor de normais padrao. Usado pelas checagens de gradiente e pelo
/// `--gemmbench`, que precisam de entradas aleatorias mas reprodutiveis.
pub fn rand_vec(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| rng.normal()).collect()
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn mesma_semente_mesma_sequencia() {
        let (mut a, mut b) = (Rng::new(1234), Rng::new(1234));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn uniform_no_intervalo() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let x = rng.uniform();
            assert!((0.0..1.0).contains(&x), "uniform devolveu {x}");
        }
    }

    #[test]
    fn below_no_intervalo() {
        let mut rng = Rng::new(9);
        for n in [1usize, 2, 7, 100] {
            for _ in 0..1000 {
                assert!(rng.below(n) < n);
            }
        }
    }

    /// O embaralhamento e uma permutacao: nenhum indice pode sumir nem
    /// duplicar, porque `order` indexa o dataset.
    #[test]
    fn shuffle_e_permutacao() {
        let mut rng = Rng::new(3);
        let mut v: Vec<usize> = (0..500).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..500).collect::<Vec<_>>());
        assert_ne!(v, sorted, "500 elementos sairem em ordem seria suspeito");
    }
}
