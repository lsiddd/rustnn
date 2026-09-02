//! Formato de checkpoint compartilhado pelas duas arquiteturas.

use crate::nn::Param;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 7] = b"RUSTNN\0";
const HEADER_LEN: usize = MAGIC.len() + 4 + 8 + 8;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
pub struct Layout {
    tensors: u32,
    scalars: u64,
    config_hash: u64,
}

impl Layout {
    fn new(tensors: usize, scalars: usize, config_hash: u64) -> io::Result<Self> {
        Ok(Self {
            tensors: u32::try_from(tensors)
                .map_err(|_| invalid("checkpoint tem mais tensores que o formato suporta"))?,
            scalars: u64::try_from(scalars)
                .map_err(|_| invalid("checkpoint tem mais escalares que o formato suporta"))?,
            config_hash,
        })
    }

    fn payload_bytes(self) -> io::Result<usize> {
        usize::try_from(self.scalars)
            .ok()
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or_else(|| invalid("tamanho do checkpoint excede a plataforma"))
    }
}

/// Um modelo que sabe percorrer seus tensores sabe se salvar.
///
/// As duas arquiteturas diferem apenas no que entra no hash da configuracao.
/// Contar tensores, escrever e ler e o mesmo codigo nas duas, e estava
/// duplicado linha a linha entre `vit.rs` e `model.rs`.
pub trait Checkpointed {
    fn visit_params(&mut self, f: &mut dyn FnMut(&mut Param));
    fn visit_buffers(&mut self, f: &mut dyn FnMut(&mut Vec<f32>));

    /// FNV-1a dos campos que mudam a forma ou o significado dos pesos. E o que
    /// faz um checkpoint de outra configuracao ser recusado em vez de carregado
    /// torto.
    fn config_hash(&mut self) -> u64;

    fn layout(&mut self) -> io::Result<Layout> {
        let config = self.config_hash();
        let (mut tensors, mut scalars) = (0usize, 0usize);
        self.visit_params(&mut |p| {
            tensors += 1;
            scalars += p.v.len();
        });
        self.visit_buffers(&mut |buffer| {
            tensors += 1;
            scalars += buffer.len();
        });
        Layout::new(tensors, scalars, config)
    }

    fn save(&mut self, path: &str) -> io::Result<()> {
        let layout = self.layout()?;
        write_checkpoint(path, layout, |write| {
            self.visit_params(&mut |p| write(&p.v));
            self.visit_buffers(&mut |buffer| write(buffer));
        })
    }

    fn load(&mut self, path: &str) -> io::Result<()> {
        let layout = self.layout()?;
        read_checkpoint(path, layout, |read| {
            self.visit_params(&mut |p| read(&mut p.v));
            self.visit_buffers(&mut |buffer| read(buffer));
        })
    }
}

/// FNV-1a dos campos que mudam o significado ou a forma dos pesos.
pub fn config_hash(kind: &[u8], values: &[u64]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in kind {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn write_checkpoint(
    path: &str,
    layout: Layout,
    mut visit: impl FnMut(&mut dyn FnMut(&[f32])),
) -> io::Result<()> {
    let temporary = format!("{path}.tmp");
    let result = (|| {
        let mut file = BufWriter::new(File::create(&temporary)?);
        file.write_all(MAGIC)?;
        file.write_all(&layout.tensors.to_le_bytes())?;
        file.write_all(&layout.scalars.to_le_bytes())?;
        file.write_all(&layout.config_hash.to_le_bytes())?;

        let mut result = Ok(());
        visit(&mut |values| {
            if result.is_ok() {
                result = write_f32s(&mut file, values);
            }
        });
        result?;
        file.flush()?;
        file.get_ref().sync_all()
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    fs::rename(temporary, path)
}

fn read_checkpoint(
    path: &str,
    expected: Layout,
    mut visit: impl FnMut(&mut dyn FnMut(&mut [f32])),
) -> io::Result<()> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let expected_bytes = expected.payload_bytes()?;

    let payload = if bytes.starts_with(MAGIC) {
        let tensors = read_u32(&bytes, MAGIC.len(), "contagem de tensores")?;
        let scalars = read_u64(&bytes, MAGIC.len() + 4, "contagem de escalares")?;
        let config_hash = read_u64(&bytes, MAGIC.len() + 12, "hash da configuracao")?;
        if tensors != expected.tensors {
            return Err(invalid(format!(
                "checkpoint tem {tensors} tensores; modelo espera {}",
                expected.tensors
            )));
        }
        if scalars != expected.scalars {
            return Err(invalid(format!(
                "checkpoint tem {scalars} escalares; modelo espera {}",
                expected.scalars
            )));
        }
        if config_hash != expected.config_hash {
            return Err(invalid(format!(
                "checkpoint incompativel com a configuracao (hash {config_hash:016x}, esperado {:016x})",
                expected.config_hash
            )));
        }
        let actual = bytes
            .len()
            .checked_sub(HEADER_LEN)
            .ok_or_else(|| invalid("checkpoint truncado antes do fim do cabecalho"))?;
        if actual != expected_bytes {
            return Err(invalid(format!(
                "payload tem {actual} bytes; esperado {expected_bytes}"
            )));
        }
        &bytes[HEADER_LEN..]
    } else if bytes.len() == expected_bytes {
        eprintln!(
            "aviso: checkpoint legado sem cabecalho: {}",
            Path::new(path).display()
        );
        &bytes[..]
    } else {
        return Err(invalid(format!(
            "magic ausente e tamanho legado invalido: {} bytes; esperado {expected_bytes}",
            bytes.len()
        )));
    };

    let mut offset = 0usize;
    let mut result = Ok(());
    visit(&mut |values| {
        if result.is_ok() {
            result = read_f32s(payload, &mut offset, values);
        }
    });
    result?;
    if offset != payload.len() {
        return Err(invalid(format!(
            "checkpoint contem {} bytes nao consumidos",
            payload.len() - offset
        )));
    }
    Ok(())
}

fn write_f32s(writer: &mut impl Write, values: &[f32]) -> io::Result<()> {
    const FLOATS_PER_CHUNK: usize = 1024;
    let mut bytes = [0u8; FLOATS_PER_CHUNK * size_of::<f32>()];
    for chunk in values.chunks(FLOATS_PER_CHUNK) {
        for (dst, value) in bytes.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..chunk.len() * 4])?;
    }
    Ok(())
}

fn read_f32s(payload: &[u8], offset: &mut usize, values: &mut [f32]) -> io::Result<()> {
    let byte_len = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| invalid("tensor grande demais"))?;
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| invalid("offset do checkpoint excede a plataforma"))?;
    let bytes = payload
        .get(*offset..end)
        .ok_or_else(|| invalid("checkpoint truncado no meio de um tensor"))?;
    for (value, raw) in values.iter_mut().zip(bytes.chunks_exact(4)) {
        *value = f32::from_le_bytes(raw.try_into().expect("chunk de quatro bytes"));
    }
    *offset = end;
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize, field: &str) -> io::Result<u32> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| invalid(format!("checkpoint truncado em {field}")))?;
    Ok(u32::from_le_bytes(raw.try_into().expect("quatro bytes")))
}

fn read_u64(bytes: &[u8], offset: usize, field: &str) -> io::Result<u64> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| invalid(format!("checkpoint truncado em {field}")))?;
    Ok(u64::from_le_bytes(raw.try_into().expect("oito bytes")))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::{Checkpointed, MAGIC};
    use crate::nn::Param;
    use std::path::PathBuf;

    /// Modelo minimo: dois parametros e um buffer, com um hash de configuracao
    /// que o teste controla.
    struct Fake {
        params: Vec<Param>,
        buffer: Vec<f32>,
        config: u64,
    }

    impl Fake {
        fn new(config: u64, fill: f32) -> Self {
            let mut params = vec![Param::new(6, true), Param::new(4, false)];
            for (i, p) in params.iter_mut().enumerate() {
                for (j, v) in p.v.iter_mut().enumerate() {
                    *v = fill + i as f32 * 10.0 + j as f32;
                }
            }
            Fake {
                params,
                buffer: vec![fill * 2.0; 3],
                config,
            }
        }
    }

    impl Checkpointed for Fake {
        fn visit_params(&mut self, f: &mut dyn FnMut(&mut Param)) {
            for p in &mut self.params {
                f(p);
            }
        }
        fn visit_buffers(&mut self, f: &mut dyn FnMut(&mut Vec<f32>)) {
            f(&mut self.buffer);
        }
        fn config_hash(&mut self) -> u64 {
            self.config
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("rustnn-ckpt-{name}-{}.bin", std::process::id()));
        p
    }

    #[test]
    fn round_trip_preserva_tudo() {
        let path = scratch("roundtrip");
        let mut saved = Fake::new(0xABCD, 1.5);
        saved.save(path.to_str().unwrap()).unwrap();

        let mut loaded = Fake::new(0xABCD, 0.0);
        loaded.load(path.to_str().unwrap()).unwrap();
        for (a, b) in saved.params.iter().zip(&loaded.params) {
            assert_eq!(a.v, b.v);
        }
        assert_eq!(saved.buffer, loaded.buffer);

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(MAGIC), "o arquivo tem de trazer o magic");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recusa_configuracao_diferente() {
        let path = scratch("config");
        Fake::new(0x1111, 1.0).save(path.to_str().unwrap()).unwrap();
        let err = Fake::new(0x2222, 0.0).load(path.to_str().unwrap()).unwrap_err();
        assert!(
            err.to_string().contains("incompativel"),
            "mensagem inesperada: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recusa_arquivo_truncado() {
        let path = scratch("trunc");
        Fake::new(7, 1.0).save(path.to_str().unwrap()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();
        assert!(Fake::new(7, 0.0).load(path.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    /// Os checkpoints gravados antes do cabecalho existirem sao so o payload.
    #[test]
    fn aceita_legado_sem_cabecalho() {
        let path = scratch("legado");
        let mut original = Fake::new(42, 3.0);
        original.save(path.to_str().unwrap()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[super::HEADER_LEN..]).unwrap();

        let mut loaded = Fake::new(42, 0.0);
        loaded.load(path.to_str().unwrap()).unwrap();
        assert_eq!(original.params[0].v, loaded.params[0].v);
        assert_eq!(original.buffer, loaded.buffer);
        let _ = std::fs::remove_file(&path);
    }

    /// Um save que falha nao pode deixar o `.tmp` para tras nem tocar no
    /// arquivo bom que ja existia.
    #[test]
    fn save_nao_destroi_o_anterior_ao_falhar() {
        let path = scratch("atomico");
        Fake::new(1, 5.0).save(path.to_str().unwrap()).unwrap();
        let bom = std::fs::read(&path).unwrap();

        let ruim = path.join("nao-e-diretorio");
        assert!(Fake::new(1, 9.0).save(ruim.to_str().unwrap()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bom);
        assert!(!PathBuf::from(format!("{}.tmp", ruim.display())).exists());
        let _ = std::fs::remove_file(&path);
    }
}
