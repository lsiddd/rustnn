//! Formato de checkpoint compartilhado pelas duas arquiteturas.

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
    pub fn new(tensors: usize, scalars: usize, config_hash: u64) -> io::Result<Self> {
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

pub fn save(
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

pub fn load(
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
