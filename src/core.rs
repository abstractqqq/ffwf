use arrow_array::builder::{GenericByteViewBuilder, PrimitiveBuilder};
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, StringViewType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::sync::Arc;

pub static WHITESPACE_LUT: [u8; 256] = {
    let mut table = [0u8; 256];
    table[b' ' as usize] = 1;
    table[b'\t' as usize] = 1;
    table[b'\n' as usize] = 1;
    table[b'\r' as usize] = 1;
    table
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    String,
}

impl DType {
    pub fn to_arrow(&self) -> DataType {
        match self {
            DType::I8 => DataType::Int8,
            DType::I16 => DataType::Int16,
            DType::I32 => DataType::Int32,
            DType::I64 => DataType::Int64,
            DType::U8 => DataType::UInt8,
            DType::U16 => DataType::UInt16,
            DType::U32 => DataType::UInt32,
            DType::U64 => DataType::UInt64,
            DType::F32 => DataType::Float32,
            DType::F64 => DataType::Float64,
            DType::String => DataType::Utf8View,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub offset: usize,
    pub length: usize,
    pub dtype: DType,
    pub padding: Option<u8>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum ErrorStrategy {
    PushNull,
}

#[derive(Debug, Clone, Copy)]
pub enum Par {
    Seq,
    Fixed(usize),
}

pub struct FwfParser {
    pub specs: Vec<FieldSpec>,
    pub line_length: usize,
    pub schema: Arc<Schema>,
    pub chunk_size: usize,
    pub parallelism: Par,
    pub error_strategy: ErrorStrategy,
}

impl FwfParser {
    pub fn new(specs: Vec<FieldSpec>, line_length: usize) -> Self {
        let fields: Vec<Field> = specs
            .iter()
            .map(|s| Field::new(&s.name, s.dtype.to_arrow(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));

        let chunk_size = Self::infer_chunk_size(&specs);

        Self {
            specs,
            line_length,
            schema,
            chunk_size,
            parallelism: Par::Fixed(0),
            error_strategy: ErrorStrategy::PushNull,
        }
    }

    pub fn infer_chunk_size(specs: &[FieldSpec]) -> usize {
        let mut est_bytes_per_row = 0;
        for s in specs {
            match s.dtype {
                DType::String => est_bytes_per_row += 16 + s.length,
                _ => est_bytes_per_row += 8,
            }
        }

        let target_batch_size_bytes = 32 * 1024 * 1024; // 32MB target
        let inferred = target_batch_size_bytes / est_bytes_per_row.max(1);
        inferred.clamp(1024, 65536)
    }

    pub fn detect_line_length(path: &str, newline: &[u8]) -> std::io::Result<(usize, usize)> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Use a simple window search for the newline symbol
        let pos = mmap
            .windows(newline.len())
            .position(|window| window == newline)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Could not detect newline symbol in FWF file",
                )
            })?;

        let stride = pos + newline.len();
        let data_len = pos;
        Ok((stride, data_len))
    }

    pub fn parse_path(&self, path: &str) -> std::io::Result<Vec<RecordBatch>> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(self.parse(&mmap))
    }

    pub fn parse(&self, data: &[u8]) -> Vec<RecordBatch> {
        let total_rows = data.len() / self.line_length;
        if total_rows == 0 {
            return vec![];
        }

        let chunk_size_bytes = self.chunk_size * self.line_length;

        match self.parallelism {
            Par::Seq => data
                .chunks(chunk_size_bytes)
                .map(|chunk| self.parse_batch(chunk))
                .collect(),
            Par::Fixed(n) => {
                let num_threads = if n == 0 {
                    rayon::current_num_threads()
                } else {
                    n.min(rayon::current_num_threads())
                };

                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap();

                pool.install(|| {
                    data.par_chunks(chunk_size_bytes)
                        .map(|chunk| self.parse_batch(chunk))
                        .collect()
                })
            }
        }
    }

    pub fn parse_batch(&self, data: &[u8]) -> RecordBatch {
        let num_rows = data.len() / self.line_length;
        let mut builders = self
            .specs
            .iter()
            .map(|s| match s.dtype {
                DType::I8 => ColumnBuilder::I8(PrimitiveBuilder::with_capacity(num_rows)),
                DType::I16 => ColumnBuilder::I16(PrimitiveBuilder::with_capacity(num_rows)),
                DType::I32 => ColumnBuilder::I32(PrimitiveBuilder::with_capacity(num_rows)),
                DType::I64 => ColumnBuilder::I64(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U8 => ColumnBuilder::U8(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U16 => ColumnBuilder::U16(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U32 => ColumnBuilder::U32(PrimitiveBuilder::with_capacity(num_rows)),
                DType::U64 => ColumnBuilder::U64(PrimitiveBuilder::with_capacity(num_rows)),
                DType::F32 => ColumnBuilder::F32(PrimitiveBuilder::with_capacity(num_rows)),
                DType::F64 => ColumnBuilder::F64(PrimitiveBuilder::with_capacity(num_rows)),
                DType::String => {
                    ColumnBuilder::String(GenericByteViewBuilder::with_capacity(num_rows))
                }
            })
            .collect::<Vec<_>>();

        for (spec, builder) in self.specs.iter().zip(builders.iter_mut()) {
            let start = spec.offset;
            let end = spec.offset + spec.length;
            let padding = spec.padding;
            match builder {
                ColumnBuilder::I8(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<i8>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::I16(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<i16>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::I32(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<i32>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::I64(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<i64>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::U8(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<u8>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::U16(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<u16>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::U32(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<u32>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::U64(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<u64>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::F32(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<f32>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::F64(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match lexical_core::parse::<f64>(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
                ColumnBuilder::String(b) => {
                    for row_idx in 0..num_rows {
                        let row_start = row_idx * self.line_length;
                        let field = trim_custom(&data[row_start + start..row_start + end], padding);
                        match std::str::from_utf8(field) {
                            Ok(v) => b.append_value(v),
                            Err(_) => b.append_null(),
                        }
                    }
                }
            }
        }

        let arrays = builders
            .into_iter()
            .map(|b| match b {
                ColumnBuilder::I8(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::I16(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::I32(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::I64(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U8(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U16(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U32(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::U64(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::F32(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::F64(mut b) => Arc::new(b.finish()) as ArrayRef,
                ColumnBuilder::String(mut b) => Arc::new(b.finish()) as ArrayRef,
            })
            .collect();

        RecordBatch::try_new(self.schema.clone(), arrays).expect("Failed to create RecordBatch")
    }
}

pub enum ColumnBuilder {
    I8(PrimitiveBuilder<Int8Type>),
    I16(PrimitiveBuilder<Int16Type>),
    I32(PrimitiveBuilder<Int32Type>),
    I64(PrimitiveBuilder<Int64Type>),
    U8(PrimitiveBuilder<UInt8Type>),
    U16(PrimitiveBuilder<UInt16Type>),
    U32(PrimitiveBuilder<UInt32Type>),
    U64(PrimitiveBuilder<UInt64Type>),
    F32(PrimitiveBuilder<Float32Type>),
    F64(PrimitiveBuilder<Float64Type>),
    String(GenericByteViewBuilder<StringViewType>),
}

#[inline(always)]
pub fn trim_ascii_spaces(slice: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < slice.len() && WHITESPACE_LUT[slice[start] as usize] == 1 {
        start += 1;
    }
    let mut end = slice.len();
    while end > start && WHITESPACE_LUT[slice[end - 1] as usize] == 1 {
        end -= 1;
    }
    &slice[start..end]
}

#[inline(always)]
pub fn trim_custom(slice: &[u8], padding: Option<u8>) -> &[u8] {
    match padding {
        None | Some(b' ') => trim_ascii_spaces(slice),
        Some(p) => {
            let mut start = 0;
            while start < slice.len() && slice[start] == p {
                start += 1;
            }
            let mut end = slice.len();
            while end > start && slice[end - 1] == p {
                end -= 1;
            }
            &slice[start..end]
        }
    }
}

pub struct FwfReader {
    mmap: Mmap,
    parser: FwfParser,
    offset: usize,
    burst_size: usize,
}

impl FwfReader {
    pub fn new(
        path: &str,
        specs: Vec<FieldSpec>,
        line_length: usize,
        parallel: Option<bool>,
        chunk_size: Option<usize>,
    ) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut parser = FwfParser::new(specs, line_length);

        let mut burst = 1;
        if let Some(p) = parallel {
            parser.parallelism = if p {
                burst = rayon::current_num_threads().max(1);
                Par::Fixed(0)
            } else {
                Par::Seq
            };
        }

        if let Some(c) = chunk_size {
            parser.chunk_size = c;
        }

        Ok(Self {
            mmap,
            parser,
            offset: 0,
            burst_size: burst,
        })
    }

    pub fn next_burst(&mut self) -> Vec<RecordBatch> {
        if self.offset >= self.mmap.len() {
            return vec![];
        }

        let batch_bytes = self.parser.chunk_size * self.parser.line_length;
        let burst_bytes = batch_bytes * self.burst_size;

        let end = (self.offset + burst_bytes).min(self.mmap.len());

        let actual_end = if end == self.mmap.len() {
            end
        } else {
            self.offset + ((end - self.offset) / self.parser.line_length) * self.parser.line_length
        };

        if actual_end <= self.offset {
            return vec![];
        }

        let batches = self.parser.parse(&self.mmap[self.offset..actual_end]);
        self.offset = actual_end;
        batches
    }
}
