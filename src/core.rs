use arrow_array::builder::GenericByteViewBuilder;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, StringViewType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;

pub static WHITESPACE_LUT: [u8; 256] = {
    let mut table = [0u8; 256];
    table[b' ' as usize] = 1;
    table[b'\t' as usize] = 1;
    table[b'\n' as usize] = 1;
    table[b'\r' as usize] = 1;
    table
};

#[derive(Clone, Copy)]
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
    pub fn to_arrow(self) -> DataType {
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

    pub fn max_width(&self) -> Option<usize> {
        match self {
            DType::I8 => Some(4),
            DType::U8 => Some(3),
            DType::I16 => Some(6),
            DType::U16 => Some(5),
            DType::I32 => Some(11),
            DType::U32 => Some(10),
            DType::I64 => Some(20),
            DType::U64 => Some(20),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FillValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
}

#[derive(Clone, PartialEq)]
#[allow(dead_code)]
pub enum ErrorStrategy {
    PushNull,
    Fill(FillValue),
}

#[derive(Clone)]
pub struct FieldSpec {
    pub name: String,
    pub offset: usize,
    pub length: usize,
    pub dtype: DType,
    pub padding: Option<u8>,
    pub error_strategy: ErrorStrategy,
}

#[derive(Clone, Copy)]
pub enum Par {
    Seq,
    Fixed(usize),
}

pub struct FwfParser {
    pub specs: Vec<FieldSpec>,
    pub line_length: usize,
    pub schema: Arc<Schema>,
    pub chunk_size: usize,
    pub user_chunk_size: bool,
    pub parallelism: Par,
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
            user_chunk_size: false,
            parallelism: Par::Fixed(0),
        }
    }

    pub fn infer_chunk_size(specs: &[FieldSpec]) -> usize {
        let mut est_bytes_per_row = 0;
        for s in specs {
            match s.dtype {
                DType::String => {
                    if s.length <= 12 {
                        est_bytes_per_row += 16;
                    } else {
                        est_bytes_per_row += 16 + s.length;
                    }
                }
                DType::I8 | DType::U8 => est_bytes_per_row += 1,
                DType::I16 | DType::U16 => est_bytes_per_row += 2,
                DType::I32 | DType::U32 | DType::F32 => est_bytes_per_row += 4,
                DType::I64 | DType::U64 | DType::F64 => est_bytes_per_row += 8,
            }
        }

        let target_batch_size_bytes = 8 * 1024 * 1024; // 8MB target for better L3 cache fit and lower peak memory
        let inferred = target_batch_size_bytes / est_bytes_per_row.max(1);
        inferred.clamp(1024, 65536)
    }

    pub fn detect_line_length(path: &str, newline: &[u8]) -> std::io::Result<(usize, usize)> {
        if newline.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Newline symbol cannot be empty",
            ));
        }
        let mut file = File::open(path)?;
        let mut buf = [0u8; 65536];
        let n = file.read(&mut buf)?;
        if n > 0
            && let Some(pos) = buf[..n]
                .windows(newline.len())
                .position(|window| window == newline)
        {
            let stride = pos + newline.len();
            return Ok((stride, pos));
        }

        let mmap = unsafe { Mmap::map(&file)? };
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
        let aligned_data = &data[..total_rows * self.line_length];

        match self.parallelism {
            Par::Seq => {
                let chunk_size_bytes = self.chunk_size.max(1) * self.line_length;
                aligned_data
                    .chunks(chunk_size_bytes)
                    .map(|chunk| self.parse_batch(chunk))
                    .collect()
            }
            Par::Fixed(n) => {
                let max_threads = rayon::current_num_threads().max(1);
                let num_threads = if n == 0 {
                    max_threads
                } else {
                    n.min(max_threads).max(1)
                };

                let effective_chunk_rows = if self.user_chunk_size {
                    self.chunk_size.max(1)
                } else {
                    let balanced = total_rows.div_ceil(num_threads).max(512);
                    self.chunk_size.min(balanced).max(1)
                };
                let chunk_size_bytes = effective_chunk_rows * self.line_length;

                if num_threads >= max_threads {
                    aligned_data
                        .par_chunks(chunk_size_bytes)
                        .map(|chunk| self.parse_batch(chunk))
                        .collect()
                } else {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(num_threads)
                        .build()
                        .unwrap();

                    pool.install(|| {
                        aligned_data
                            .par_chunks(chunk_size_bytes)
                            .map(|chunk| self.parse_batch(chunk))
                            .collect()
                    })
                }
            }
        }
    }

    pub fn parse_batch(&self, data: &[u8]) -> RecordBatch {
        let num_rows = data.len() / self.line_length;
        if num_rows == 0 || self.specs.is_empty() {
            let arrays: Vec<ArrayRef> = self
                .specs
                .iter()
                .map(|spec| parse_column_range(data, 0, num_rows, num_rows, self.line_length, spec))
                .collect();
            return RecordBatch::try_new(self.schema.clone(), arrays)
                .expect("Failed to create RecordBatch");
        }

        // Target ~2MB of input data per tile so all columns stay in L2/L3 cache with minimal tile overhead
        let tile_rows = if self.specs.len() <= 4 {
            num_rows
        } else {
            (2 * 1024 * 1024 / self.line_length.max(1))
                .clamp(512, 8192)
                .min(num_rows)
        };

        if tile_rows >= num_rows {
            let arrays: Vec<ArrayRef> = self
                .specs
                .iter()
                .map(|spec| parse_column_range(data, 0, num_rows, num_rows, self.line_length, spec))
                .collect();
            return RecordBatch::try_new(self.schema.clone(), arrays)
                .expect("Failed to create RecordBatch");
        }

        let mut accs: Vec<ColumnAcc> = self
            .specs
            .iter()
            .map(|spec| ColumnAcc::with_capacity(spec, num_rows))
            .collect();

        let mut tile_start = 0;
        while tile_start < num_rows {
            let tile_end = (tile_start + tile_rows).min(num_rows);
            for (spec, acc) in self.specs.iter().zip(accs.iter_mut()) {
                acc.parse_tile(data, tile_start, tile_end, num_rows, self.line_length, spec);
            }
            tile_start = tile_end;
        }

        let arrays: Vec<ArrayRef> = accs.into_iter().map(|acc| acc.finish(num_rows)).collect();
        RecordBatch::try_new(self.schema.clone(), arrays).expect("Failed to create RecordBatch")
    }
}

struct TypedColAcc<T> {
    values: Vec<T>,
    null_bitmap: Option<Vec<u8>>,
}

impl<T> TypedColAcc<T> {
    fn with_capacity(num_rows: usize) -> Self {
        Self {
            values: Vec::with_capacity(num_rows),
            null_bitmap: None,
        }
    }
}

enum ColumnAcc {
    I8(TypedColAcc<i8>),
    I16(TypedColAcc<i16>),
    I32(TypedColAcc<i32>),
    I64(TypedColAcc<i64>),
    U8(TypedColAcc<u8>),
    U16(TypedColAcc<u16>),
    U32(TypedColAcc<u32>),
    U64(TypedColAcc<u64>),
    F32(TypedColAcc<f32>),
    F64(TypedColAcc<f64>),
    ShortString(TypedColAcc<u128>),
    LongString(GenericByteViewBuilder<StringViewType>),
}

impl ColumnAcc {
    fn with_capacity(spec: &FieldSpec, num_rows: usize) -> Self {
        match spec.dtype {
            DType::I8 => ColumnAcc::I8(TypedColAcc::with_capacity(num_rows)),
            DType::I16 => ColumnAcc::I16(TypedColAcc::with_capacity(num_rows)),
            DType::I32 => ColumnAcc::I32(TypedColAcc::with_capacity(num_rows)),
            DType::I64 => ColumnAcc::I64(TypedColAcc::with_capacity(num_rows)),
            DType::U8 => ColumnAcc::U8(TypedColAcc::with_capacity(num_rows)),
            DType::U16 => ColumnAcc::U16(TypedColAcc::with_capacity(num_rows)),
            DType::U32 => ColumnAcc::U32(TypedColAcc::with_capacity(num_rows)),
            DType::U64 => ColumnAcc::U64(TypedColAcc::with_capacity(num_rows)),
            DType::F32 => ColumnAcc::F32(TypedColAcc::with_capacity(num_rows)),
            DType::F64 => ColumnAcc::F64(TypedColAcc::with_capacity(num_rows)),
            DType::String => {
                let can_inline = spec.length <= 12
                    && match &spec.error_strategy {
                        ErrorStrategy::Fill(FillValue::String(s)) => s.len() <= 12,
                        _ => true,
                    };
                if can_inline {
                    ColumnAcc::ShortString(TypedColAcc::with_capacity(num_rows))
                } else {
                    ColumnAcc::LongString(GenericByteViewBuilder::with_capacity(num_rows))
                }
            }
        }
    }

    #[inline]
    fn parse_tile(
        &mut self,
        data: &[u8],
        start_row: usize,
        end_row: usize,
        num_rows: usize,
        line_length: usize,
        spec: &FieldSpec,
    ) {
        match (self, &spec.error_strategy) {
            (ColumnAcc::I8(acc), ErrorStrategy::Fill(FillValue::I8(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::I8(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::I16(acc), ErrorStrategy::Fill(FillValue::I16(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::I16(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::I32(acc), ErrorStrategy::Fill(FillValue::I32(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::I32(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::I64(acc), ErrorStrategy::Fill(FillValue::I64(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::I64(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::U8(acc), ErrorStrategy::Fill(FillValue::U8(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::U8(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::U16(acc), ErrorStrategy::Fill(FillValue::U16(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::U16(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::U32(acc), ErrorStrategy::Fill(FillValue::U32(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::U32(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::U64(acc), ErrorStrategy::Fill(FillValue::U64(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::U64(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::F32(acc), ErrorStrategy::Fill(FillValue::F32(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::F32(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::F64(acc), ErrorStrategy::Fill(FillValue::F64(f))) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, Some(*f), acc)
            }
            (ColumnAcc::F64(acc), _) => {
                parse_primitive_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::ShortString(acc), ErrorStrategy::Fill(FillValue::String(f))) => {
                parse_short_string_tile(data, start_row, end_row, num_rows, line_length, spec, Some(f.as_str()), acc)
            }
            (ColumnAcc::ShortString(acc), _) => {
                parse_short_string_tile(data, start_row, end_row, num_rows, line_length, spec, None, acc)
            }
            (ColumnAcc::LongString(builder), ErrorStrategy::Fill(FillValue::String(f))) => {
                parse_long_string_tile(data, start_row, end_row, line_length, spec, Some(f.as_str()), builder)
            }
            (ColumnAcc::LongString(builder), _) => {
                parse_long_string_tile(data, start_row, end_row, line_length, spec, None, builder)
            }
        }
    }

    fn finish(self, num_rows: usize) -> ArrayRef {
        match self {
            ColumnAcc::I8(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<Int8Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::I16(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<Int16Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::I32(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<Int32Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::I64(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<Int64Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::U8(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<UInt8Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::U16(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<UInt16Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::U32(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<UInt32Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::U64(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<UInt64Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::F32(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<Float32Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::F64(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                Arc::new(PrimitiveArray::<Float64Type>::new(
                    ScalarBuffer::from(acc.values),
                    finish_nulls(acc.null_bitmap, num_rows),
                ))
            }
            ColumnAcc::ShortString(mut acc) => {
                unsafe { acc.values.set_len(num_rows) };
                let nulls = finish_nulls(acc.null_bitmap, num_rows);
                let array = unsafe {
                    StringViewArray::new_unchecked(ScalarBuffer::from(acc.values), vec![], nulls)
                };
                Arc::new(array)
            }
            ColumnAcc::LongString(mut builder) => Arc::new(builder.finish()),
        }
    }
}

fn parse_column_range(
    data: &[u8],
    start_row: usize,
    end_row: usize,
    num_rows: usize,
    line_length: usize,
    spec: &FieldSpec,
) -> ArrayRef {
    let mut acc = ColumnAcc::with_capacity(spec, num_rows);
    acc.parse_tile(data, start_row, end_row, num_rows, line_length, spec);
    acc.finish(num_rows)
}

#[cold]
#[inline(never)]
fn mark_null(null_bitmap: &mut Option<Vec<u8>>, row_idx: usize, num_rows: usize) {
    let bitmap = null_bitmap.get_or_insert_with(|| vec![0xFFu8; num_rows.div_ceil(8)]);
    bitmap[row_idx >> 3] &= !(1u8 << (row_idx & 7));
}

#[inline]
fn finish_nulls(null_bitmap: Option<Vec<u8>>, num_rows: usize) -> Option<NullBuffer> {
    null_bitmap.map(|bytes| {
        let buf = Buffer::from_vec(bytes);
        NullBuffer::new(BooleanBuffer::new(buf, 0, num_rows))
    })
}

#[allow(clippy::too_many_arguments)]
fn parse_primitive_tile<N>(
    data: &[u8],
    start_row: usize,
    end_row: usize,
    num_rows: usize,
    line_length: usize,
    spec: &FieldSpec,
    fill: Option<N>,
    acc: &mut TypedColAcc<N>,
) where
    N: lexical_core::FromLexical + Default + Copy,
{
    assert!(spec.offset + spec.length <= line_length);
    assert!(end_row * line_length <= data.len());
    let values_ptr = acc.values.as_mut_ptr();
    let null_bitmap = &mut acc.null_bitmap;
    let len = spec.length;
    let mut pos = start_row * line_length + spec.offset;

    match (fill, spec.padding) {
        (Some(fill_val), None | Some(b' ')) => {
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_ascii_spaces(raw);
                let val = lexical_core::parse::<N>(field).unwrap_or(fill_val);
                unsafe { *values_ptr.add(row_idx) = val };
            }
        }
        (Some(fill_val), Some(pad)) => {
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_byte(raw, pad);
                let val = lexical_core::parse::<N>(field).unwrap_or(fill_val);
                unsafe { *values_ptr.add(row_idx) = val };
            }
        }
        (None, None | Some(b' ')) => {
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_ascii_spaces(raw);
                match lexical_core::parse::<N>(field) {
                    Ok(v) => unsafe { *values_ptr.add(row_idx) = v },
                    Err(_) => {
                        unsafe { *values_ptr.add(row_idx) = N::default() };
                        mark_null(null_bitmap, row_idx, num_rows);
                    }
                }
            }
        }
        (None, Some(pad)) => {
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_byte(raw, pad);
                match lexical_core::parse::<N>(field) {
                    Ok(v) => unsafe { *values_ptr.add(row_idx) = v },
                    Err(_) => {
                        unsafe { *values_ptr.add(row_idx) = N::default() };
                        mark_null(null_bitmap, row_idx, num_rows);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn make_inline_view(s: &[u8]) -> u128 {
    debug_assert!(s.len() <= 12);
    let mut buf = [0u8; 16];
    let len = s.len();
    buf[0..4].copy_from_slice(&(len as u32).to_le_bytes());
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf.as_mut_ptr().add(4), len);
    }
    u128::from_ne_bytes(buf)
}

#[inline(always)]
fn is_inline_view_ascii(view: u128) -> bool {
    (view & 0x8080_8080_8080_8080_8080_8080_8080_8080u128) == 0
}

#[allow(clippy::too_many_arguments)]
fn parse_short_string_tile(
    data: &[u8],
    start_row: usize,
    end_row: usize,
    num_rows: usize,
    line_length: usize,
    spec: &FieldSpec,
    fill: Option<&str>,
    acc: &mut TypedColAcc<u128>,
) {
    assert!(spec.offset + spec.length <= line_length);
    assert!(end_row * line_length <= data.len());
    let views_ptr = acc.values.as_mut_ptr();
    let null_bitmap = &mut acc.null_bitmap;
    let len = spec.length;
    let mut pos = start_row * line_length + spec.offset;

    match (fill, spec.padding) {
        (Some(fill_str), None | Some(b' ')) => {
            let fill_view = make_inline_view(fill_str.as_bytes());
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_ascii_spaces(raw);
                let view = make_inline_view(field);
                if is_inline_view_ascii(view) || std::str::from_utf8(field).is_ok() {
                    unsafe { *views_ptr.add(row_idx) = view };
                } else {
                    unsafe { *views_ptr.add(row_idx) = fill_view };
                }
            }
        }
        (Some(fill_str), Some(pad)) => {
            let fill_view = make_inline_view(fill_str.as_bytes());
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_byte(raw, pad);
                let view = make_inline_view(field);
                if is_inline_view_ascii(view) || std::str::from_utf8(field).is_ok() {
                    unsafe { *views_ptr.add(row_idx) = view };
                } else {
                    unsafe { *views_ptr.add(row_idx) = fill_view };
                }
            }
        }
        (None, None | Some(b' ')) => {
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_ascii_spaces(raw);
                let view = make_inline_view(field);
                if is_inline_view_ascii(view) || std::str::from_utf8(field).is_ok() {
                    unsafe { *views_ptr.add(row_idx) = view };
                } else {
                    unsafe { *views_ptr.add(row_idx) = 0 };
                    mark_null(null_bitmap, row_idx, num_rows);
                }
            }
        }
        (None, Some(pad)) => {
            for row_idx in start_row..end_row {
                let raw = unsafe { data.get_unchecked(pos..pos + len) };
                pos += line_length;
                let field = trim_byte(raw, pad);
                let view = make_inline_view(field);
                if is_inline_view_ascii(view) || std::str::from_utf8(field).is_ok() {
                    unsafe { *views_ptr.add(row_idx) = view };
                } else {
                    unsafe { *views_ptr.add(row_idx) = 0 };
                    mark_null(null_bitmap, row_idx, num_rows);
                }
            }
        }
    }
}

fn parse_long_string_tile(
    data: &[u8],
    start_row: usize,
    end_row: usize,
    line_length: usize,
    spec: &FieldSpec,
    fill: Option<&str>,
    builder: &mut GenericByteViewBuilder<StringViewType>,
) {
    assert!(spec.offset + spec.length <= line_length);
    assert!(end_row * line_length <= data.len());
    let len = spec.length;
    let mut pos = start_row * line_length + spec.offset;
    for _ in start_row..end_row {
        let raw = unsafe { data.get_unchecked(pos..pos + len) };
        pos += line_length;
        let field = trim_custom(raw, spec.padding);
        match (std::str::from_utf8(field), fill) {
            (Ok(v), _) => builder.append_value(v),
            (Err(_), Some(fill_str)) => builder.append_value(fill_str),
            (Err(_), None) => builder.append_null(),
        }
    }
}

#[inline(always)]
pub fn trim_ascii_spaces(slice: &[u8]) -> &[u8] {
    let len = slice.len();
    let mut start = 0;
    while start < len && unsafe { WHITESPACE_LUT[*slice.get_unchecked(start) as usize] } == 1 {
        start += 1;
    }
    let mut end = len;
    while end > start && unsafe { WHITESPACE_LUT[*slice.get_unchecked(end - 1) as usize] } == 1 {
        end -= 1;
    }
    unsafe { slice.get_unchecked(start..end) }
}

#[inline(always)]
pub fn trim_byte(slice: &[u8], p: u8) -> &[u8] {
    let len = slice.len();
    let mut start = 0;
    while start < len && unsafe { *slice.get_unchecked(start) } == p {
        start += 1;
    }
    let mut end = len;
    while end > start && unsafe { *slice.get_unchecked(end - 1) } == p {
        end -= 1;
    }
    unsafe { slice.get_unchecked(start..end) }
}

#[inline(always)]
pub fn trim_custom(slice: &[u8], padding: Option<u8>) -> &[u8] {
    match padding {
        None | Some(b' ') => trim_ascii_spaces(slice),
        Some(p) => trim_byte(slice, p),
    }
}

pub struct FwfReader {
    mmap: Mmap,
    parser: FwfParser,
    offset: usize,
    end_limit: usize,
    burst_size: usize,
}

impl FwfReader {
    pub fn new(
        path: &str,
        specs: Vec<FieldSpec>,
        line_length: usize,
        parallel: Option<bool>,
        chunk_size: Option<usize>,
        max_rows: Option<usize>,
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
            parser.user_chunk_size = true;
        }

        let end_limit = match max_rows {
            Some(rows) => rows.saturating_mul(line_length).min(mmap.len()),
            None => mmap.len(),
        };

        Ok(Self {
            mmap,
            parser,
            offset: 0,
            end_limit,
            burst_size: burst,
        })
    }

    pub fn next_burst(&mut self) -> Vec<RecordBatch> {
        if self.offset >= self.end_limit {
            return vec![];
        }

        let batch_bytes = self.parser.chunk_size * self.parser.line_length;
        let burst_bytes = batch_bytes * self.burst_size;

        let end = (self.offset + burst_bytes).min(self.end_limit);

        let actual_end =
            self.offset + ((end - self.offset) / self.parser.line_length) * self.parser.line_length;

        if actual_end <= self.offset {
            return vec![];
        }

        let batches = self.parser.parse(&self.mmap[self.offset..actual_end]);
        self.offset = actual_end;
        batches
    }
}
