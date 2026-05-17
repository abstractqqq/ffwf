mod core;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{Array, RecordBatch, StructArray};
use pyo3::prelude::*;

#[pyclass(name = "DType", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyDType {
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

impl From<PyDType> for core::DType {
    fn from(dtype: PyDType) -> Self {
        match dtype {
            PyDType::I8 => core::DType::I8,
            PyDType::I16 => core::DType::I16,
            PyDType::I32 => core::DType::I32,
            PyDType::I64 => core::DType::I64,
            PyDType::U8 => core::DType::U8,
            PyDType::U16 => core::DType::U16,
            PyDType::U32 => core::DType::U32,
            PyDType::U64 => core::DType::U64,
            PyDType::F32 => core::DType::F32,
            PyDType::F64 => core::DType::F64,
            PyDType::String => core::DType::String,
        }
    }
}

#[pyclass(name = "FieldSpec")]
#[derive(Debug, Clone)]
pub struct PyFieldSpec {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub offset: usize,
    #[pyo3(get, set)]
    pub length: usize,
    #[pyo3(get, set)]
    pub dtype: PyDType,
    #[pyo3(get, set)]
    pub padding: Option<u8>,
}

#[pymethods]
impl PyFieldSpec {
    #[new]
    #[pyo3(signature = (name, offset, length, dtype, padding=None))]
    pub fn new(
        name: String,
        offset: usize,
        length: usize,
        dtype: PyDType,
        padding: Option<u8>,
    ) -> Self {
        Self {
            name,
            offset,
            length,
            dtype,
            padding,
        }
    }
}

#[pyclass(name = "FwfParser")]
pub struct PyFwfParser {
    inner: core::FwfParser,
}

#[pymethods]
impl PyFwfParser {
    #[new]
    #[pyo3(signature = (specs, line_length, parallel=None, chunk_size=None))]
    pub fn new(
        specs: Vec<PyFieldSpec>,
        line_length: usize,
        parallel: Option<bool>,
        chunk_size: Option<usize>,
    ) -> Self {
        let core_specs = specs
            .into_iter()
            .map(|s| core::FieldSpec {
                name: s.name,
                offset: s.offset,
                length: s.length,
                dtype: s.dtype.into(),
                padding: s.padding,
            })
            .collect();
        let mut inner = core::FwfParser::new(core_specs, line_length);
        if let Some(p) = parallel {
            inner.parallelism = if p {
                core::Par::Fixed(0)
            } else {
                core::Par::Seq
            };
        }
        if let Some(c) = chunk_size {
            inner.chunk_size = c;
        }
        Self { inner }
    }

    pub fn set_chunk_size(&mut self, chunk_size: usize) {
        self.inner.chunk_size = chunk_size;
    }

    #[staticmethod]
    pub fn detect_line_length(path: &str, newline: &[u8]) -> PyResult<(usize, usize)> {
        core::FwfParser::detect_line_length(path, newline)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))
    }

    #[staticmethod]
    pub fn infer_chunk_size(specs: Vec<PyFieldSpec>) -> usize {
        let core_specs: Vec<core::FieldSpec> = specs
            .into_iter()
            .map(|s| core::FieldSpec {
                name: s.name,
                offset: s.offset,
                length: s.length,
                dtype: s.dtype.into(),
                padding: s.padding,
            })
            .collect();
        core::FwfParser::infer_chunk_size(&core_specs)
    }

    pub fn parse(&self, py: Python, data: &[u8]) -> PyResult<Vec<PyObject>> {
        let batches = self.inner.parse(data);
        let mut py_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            py_batches.push(record_batch_to_capsule(py, batch)?);
        }
        Ok(py_batches)
    }

    pub fn _parse_path(&self, py: Python, path: &str) -> PyResult<Vec<PyObject>> {
        let batches = self
            .inner
            .parse_path(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        let mut py_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            py_batches.push(record_batch_to_capsule(py, batch)?);
        }
        Ok(py_batches)
    }
}

#[pyclass(name = "FwfReader")]
pub struct PyFwfReader {
    inner: core::FwfReader,
}

#[pymethods]
impl PyFwfReader {
    #[new]
    #[pyo3(signature = (path, specs, line_length, parallel=None, chunk_size=None))]
    pub fn new(
        path: &str,
        specs: Vec<PyFieldSpec>,
        line_length: usize,
        parallel: Option<bool>,
        chunk_size: Option<usize>,
    ) -> PyResult<Self> {
        let core_specs = specs
            .into_iter()
            .map(|s| core::FieldSpec {
                name: s.name,
                offset: s.offset,
                length: s.length,
                dtype: s.dtype.into(),
                padding: s.padding,
            })
            .collect();
        let inner = core::FwfReader::new(path, core_specs, line_length, parallel, chunk_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        Ok(Self { inner })
    }

    pub fn next_burst(&mut self, py: Python) -> PyResult<Vec<PyObject>> {
        let batches = self.inner.next_burst();
        let mut py_batches = Vec::with_capacity(batches.len());
        for batch in batches {
            py_batches.push(record_batch_to_capsule(py, batch)?);
        }
        Ok(py_batches)
    }
}

unsafe extern "C" fn array_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, c"arrow_array".as_ptr()) };
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut FFI_ArrowArray) };
    }
}

unsafe extern "C" fn schema_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, c"arrow_schema".as_ptr()) };
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut FFI_ArrowSchema) };
    }
}

fn record_batch_to_capsule(py: Python, batch: RecordBatch) -> PyResult<PyObject> {
    let struct_array: StructArray = batch.into();
    let array_data = struct_array.to_data();

    let ffi_array = FFI_ArrowArray::new(&array_data);
    let ffi_schema = FFI_ArrowSchema::try_from(array_data.data_type())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;

    let array_ptr = Box::into_raw(Box::new(ffi_array));
    let schema_ptr = Box::into_raw(Box::new(ffi_schema));

    unsafe {
        let array_capsule = pyo3::ffi::PyCapsule_New(
            array_ptr as *mut _,
            c"arrow_array".as_ptr(),
            Some(array_destructor),
        );
        let schema_capsule = pyo3::ffi::PyCapsule_New(
            schema_ptr as *mut _,
            c"arrow_schema".as_ptr(),
            Some(schema_destructor),
        );

        if array_capsule.is_null() || schema_capsule.is_null() {
            return Err(PyErr::fetch(py));
        }

        let array_obj = PyObject::from_owned_ptr(py, array_capsule);
        let schema_obj = PyObject::from_owned_ptr(py, schema_capsule);

        Ok((array_obj, schema_obj).into_py(py))
    }
}

#[pymodule]
fn _fwf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDType>()?;
    m.add_class::<PyFieldSpec>()?;
    m.add_class::<PyFwfParser>()?;
    m.add_class::<PyFwfReader>()?;
    Ok(())
}
