//! Define the core interface between Rust and OpenVINO's C
//! [API](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__core__c__api.html).

use crate::error::LoadingError;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use crate::{model::CompiledModel, Model};
use crate::{SetupError, Tensor};
use openvino_sys::{
    self, ov_core_compile_model, ov_core_create, ov_core_create_with_config, ov_core_free,
    ov_core_read_model, ov_core_read_model_from_memory_buffer, ov_core_t,
};

use std::os::raw::c_char;

/// See [`Core`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__core__c__api.html).
pub struct Core {
    ptr: *mut ov_core_t,
}
drop_using_function!(Core, ov_core_free);

unsafe impl Send for Core {}

impl Core {
    /// Construct a new OpenVINO [`Core`].
    pub fn new() -> std::result::Result<Core, SetupError> {
        openvino_sys::library::load().map_err(LoadingError::SystemFailure)?;
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_core_create(std::ptr::addr_of_mut!(ptr)))?;
        Ok(Core { ptr })
    }

    /// Construct a new OpenVINO [`Core`] with config specified in an xml file.
    pub fn new_with_config(xml_config_file: &str) -> std::result::Result<Core, SetupError> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_core_create_with_config(
            cstr!(xml_config_file.to_string()),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Core { ptr })
    }

    /// Read a Model from a pair of files: `model_path` points to an XML file containing the
    /// OpenVINO model IR and `weights_path` points to the binary weights file.
    pub fn read_model_from_file(&mut self, model_path: &str, weights_path: &str) -> Result<Model> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model(
            self.ptr,
            cstr!(model_path),
            cstr!(weights_path),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Model::from_ptr(ptr))
    }

    /// Read model with model and weights loaded in memory.
    pub fn read_model_from_buffer<'a, T: Into<Option<&'a Tensor>>>(
        &mut self,
        model_str: &[u8],
        weights_buffer: T,
    ) -> Result<Model> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model_from_memory_buffer(
            self.ptr,
            model_str.as_ptr().cast::<c_char>(),
            model_str.len(),
            weights_buffer
                .into()
                .map_or(std::ptr::null(), |tensor| tensor.as_ptr().cast_const()),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Model::from_ptr(ptr))
    }

    /// Compile a model to `CompiledModel`.
    pub fn compile_model(&mut self, model: &Model, device: &str) -> Result<CompiledModel> {
        let mut compiled_model = std::ptr::null_mut();
        let num_property_args = 0;
        try_unsafe!(ov_core_compile_model(
            self.ptr,
            model.as_ptr(),
            cstr!(device),
            num_property_args,
            std::ptr::addr_of_mut!(compiled_model)
        ))?;
        Ok(CompiledModel::from_ptr(compiled_model))
    }
}

#[cfg(test)]
mod core_tests {
    use super::*;

    #[test]
    fn test_new() {
        let core = Core::new();
        assert!(core.is_ok());
    }

    #[test]
    fn test_load_onnx_from_buffer() {
        let model = b"\x08\x07\x12\nonnx-wally:j\n*\n\x06inputs\x12\x07outputs\x1a\ridentity_node\"\x08Identity\x12\x0bno-op-modelZ\x16\n\x06inputs\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08\x02b\x17\n\x07outputs\x12\x0c\n\n\x08\x01\x12\x06\n\x00\n\x02\x08\x02B\x02\x10\x0c";
        let mut core = Core::new().unwrap();
        let model = core.read_model_from_buffer(model, None);
        assert!(model.is_ok());
    }
}
