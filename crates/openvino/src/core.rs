//! Define the core interface between Rust and OpenVINO's C
//! [API](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__core__c__api.html).

use crate::error::LoadingError;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use crate::{model::CompiledModel, Model};
use crate::{SetupError, Tensor};
use std::path::Path;

use openvino_sys::{
    self, ov_core_compile_model, ov_core_create, ov_core_create_with_config, ov_core_free,
    ov_core_read_model, ov_core_read_model_from_memory_buffer, ov_core_t,
};

/// See [Core](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__core__c__api.html).
pub struct Core {
    instance: *mut ov_core_t,
}
drop_using_function!(Core, ov_core_free);

unsafe impl Send for Core {}

impl Core {
    /// Construct a new OpenVINO [`Core`].
    pub fn new() -> std::result::Result<Core, SetupError> {
        openvino_sys::library::load().map_err(LoadingError::SystemFailure)?;
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_create(std::ptr::addr_of_mut!(instance)))?;
        Ok(Core { instance })
    }

    /// Construct a new OpenVINO [`Core`] with an XML config file.
    pub fn new_with_config(
        xml_config_file: Option<&Path>,
    ) -> std::result::Result<Core, SetupError> {
        openvino_sys::library::load().map_err(LoadingError::SystemFailure)?;
        let file = if let Some(file) = xml_config_file {
            cstr!(file.to_str().ok_or(LoadingError::CannotStringifyPath)?)
        } else if let Some(file) = openvino_finder::find_plugins_xml() {
            cstr!(file.to_str().ok_or(LoadingError::CannotStringifyPath)?)
        } else {
            cstr!("")
        };

        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_create_with_config(
            file.as_ptr(),
            std::ptr::addr_of_mut!(instance)
        ))?;

        Ok(Core { instance })
    }

    /// Read a Model from a pair of files: `model_path` points to an XML file containing the
    /// OpenVINO model IR and `weights_path` points to the binary weights file.
    pub fn read_model_from_file(&mut self, model_path: &str, weights_path: &str) -> Result<Model> {
        let model_path = cstr!(model_path);
        let weights_path = cstr!(weights_path);

        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model(
            self.instance,
            model_path.as_ptr(),
            weights_path.as_ptr(),
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(Model { instance })
    }

    /// Read model with model and weights loaded in memory.
    pub fn read_model_from_buffer(
        &mut self,
        model_str: &str,
        weights_buffer: &Tensor,
    ) -> Result<Model> {
        let model_name = cstr!(model_str);

        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model_from_memory_buffer(
            self.instance,
            model_name.as_ptr(),
            model_str.len(),
            weights_buffer.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(Model { instance })
    }

    /// Compile a model.
    pub fn compile_model(&mut self, model: &Model, device: &str) -> Result<CompiledModel> {
        let device = cstr!(device);
        let mut compiled_model = CompiledModel {
            instance: std::ptr::null_mut(),
        };
        let num_property_args = 0;
        try_unsafe!(ov_core_compile_model(
            self.instance,
            model.instance,
            device.as_ptr(),
            num_property_args,
            std::ptr::addr_of_mut!(compiled_model.instance)
        ))?;
        Ok(compiled_model)
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
}
