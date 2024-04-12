//! Define the core interface between Rust and OpenVINO's C
//! [API](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__core__c__api.html).

use crate::error::{IOError, LoadingError, PathError};
use crate::property::PropertyKey;
use crate::{cstr, drop_using_function, try_unsafe, util::Result, DeviceType, Version};
use crate::{model::CompiledModel, Model};
use crate::{SetupError, Tensor};
use openvino_sys::{
    self, ov_core_compile_model, ov_core_create, ov_core_create_with_config, ov_core_free,
    ov_core_read_model, ov_core_read_model_from_memory_buffer, ov_core_t,
};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

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
    pub fn new_with_config<P>(xml_config_file: Option<P>) -> std::result::Result<Core, SetupError>
    where
        P: AsRef<Path>,
    {
        openvino_sys::library::load().map_err(LoadingError::SystemFailure)?;
        let file = if let Some(file) = xml_config_file {
            cstr!(file
                .as_ref()
                .to_str()
                .ok_or(LoadingError::InvalidPath(PathError::CannotStringify))?)
        } else if let Some(file) = openvino_finder::find_plugins_xml() {
            cstr!(file
                .to_str()
                .ok_or(LoadingError::InvalidPath(PathError::CannotStringify))?)
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

    /// Gets device plugins version information.
    ///
    /// Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
    /// in this case, the returned map contains multiple entries, each per device.
    pub fn versions(&self, device_name: &str) -> HashMap<DeviceType, Version> {
        todo!()
    }

    /// Gets devices available for inference.
    pub fn available_devices(&self) -> Vec<DeviceType> {
        todo!()
    }

    /// Gets properties related to device behaviour.
    ///
    /// The method extracts information that can be set via the [set_property] method.
    pub fn property(&self, device_name: DeviceType, key: PropertyKey) -> &str {
        todo!()
    }

    /// Sets properties for a device.
    pub fn set_property(&mut self, device_name: DeviceType, key: PropertyKey, value: &str) {
        todo!()
    }

    /// Read a Model from a pair of files: `model` points to an XML file containing the
    /// OpenVINO model IR and `weights` points to the binary weights file.
    pub fn read_model_from_file<MP, WP>(
        &mut self,
        model: MP,
        weights: WP,
    ) -> std::result::Result<Model, IOError>
    where
        MP: AsRef<Path>,
        WP: AsRef<Path>,
    {
        let model_path = cstr!(model.as_ref().to_str().ok_or(PathError::CannotStringify)?);
        let weights_path = cstr!(weights
            .as_ref()
            .to_str()
            .ok_or(PathError::CannotStringify)?);

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
    pub fn read_model_from_buffer(&mut self, model_data: &[u8], weights: &Tensor) -> Result<Model> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model_from_memory_buffer(
            self.instance,
            model_data.as_ptr().cast(),
            model_data.len(),
            weights.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(Model { instance })
    }

    /// Compile a model.
    pub fn compile_model(&mut self, model: &Model, device: DeviceType) -> Result<CompiledModel> {
        let device: CString = device.into();
        let num_property_args = 0;
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_compile_model(
            self.instance,
            model.instance,
            device.as_ptr(),
            num_property_args,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(CompiledModel { instance })
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
