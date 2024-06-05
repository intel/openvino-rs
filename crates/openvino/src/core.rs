//! Define the core interface between Rust and OpenVINO's C
//! [API](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__core__c__api.html).

use crate::error::LoadingError;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use crate::{model::CompiledModel, Model};
use crate::{DeviceType, PropertyKey, RwPropertyKey, SetupError, Tensor, Version};
use openvino_sys::{
    self, ov_available_devices_free, ov_core_compile_model, ov_core_create,
    ov_core_create_with_config, ov_core_free, ov_core_get_available_devices, ov_core_get_property,
    ov_core_get_versions_by_device_name, ov_core_read_model, ov_core_read_model_from_memory_buffer,
    ov_core_set_property, ov_core_t, ov_core_versions_free,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;
use std::str::FromStr;

const EMPTY_C_STR: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") };

/// See [`Core`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__core__c__api.html).
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
        let xml_config_file = cstr!(xml_config_file);
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_core_create_with_config(
            xml_config_file.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Core { ptr })
    }

    /// Gets device plugins version information.
    /// Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
    /// in this case, the returned map contains multiple entries, each per device.
    pub fn versions(&self, device_name: &str) -> Result<Vec<(DeviceType, Version)>> {
        let device_name = cstr!(device_name);
        let mut ov_version_list = openvino_sys::ov_core_version_list_t {
            versions: std::ptr::null_mut(),
            size: 0,
        };
        try_unsafe!(ov_core_get_versions_by_device_name(
            self.ptr,
            device_name.as_ptr(),
            std::ptr::addr_of_mut!(ov_version_list)
        ))?;

        let ov_versions =
            unsafe { slice::from_raw_parts(ov_version_list.versions, ov_version_list.size) };

        let mut versions: Vec<(DeviceType, Version)> = Vec::with_capacity(ov_version_list.size);
        for ov_version in ov_versions {
            let c_str_device_name = unsafe { std::ffi::CStr::from_ptr(ov_version.device_name) };
            let device_name = c_str_device_name.to_string_lossy();
            let device_type = DeviceType::from_str(device_name.as_ref()).unwrap();
            versions.push((device_type, Version::from(&ov_version.version)));
        }

        unsafe { ov_core_versions_free(std::ptr::addr_of_mut!(ov_version_list)) };
        Ok(versions)
    }

    /// Gets devices available for inference.
    pub fn available_devices(&self) -> Result<Vec<DeviceType>> {
        let mut ov_available_devices = openvino_sys::ov_available_devices_t {
            devices: std::ptr::null_mut(),
            size: 0,
        };
        try_unsafe!(ov_core_get_available_devices(
            self.ptr,
            std::ptr::addr_of_mut!(ov_available_devices)
        ))?;

        let ov_devices = unsafe {
            slice::from_raw_parts(ov_available_devices.devices, ov_available_devices.size)
        };

        let mut devices = Vec::with_capacity(ov_available_devices.size);
        for ov_device in ov_devices {
            let c_str_device_name = unsafe { std::ffi::CStr::from_ptr(*ov_device) };
            let device_name = c_str_device_name.to_string_lossy();
            let device_type = DeviceType::from_str(device_name.as_ref()).unwrap();
            devices.push(device_type);
        }

        unsafe { ov_available_devices_free(std::ptr::addr_of_mut!(ov_available_devices)) };
        Ok(devices)
    }

    /// Gets properties related to this Core.
    /// The method extracts information that can be set via the [set_property] method.
    pub fn get_property(&self, key: PropertyKey) -> Result<String> {
        let ov_prop_key = cstr!(key.as_ref());
        let mut ov_prop_value = std::ptr::null_mut();
        try_unsafe!(ov_core_get_property(
            self.ptr,
            EMPTY_C_STR.as_ptr(),
            ov_prop_key.as_ptr(),
            std::ptr::addr_of_mut!(ov_prop_value)
        ))?;
        let rust_prop = unsafe { CStr::from_ptr(ov_prop_value) }
            .to_str()
            .unwrap()
            .to_owned();
        Ok(rust_prop)
    }

    /// Sets a property for this Core instance.
    pub fn set_property(&mut self, key: RwPropertyKey, value: &str) -> Result<()> {
        let ov_prop_key = cstr!(key.as_ref());
        let ov_prop_value = cstr!(value);
        try_unsafe!(ov_core_set_property(
            self.ptr,
            EMPTY_C_STR.as_ptr(),
            ov_prop_key.as_ptr(),
            ov_prop_value.as_ptr(),
        ))?;
        Ok(())
    }

    /// Sets properties for this Core instance.
    pub fn set_properties<'a>(
        &mut self,
        properties: impl IntoIterator<Item = (RwPropertyKey, &'a str)>,
    ) -> Result<()> {
        for (prop_key, prop_value) in properties {
            self.set_property(prop_key, prop_value)?;
        }
        Ok(())
    }

    /// Gets properties related to device behaviour.
    /// The method extracts information that can be set via the [set_device_property] method.
    pub fn get_device_property(&self, device_name: &str, key: PropertyKey) -> Result<String> {
        let ov_device_name = cstr!(device_name);
        let ov_prop_key = cstr!(key.as_ref());
        let mut ov_prop_value = std::ptr::null_mut();
        try_unsafe!(ov_core_get_property(
            self.ptr,
            ov_device_name.as_ptr(),
            ov_prop_key.as_ptr(),
            std::ptr::addr_of_mut!(ov_prop_value)
        ))?;
        let rust_prop = unsafe { CStr::from_ptr(ov_prop_value) }
            .to_str()
            .unwrap()
            .to_owned();
        Ok(rust_prop)
    }

    /// Sets a property for a device.
    pub fn set_device_property(
        &mut self,
        device_name: &str,
        key: RwPropertyKey,
        value: &str,
    ) -> Result<()> {
        let ov_device_name = cstr!(device_name);
        let ov_prop_key = cstr!(key.as_ref());
        let ov_prop_value = cstr!(value);
        try_unsafe!(ov_core_set_property(
            self.ptr,
            ov_device_name.as_ptr(),
            ov_prop_key.as_ptr(),
            ov_prop_value.as_ptr(),
        ))?;
        Ok(())
    }

    /// Sets properties for a device.
    pub fn set_device_properties<'a>(
        &mut self,
        device_name: &str,
        properties: impl IntoIterator<Item = (RwPropertyKey, &'a str)>,
    ) -> Result<()> {
        for (prop_key, prop_value) in properties {
            self.set_device_property(device_name, prop_key, prop_value)?;
        }
        Ok(())
    }

    /// Read a Model from a pair of files: `model_path` points to an XML file containing the
    /// OpenVINO model IR and `weights_path` points to the binary weights file.
    pub fn read_model_from_file(&mut self, model_path: &str, weights_path: &str) -> Result<Model> {
        let model_path = cstr!(model_path);
        let weights_path = cstr!(weights_path);
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model(
            self.ptr,
            model_path.as_ptr(),
            weights_path.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Model::from_ptr(ptr))
    }

    /// Read model with model and weights loaded in memory.
    pub fn read_model_from_buffer(
        &mut self,
        model_str: &[u8],
        weights_buffer: Option<&Tensor>,
    ) -> Result<Model> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model_from_memory_buffer(
            self.ptr,
            model_str.as_ptr().cast::<c_char>(),
            model_str.len(),
            weights_buffer.map_or(std::ptr::null(), |tensor| tensor.as_ptr()),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Model::from_ptr(ptr))
    }

    /// Compile a model to `CompiledModel`.
    pub fn compile_model(&mut self, model: &Model, device: DeviceType) -> Result<CompiledModel> {
        let device: CString = device.into();
        let mut compiled_model = std::ptr::null_mut();
        let num_property_args = 0;
        try_unsafe!(ov_core_compile_model(
            self.ptr,
            model.as_ptr(),
            device.as_ptr(),
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
