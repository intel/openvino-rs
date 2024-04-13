//! Define the core interface between Rust and OpenVINO's C
//! [API](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__core__c__api.html).

use crate::error::{IOError, LoadingError, PathError};
use crate::property::{PropertyKey, RwPropertyKey};
use crate::{cstr, drop_using_function, try_unsafe, util::Result, DeviceType, Version};
use crate::{model::CompiledModel, Model};
use crate::{SetupError, Tensor};
use openvino_sys::{
    self, ov_available_devices_free, ov_core_compile_model, ov_core_create,
    ov_core_create_with_config, ov_core_free, ov_core_get_available_devices, ov_core_get_property,
    ov_core_get_versions_by_device_name, ov_core_read_model, ov_core_read_model_from_memory_buffer,
    ov_core_set_property, ov_core_t, ov_core_versions_free,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::slice;
use std::str::FromStr;

const EMPTY_C_STR: &'static CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") };

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
    pub fn versions(&self, device_name: impl AsRef<str>) -> Result<HashMap<DeviceType, Version>> {
        let device_name = cstr!(device_name.as_ref());
        let mut ov_version_list = openvino_sys::ov_core_version_list_t {
            versions: std::ptr::null_mut(),
            size: 0,
        };
        try_unsafe!(ov_core_get_versions_by_device_name(
            self.instance,
            device_name.as_ptr(),
            std::ptr::addr_of_mut!(ov_version_list)
        ))?;

        let ov_versions =
            unsafe { slice::from_raw_parts(ov_version_list.versions, ov_version_list.size) };

        let mut versions: HashMap<DeviceType, Version> =
            HashMap::with_capacity(ov_version_list.size);
        for ov_version in ov_versions {
            let c_str_device_name = unsafe { std::ffi::CStr::from_ptr(ov_version.device_name) };
            let device_name = c_str_device_name.to_string_lossy();
            let device_type = DeviceType::from_str(device_name.as_ref()).unwrap();
            versions.insert(device_type, Version::from(&ov_version.version));
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
            self.instance,
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
    ///
    /// The method extracts information that can be set via the [set_property] method.
    pub fn property(&self, key: PropertyKey) -> Result<Cow<str>> {
        let ov_prop_key = cstr!(key.as_ref());
        let mut ov_prop_value = std::ptr::null_mut();
        try_unsafe!(ov_core_get_property(
            self.instance,
            EMPTY_C_STR.as_ptr(),
            ov_prop_key.as_ptr(),
            std::ptr::addr_of_mut!(ov_prop_value)
        ))?;
        let rust_prop = unsafe { CStr::from_ptr(ov_prop_value) }.to_string_lossy();
        Ok(rust_prop)
    }

    /// Sets a property for this Core instance.
    pub fn set_property(&mut self, key: RwPropertyKey, value: &str) -> Result<()> {
        let ov_prop_key = cstr!(key.as_ref());
        let ov_prop_value = cstr!(value);
        // TODO unable to call variadic C functions
        // try_unsafe!(ov_core_set_property(
        //     self.instance,
        //     EMPTY_C_STR.as_ptr(),
        //     ov_prop_key.as_ptr(),
        //     ov_prop_value.as_ptr(),
        // ))?;
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
    ///
    /// The method extracts information that can be set via the [set_device_property] method.
    pub fn device_property(
        &self,
        device_name: impl AsRef<str>,
        key: PropertyKey,
    ) -> Result<Cow<str>> {
        let ov_device_name = cstr!(device_name.as_ref());
        let ov_prop_key = cstr!(key.as_ref());
        let mut ov_prop_value = std::ptr::null_mut();
        try_unsafe!(ov_core_get_property(
            self.instance,
            ov_device_name.as_ptr(),
            ov_prop_key.as_ptr(),
            std::ptr::addr_of_mut!(ov_prop_value)
        ))?;
        let rust_prop = unsafe { CStr::from_ptr(ov_prop_value) }.to_string_lossy();
        Ok(rust_prop)
    }

    /// Sets a property for a device.
    pub fn set_device_property(
        &mut self,
        device_name: impl AsRef<str>,
        key: RwPropertyKey,
        value: &str,
    ) -> Result<()> {
        // TODO
        Ok(())
    }

    /// Sets properties for a device.
    pub fn set_device_properties<'a>(
        &mut self,
        device_name: impl AsRef<str>,
        properties: impl IntoIterator<Item = (RwPropertyKey, &'a str)>,
    ) -> Result<()> {
        let device_name = device_name.as_ref();
        for (prop_key, prop_value) in properties {
            self.set_device_property(device_name, prop_key, prop_value)?;
        }
        Ok(())
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
