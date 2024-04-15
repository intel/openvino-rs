//! Contains the model representations in OpenVINO:
//!  - [`Model`] is the OpenVINO representation of a neural model
//!  - [`CompiledModel`] is the compiled representation of a [`Model`] for a device.

use crate::port::Port;
use crate::request::InferRequest;
use crate::{cstr, drop_using_function, try_unsafe, util::Result, PropertyKey, RwPropertyKey};
use std::borrow::Cow;
use std::ffi::CStr;
use std::path::Path;

use crate::error::{IOError, PathError};
use openvino_sys::{
    ov_compiled_model_create_infer_request, ov_compiled_model_export_model, ov_compiled_model_free,
    ov_compiled_model_get_property, ov_compiled_model_get_runtime_model, ov_compiled_model_input,
    ov_compiled_model_input_by_index, ov_compiled_model_input_by_name,
    ov_compiled_model_inputs_size, ov_compiled_model_output, ov_compiled_model_output_by_index,
    ov_compiled_model_output_by_name, ov_compiled_model_outputs_size,
    ov_compiled_model_set_property, ov_compiled_model_t, ov_model_const_input_by_index,
    ov_model_const_output_by_index, ov_model_free, ov_model_inputs_size, ov_model_outputs_size,
    ov_model_t,
};

/// See [`Model`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__model__c__api.html).
pub struct Model {
    pub(crate) instance: *mut ov_model_t,
}
drop_using_function!(Model, ov_model_free);

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Retrieve the number of model inputs.
    pub fn input_size(&self) -> Result<usize> {
        let mut input_size: usize = 0;
        try_unsafe!(ov_model_inputs_size(self.instance, &mut input_size))?;
        Ok(input_size)
    }

    /// Retrieve the input port by index.
    pub fn input_by_index(&self, index: usize) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_model_const_input_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Retrieve the number of model outputs.
    pub fn output_size(&self) -> Result<usize> {
        let mut output_size: usize = 0;
        try_unsafe!(ov_model_outputs_size(self.instance, &mut output_size))?;
        Ok(output_size)
    }

    /// Retrieve the output port by index.
    pub fn output_by_index(&self, index: usize) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }
}

/// See [`CompiledModel`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__compiled__model__c__api.html).
pub struct CompiledModel {
    pub(crate) instance: *mut ov_compiled_model_t,
}
drop_using_function!(CompiledModel, ov_compiled_model_free);

unsafe impl Send for CompiledModel {}

impl CompiledModel {
    /// Get the input size of the compiled model.
    pub fn input_size(&self) -> Result<usize> {
        let mut input_size: usize = 0;
        try_unsafe!(ov_compiled_model_inputs_size(
            self.instance,
            &mut input_size
        ))?;
        Ok(input_size)
    }

    /// Get the single input port of the compiled model,
    /// which only support single input model.
    pub fn input(&self) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_input(
            self.instance,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Get an input port of the compiled model by port index.
    pub fn input_by_index(&self, index: usize) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_input_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Get an input port of the compiled model by name.
    pub fn input_by_name(&self, name: &str) -> Result<Port> {
        let name = cstr!(name);
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_input_by_name(
            self.instance,
            name.as_ptr(),
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Get the output size of the compiled model.
    pub fn output_size(&self) -> Result<usize> {
        let mut output_size: usize = 0;
        try_unsafe!(ov_compiled_model_outputs_size(
            self.instance,
            &mut output_size
        ))?;
        Ok(output_size)
    }

    /// Get the single output port of the compiled model,
    /// which only support single output model.
    pub fn output(&self) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_output(
            self.instance,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Get an output port of the compiled model by port index.
    pub fn output_by_index(&self, index: usize) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_output_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Get an output port of the compiled model by name.
    pub fn output_by_name(&self, name: &str) -> Result<Port> {
        let name = cstr!(name);
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_output_by_name(
            self.instance,
            name.as_ptr(),
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Gets runtime model information from a device.
    pub fn runtime_model(&self) -> Result<Model> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_get_runtime_model(
            self.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(Model { instance })
    }

    /// Create an [`InferRequest`].
    pub fn create_infer_request(&mut self) -> Result<InferRequest> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_create_infer_request(
            self.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(InferRequest { instance })
    }

    /// Gets a property for the compiled model.
    pub fn property(&self, key: PropertyKey) -> Result<Cow<str>> {
        let ov_prop_key = cstr!(key.as_ref());
        let mut ov_prop_value = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_get_property(
            self.instance,
            ov_prop_key.as_ptr(),
            std::ptr::addr_of_mut!(ov_prop_value)
        ))?;
        let rust_prop = unsafe { CStr::from_ptr(ov_prop_value) }.to_string_lossy();
        Ok(rust_prop)
    }

    /// Sets a property for the compiled model.
    pub fn set_property(&self, key: RwPropertyKey, value: &str) -> Result<()> {
        let ov_prop_key = cstr!(key.as_ref());
        let ov_prop_value = cstr!(value);
        try_unsafe!(ov_compiled_model_set_property(
            self.instance,
            ov_prop_key.as_ptr(),
            ov_prop_value.as_ptr(),
        ))?;
        Ok(())
    }

    /// Exports the current compiled model to a file.
    pub fn export(&self, to: impl AsRef<Path>) -> std::result::Result<(), IOError> {
        let model_path = cstr!(to.as_ref().to_str().ok_or(PathError::CannotStringify)?);
        try_unsafe!(ov_compiled_model_export_model(
            self.instance,
            model_path.as_ptr()
        ))?;
        Ok(())
    }
}
