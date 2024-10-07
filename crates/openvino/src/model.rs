//! Contains the model representations in OpenVINO:
//!  - [`Model`] is the OpenVINO representation of a neural model
//!  - [`CompiledModel`] is the compiled representation of a [`CompiledModel`] for a device.

use crate::node::Node;
use crate::request::InferRequest;
use crate::{cstr, drop_using_function, try_unsafe, util::Result, PropertyKey, RwPropertyKey};
use openvino_sys::{
    ov_compiled_model_create_infer_request, ov_compiled_model_free, ov_compiled_model_get_property,
    ov_compiled_model_get_runtime_model, ov_compiled_model_input, ov_compiled_model_input_by_index,
    ov_compiled_model_input_by_name, ov_compiled_model_inputs_size, ov_compiled_model_output,
    ov_compiled_model_output_by_index, ov_compiled_model_output_by_name,
    ov_compiled_model_outputs_size, ov_compiled_model_set_property, ov_compiled_model_t,
    ov_model_const_input_by_index, ov_model_const_output_by_index, ov_model_free,
    ov_model_inputs_size, ov_model_is_dynamic, ov_model_outputs_size, ov_model_t,
};
use std::borrow::Cow;
use std::ffi::CStr;

/// See [`ov_model_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__model__c__api.html).
pub struct Model {
    ptr: *mut ov_model_t,
}
drop_using_function!(Model, ov_model_free);

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Create a new [`Model`] from an internal pointer.
    #[inline]
    pub(crate) fn from_ptr(ptr: *mut ov_model_t) -> Self {
        Self { ptr }
    }

    /// Get the pointer to the underlying [`ov_model_t`].
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const ov_model_t {
        self.ptr
    }

    /// Retrieve the number of model inputs.
    pub fn get_inputs_len(&self) -> Result<usize> {
        let mut num: usize = 0;
        try_unsafe!(ov_model_inputs_size(self.ptr, &mut num))?;
        Ok(num)
    }

    /// Retrieve the number of model outputs.
    pub fn get_outputs_len(&self) -> Result<usize> {
        let mut num: usize = 0;
        try_unsafe!(ov_model_outputs_size(self.ptr, &mut num))?;
        Ok(num)
    }

    /// Retrieve the input node by index.
    pub fn get_input_by_index(&self, index: usize) -> Result<Node> {
        let mut node = std::ptr::null_mut();
        try_unsafe!(ov_model_const_input_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(node)
        ))?;
        Ok(Node::from_ptr(node))
    }

    /// Retrieve the output node by index.
    pub fn get_output_by_index(&self, index: usize) -> Result<Node> {
        let mut node = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(node)
        ))?;
        Ok(Node::from_ptr(node))
    }

    /// Retrieve the constant output node by index.
    pub fn get_const_output_by_index(&self, index: usize) -> Result<Node> {
        let mut node = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(node)
        ))?;
        Ok(Node::from_ptr(node))
    }

    /// Returns `true` if the model contains dynamic shapes.
    pub fn is_dynamic(&self) -> bool {
        unsafe { ov_model_is_dynamic(self.ptr) }
    }
}

/// See
/// [`ov_compiled_model_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__compiled__model__c__api.html).
pub struct CompiledModel {
    ptr: *mut ov_compiled_model_t,
}
drop_using_function!(CompiledModel, ov_compiled_model_free);

unsafe impl Send for CompiledModel {}

impl CompiledModel {
    /// Create a new [`CompiledModel`] from an internal `ov_compiled_model_t` pointer.
    pub(crate) fn from_ptr(ptr: *mut ov_compiled_model_t) -> Self {
        Self { ptr }
    }

    /// Create an [`InferRequest`].
    pub fn create_infer_request(&mut self) -> Result<InferRequest> {
        let mut infer_request = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_create_infer_request(
            self.ptr,
            std::ptr::addr_of_mut!(infer_request)
        ))?;
        Ok(InferRequest::from_ptr(infer_request))
    }

    /// Get the number of inputs of the compiled model.
    pub fn get_input_size(&self) -> Result<usize> {
        let mut input_size: usize = 0;
        try_unsafe!(ov_compiled_model_inputs_size(self.ptr, &mut input_size))?;
        Ok(input_size)
    }

    /// Get the single input port of the compiled model, which we expect to have only one input.
    pub fn get_input(&self) -> Result<Node> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_input(
            self.ptr,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Node::from_ptr(port))
    }

    /// Get an input port of the compiled model by port index.
    pub fn get_input_by_index(&self, index: usize) -> Result<Node> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_input_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Node::from_ptr(port))
    }

    /// Get an input port of the compiled model by name.
    pub fn get_input_by_name(&self, name: &str) -> Result<Node> {
        let name = cstr!(name);
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_input_by_name(
            self.ptr,
            name.as_ptr(),
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Node::from_ptr(port))
    }

    /// Get the number of outputs of the compiled model.
    pub fn get_output_size(&self) -> Result<usize> {
        let mut output_size: usize = 0;
        try_unsafe!(ov_compiled_model_outputs_size(self.ptr, &mut output_size))?;
        Ok(output_size)
    }

    /// Get the single output port of the compiled model, which we expect to have a single output.
    pub fn get_output(&self) -> Result<Node> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_output(
            self.ptr,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Node::from_ptr(port))
    }

    /// Get an output port of the compiled model by port index.
    pub fn get_output_by_index(&self, index: usize) -> Result<Node> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_output_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Node::from_ptr(port))
    }

    /// Get an output port of the compiled model by name.
    pub fn get_output_by_name(&self, name: &str) -> Result<Node> {
        let name = cstr!(name);
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_output_by_name(
            self.ptr,
            name.as_ptr(),
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Node::from_ptr(port))
    }

    /// Gets runtime model information from a device.
    pub fn get_runtime_model(&self) -> Result<Model> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_get_runtime_model(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Model { ptr })
    }

    /// Gets a property for the compiled model.
    pub fn get_property(&self, key: &PropertyKey) -> Result<Cow<str>> {
        let ov_prop_key = cstr!(key.as_ref());
        let mut ov_prop_value = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_get_property(
            self.ptr,
            ov_prop_key.as_ptr(),
            std::ptr::addr_of_mut!(ov_prop_value)
        ))?;
        let rust_prop = unsafe { CStr::from_ptr(ov_prop_value) }.to_string_lossy();
        Ok(rust_prop)
    }

    /// Sets a property for the compiled model.
    pub fn set_property(&mut self, key: &RwPropertyKey, value: &str) -> Result<()> {
        let ov_prop_key = cstr!(key.as_ref());
        let ov_prop_value = cstr!(value);
        try_unsafe!(ov_compiled_model_set_property(
            self.ptr,
            ov_prop_key.as_ptr(),
            ov_prop_value.as_ptr(),
        ))?;
        Ok(())
    }
}
