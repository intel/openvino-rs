//! Contains the model representations in OpenVINO:
//!  - [`Model`] is the OpenVINO representation of a neural model
//!  - [`CompiledModel`] is the compiled representation of a [`CompiledModel`] for a device.

use crate::node::Node;
use crate::request::InferRequest;
use crate::{drop_using_function, try_unsafe, util::Result};

use openvino_sys::{
    ov_compiled_model_create_infer_request, ov_compiled_model_free, ov_compiled_model_t,
    ov_model_const_input_by_index, ov_model_const_output_by_index, ov_model_free,
    ov_model_inputs_size, ov_model_outputs_size, ov_model_t,
};

/// See [`Model`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__model__c__api.html).
pub struct Model {
    instance: *mut ov_model_t,
}
drop_using_function!(Model, ov_model_free);

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Create a new instance of the Model struct.
    pub(crate) fn new_from_instance(instance: *mut ov_model_t) -> Self {
        Self { instance }
    }

    /// Get the pointer to the underlying [`ov_model_t`].
    pub(crate) fn instance(&self) -> *mut ov_model_t {
        self.instance
    }

    /// Retrieve the number of model inputs.
    pub fn get_inputs_len(&self) -> Result<usize> {
        let mut num: usize = 0;
        try_unsafe!(ov_model_inputs_size(self.instance, &mut num))?;
        Ok(num)
    }

    /// Retrieve the number of model outputs.
    pub fn get_outputs_len(&self) -> Result<usize> {
        let mut num: usize = 0;
        try_unsafe!(ov_model_outputs_size(self.instance, &mut num))?;
        Ok(num)
    }

    /// Retrieve the input node by index.
    pub fn get_input_by_index(&self, index: usize) -> Result<Node> {
        let mut node = std::ptr::null_mut();
        try_unsafe!(ov_model_const_input_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(node)
        ))?;
        Ok(Node::new(node))
    }

    /// Retrieve the output node by index.
    pub fn get_output_by_index(&self, index: usize) -> Result<Node> {
        let mut node = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(node)
        ))?;
        Ok(Node::new(node))
    }

    /// Retrieve the constant output node by index.
    pub fn get_const_output_by_index(&self, index: usize) -> Result<Node> {
        let mut node = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(node)
        ))?;
        Ok(Node::new(node))
    }
}

/// See [`CompiledModel`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__compiled__model__c__api.html).
pub struct CompiledModel {
    instance: *mut ov_compiled_model_t,
}
drop_using_function!(CompiledModel, ov_compiled_model_free);

unsafe impl Send for CompiledModel {}

impl CompiledModel {
    /// Create a new instance of the CompiledModel struct from ov_compiled_model_t.
    pub(crate) fn new(instance: *mut ov_compiled_model_t) -> Self {
        Self { instance }
    }

    /// Create an [`InferRequest`].
    pub fn create_infer_request(&mut self) -> Result<InferRequest> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_create_infer_request(
            self.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(InferRequest::new(instance))
    }
}
