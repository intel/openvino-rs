//! Contains the model representations in OpenVINO:
//!  - [`Model`] is the OpenVINO representation of a neural model
//!  - [`CompiledModel`] is the compiled representation of a [`Model`] for a device.

use crate::port::Port;
use crate::request::InferRequest;
use crate::{drop_using_function, try_unsafe, util::Result};

use openvino_sys::{
    ov_compiled_model_create_infer_request, ov_compiled_model_free, ov_compiled_model_t,
    ov_model_const_input_by_index, ov_model_const_output_by_index, ov_model_free,
    ov_model_inputs_size, ov_model_outputs_size, ov_model_t,
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

    /// Retrieve the number of model outputs.
    pub fn output_size(&self) -> Result<usize> {
        let mut output_size: usize = 0;
        try_unsafe!(ov_model_outputs_size(self.instance, &mut output_size))?;
        Ok(output_size)
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
    /// Create an [`InferRequest`].
    pub fn create_infer_request(&mut self) -> Result<InferRequest> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_create_infer_request(
            self.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(InferRequest { instance })
    }
}
