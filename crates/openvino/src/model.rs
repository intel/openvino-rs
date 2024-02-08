//! Contains the model representations in OpenVINO:
//!  - [`CNNmodel`] is the OpenVINO representation of a neural model
//!  - [`Executablemodel`] is the compiled representation of a [`CNNmodel`] for a device.

use crate::port::Port;
use crate::request::InferRequest;
use crate::{drop_using_function, try_unsafe, util::Result};

use openvino_sys::{
    ov_compiled_model_create_infer_request, ov_compiled_model_free, ov_compiled_model_t,
    ov_model_const_input_by_index, ov_model_const_output_by_index, ov_model_free,
    ov_model_inputs_size, ov_model_outputs_size, ov_model_t,
};

/// See
/// [`CNNmodel`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1CNNmodel.html).

pub struct Model {
    pub(crate) instance: *mut ov_model_t,
}
drop_using_function!(Model, ov_model_free);

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Create a new instance of the Model struct.
    pub fn new() -> Result<Self> {
        let instance = std::ptr::null_mut();
        Ok(Self { instance })
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

    /// Retrieve the input port by index.
    pub fn get_input_by_index(&self, index: usize) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_model_const_input_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Retrieve the output port by index.
    pub fn get_output_by_index(&self, index: usize) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }

    /// Retrieve the constant output port by index.
    pub fn get_const_output_by_index(&self, index: usize) -> Result<Port> {
        let mut port = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(port)
        ))?;
        Ok(Port { instance: port })
    }
}

/// See
/// [`CompiledModel`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__compiled__model__c__api.html).
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
