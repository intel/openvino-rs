use crate::tensor::Tensor;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    ov_infer_request_free, ov_infer_request_get_tensor, ov_infer_request_infer,
    ov_infer_request_set_tensor, ov_infer_request_start_async, ov_infer_request_t,
    ov_infer_request_wait_for,
};

/// See [`InferRequest`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__infer__request__c__api.html).
pub struct InferRequest {
    instance: *mut ov_infer_request_t,
}
drop_using_function!(InferRequest, ov_infer_request_free);

unsafe impl Send for InferRequest {}
unsafe impl Sync for InferRequest {}

impl InferRequest {
    /// Create a new [`InferRequest`] from [`ov_infer_request_t`].
    pub(crate) fn new(instance: *mut ov_infer_request_t) -> Self {
        Self { instance }
    }
    /// Assign a [`Tensor`] to the input on the model.
    pub fn set_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_tensor(
            self.instance,
            cstr!(name),
            tensor.instance()
        ))?;
        Ok(())
    }

    /// Retrieve a [`Tensor`] from the output on the model.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_tensor(
            self.instance,
            cstr!(name),
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor::new_from_instance(tensor).unwrap())
    }

    /// Execute the inference request.
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_infer(self.instance))
    }

    /// Execute the inference request asyncroneously.
    pub fn infer_async(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_start_async(self.instance))
    }

    /// Wait for the result of the inference asyncroneous request.
    pub fn wait(&mut self, timeout: i64) -> Result<()> {
        try_unsafe!(ov_infer_request_wait_for(self.instance, timeout))
    }
}
