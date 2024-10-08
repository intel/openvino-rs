use crate::tensor::Tensor;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    ov_infer_request_cancel, ov_infer_request_free, ov_infer_request_get_input_tensor,
    ov_infer_request_get_output_tensor, ov_infer_request_get_output_tensor_by_index,
    ov_infer_request_get_tensor, ov_infer_request_infer, ov_infer_request_set_input_tensor,
    ov_infer_request_set_input_tensor_by_index, ov_infer_request_set_output_tensor,
    ov_infer_request_set_output_tensor_by_index, ov_infer_request_set_tensor,
    ov_infer_request_start_async, ov_infer_request_t, ov_infer_request_wait_for,
};

/// See
/// [`ov_infer_request_t`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__infer__request__c__api.html).
pub struct InferRequest {
    ptr: *mut ov_infer_request_t,
}
drop_using_function!(InferRequest, ov_infer_request_free);

unsafe impl Send for InferRequest {}
unsafe impl Sync for InferRequest {}

impl InferRequest {
    /// Create a new [`InferRequest`] from [`ov_infer_request_t`].
    #[inline]
    pub(crate) fn from_ptr(ptr: *mut ov_infer_request_t) -> Self {
        Self { ptr }
    }

    /// Assign a [`Tensor`] to the input on the model.
    pub fn set_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        let name = cstr!(name);
        try_unsafe!(ov_infer_request_set_tensor(
            self.ptr,
            name.as_ptr(),
            tensor.as_ptr()
        ))?;
        Ok(())
    }

    /// Retrieve a [`Tensor`] from the output on the model.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let name = cstr!(name);
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_tensor(
            self.ptr,
            name.as_ptr(),
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor::from_ptr(tensor))
    }

    /// Get an input tensor from the model with only one input tensor.
    pub fn get_input_tensor(&self) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_input_tensor(
            self.ptr,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor::from_ptr(tensor))
    }

    /// Set an input tensor for infer models with single input.
    pub fn set_input_tensor(&mut self, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_input_tensor(self.ptr, tensor.as_ptr()))
    }

    /// Assing an input [`Tensor`] to the model by its index.
    pub fn set_input_tensor_by_index(&mut self, index: usize, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_input_tensor_by_index(
            self.ptr,
            index,
            tensor.as_ptr()
        ))?;
        Ok(())
    }

    /// Retrieve an output [`Tensor`] from the model by its index.
    pub fn get_output_tensor_by_index(&self, index: usize) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_output_tensor_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor::from_ptr(tensor))
    }

    /// Get an output tensor from the model with only one output tensor.
    pub fn get_output_tensor(&self) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_output_tensor(
            self.ptr,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor::from_ptr(tensor))
    }

    /// Set an output tensor to infer models with single output.
    pub fn set_output_tensor(&mut self, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_output_tensor(
            self.ptr,
            tensor.as_ptr()
        ))
    }

    /// Set an output tensor to infer by the index of output tensor.
    pub fn set_output_tensor_by_index(&mut self, index: usize, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_output_tensor_by_index(
            self.ptr,
            index,
            tensor.as_ptr()
        ))
    }

    /// Execute the inference request.
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_infer(self.ptr))
    }

    /// Cancels inference request.
    pub fn cancel(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_cancel(self.ptr))
    }

    /// Execute the inference request asynchronously.
    pub fn infer_async(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_start_async(self.ptr))
    }

    /// Wait for the result of the inference asynchronous request.
    pub fn wait(&mut self, timeout: i64) -> Result<()> {
        try_unsafe!(ov_infer_request_wait_for(self.ptr, timeout))
    }
}
