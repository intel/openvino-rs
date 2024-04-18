use crate::tensor::Tensor;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    ov_callback_t, ov_infer_request_cancel, ov_infer_request_free,
    ov_infer_request_get_input_tensor, ov_infer_request_get_input_tensor_by_index,
    ov_infer_request_get_output_tensor, ov_infer_request_get_output_tensor_by_index,
    ov_infer_request_get_tensor, ov_infer_request_infer, ov_infer_request_set_callback,
    ov_infer_request_set_input_tensor, ov_infer_request_set_input_tensor_by_index,
    ov_infer_request_set_output_tensor, ov_infer_request_set_output_tensor_by_index,
    ov_infer_request_set_tensor, ov_infer_request_start_async, ov_infer_request_t,
    ov_infer_request_wait, ov_infer_request_wait_for,
};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::time::Duration;

/// See [`InferRequest`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__infer__request__c__api.html).
pub struct InferRequest {
    pub(crate) instance: *mut ov_infer_request_t,
}
drop_using_function!(InferRequest, ov_infer_request_free);

unsafe impl Send for InferRequest {}
unsafe impl Sync for InferRequest {}

impl InferRequest {
    // Tensors

    /// Retrieve a [Tensor] from the input or output of the model.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let name = cstr!(name);
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_tensor(
            self.instance,
            name.as_ptr(),
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor { instance: tensor })
    }

    /// Assign a [Tensor] to the input on the model.
    pub fn set_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        let name = cstr!(name);
        try_unsafe!(ov_infer_request_set_tensor(
            self.instance,
            name.as_ptr(),
            tensor.instance
        ))
    }

    // Input tensors

    /// Get an input tensor from the model with only one input tensor.
    pub fn input_tensor(&self) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_input_tensor(
            self.instance,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor { instance: tensor })
    }

    /// Set an input tensor for infer models with single input.
    pub fn set_input_tensor(&mut self, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_input_tensor(
            self.instance,
            tensor.instance
        ))
    }

    /// Get an input tensor by its index.
    pub fn input_tensor_by_index(&self, index: usize) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_input_tensor_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor { instance: tensor })
    }

    /// Set an input tensor to infer on by the index of tensor.
    pub fn set_input_tensor_by_index(&mut self, index: usize, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_input_tensor_by_index(
            self.instance,
            index,
            tensor.instance
        ))
    }

    // Output tensors

    /// Get an output tensor from the model with only one output tensor.
    pub fn output_tensor(&self) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_output_tensor(
            self.instance,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor { instance: tensor })
    }

    /// Set an output tensor to infer models with single output.
    pub fn set_output_tensor(&mut self, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_output_tensor(
            self.instance,
            tensor.instance
        ))
    }

    /// Get an output tensor by its index.
    pub fn output_tensor_by_index(&self, index: usize) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_output_tensor_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor { instance: tensor })
    }

    /// Set an output tensor to infer by the index of output tensor.
    pub fn set_output_tensor_by_index(&mut self, index: usize, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_infer_request_set_output_tensor_by_index(
            self.instance,
            index,
            tensor.instance
        ))
    }

    // Inference

    /// Execute the inference request synchronously (blocking).
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_infer(self.instance))
    }

    /// Starts executing the inference request asynchronously (non-blocking).
    pub fn start_async(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_start_async(self.instance))
    }

    /// Waits indefinitely for the result to become available.
    /// Blocks until the result becomes available.
    pub fn wait(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_wait(self.instance))
    }

    /// Waits for the result to become available.
    /// Blocks until the specified timeout has elapsed or the result becomes available,
    /// whichever comes first.
    pub fn wait_for(&mut self, timeout: Duration) -> Result<()> {
        let timeout_ms = timeout.as_millis() as i64;
        try_unsafe!(ov_infer_request_wait_for(self.instance, timeout_ms))
    }

    /// Sets a callback function that is called on success or failure of an asynchronous request.
    ///
    /// If the returned handle is dropped, the callback will be automatically cleared
    /// from the infer request. Therefore, the handle MUST be kept in memory as long as it's desired
    /// to receive callbacks.
    pub fn set_callback<'a, F>(
        &'a mut self,
        callback: &'a mut F,
    ) -> Result<InferRequestCallbackHandle<'a, F>>
    where
        F: FnMut() + 'a,
    {
        let ov_callback = Box::new(ov_callback_t {
            callback_func: Some(trampoline::<F>),
            args: callback as *mut F as *mut c_void,
        });
        let ov_callback_ptr: *const ov_callback_t = &*ov_callback;
        try_unsafe!(ov_infer_request_set_callback(
            self.instance,
            ov_callback_ptr
        ))?;
        Ok(InferRequestCallbackHandle {
            boo: PhantomData,
            request: self.instance,
            _callback: ov_callback,
        })
    }

    /// Cancels inference request.
    pub fn cancel(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_cancel(self.instance))
    }
}

/// Type-safe closure for infer request async callbacks.
pub struct InferRequestCallbackHandle<'a, F>
where
    F: FnMut() + 'a,
{
    boo: PhantomData<&'a mut F>,
    request: *mut ov_infer_request_t,
    _callback: Box<ov_callback_t>,
}

unsafe impl<F> Send for InferRequestCallbackHandle<'_, F> where F: FnMut() {}

impl<F> Drop for InferRequestCallbackHandle<'_, F>
where
    F: FnMut(),
{
    fn drop(&mut self) {
        // If callback handle is dropped, automatically clear it from infer request
        unsafe { ov_infer_request_set_callback(self.request, std::ptr::addr_of!(NOOP.callback)) };
    }
}

/// A trampoline C/FFI function to launch into the Rust closure
unsafe extern "C" fn trampoline<F>(user_data: *mut c_void)
where
    F: FnMut(),
{
    let user_data = &mut *(user_data as *mut F);
    user_data();
}

/// An empty C/FFI function used to clear the callback for an infer request
unsafe extern "C" fn noop(_: *mut c_void) {}

struct Noop {
    callback: ov_callback_t,
}

unsafe impl Send for Noop {}
unsafe impl Sync for Noop {}

static NOOP: Noop = Noop {
    callback: ov_callback_t {
        callback_func: Some(noop),
        args: std::ptr::null_mut(),
    },
};
