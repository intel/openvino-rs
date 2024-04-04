use crate::tensor::Tensor;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    ov_callback_t, ov_infer_request_cancel, ov_infer_request_free, ov_infer_request_get_tensor,
    ov_infer_request_infer, ov_infer_request_set_callback, ov_infer_request_set_tensor,
    ov_infer_request_start_async, ov_infer_request_t, ov_infer_request_wait,
    ov_infer_request_wait_for,
};
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

/// See [`InferRequest`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__infer__request__c__api.html).
pub struct InferRequest {
    pub(crate) instance: *mut ov_infer_request_t,
}
drop_using_function!(InferRequest, ov_infer_request_free);

unsafe impl Send for InferRequest {}
unsafe impl Sync for InferRequest {}

impl InferRequest {
    /// Retrieve a [Tensor] from the output on the model.
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

    /// Execute the inference request synchronously (blocking).
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_infer(self.instance))
    }

    /// Starts executing the inference request and asynchronously wait for the result.
    pub fn infer_async(&mut self) -> impl Future<Output = Result<()>> {
        let result = async {
            // Set callback
            let callback = InferRequestClosure::set_callback(self, || {
                // TODO notify future complete (how?)
            })?;
            self.start_async()
            // TODO what if an error happens here and the callback goes out of scope?
            // If the callback goes out of scope, the request needs to be cancelled or
            // the callback needs to be set to a callback with a static lifetime
        };
        InferAsync {}
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

    // /// Sets a callback function that is called on success or failure of an asynchronous request.
    // pub fn set_callback<'a, F>(&'a mut self, callback: F) -> Result<()>
    // where
    //     F: FnMut() + 'a,
    // {
    //     // TODO how to require closure F to live as long as 'self'?
    //     let callback = InferRequestCallback::new(callback);
    //     try_unsafe!(ov_infer_request_set_callback(
    //         self.instance,
    //         std::ptr::addr_of!(callback.instance)
    //     ))
    // }

    /// Cancels inference request.
    pub fn cancel(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_cancel(self.instance))
    }
}

/// Type-safe closure for infer request async callbacks.
struct InferRequestClosure<F>
where
    F: FnMut(),
{
    func: F,
}

impl<F> InferRequestClosure<F>
where
    F: FnMut(),
{
    pub fn set_callback(request: &mut InferRequest, mut func: F) -> Result<Self> {
        let callback = ov_callback_t {
            callback_func: Some(trampoline::<F>),
            args: &mut func as *mut _ as *mut c_void,
        };
        try_unsafe!(ov_infer_request_set_callback(
            request.instance,
            std::ptr::addr_of!(callback)
        ))?;
        Ok(Self { func })
    }
}

unsafe extern "C" fn trampoline<F>(user_data: *mut c_void)
where
    F: FnMut(),
{
    let user_data = &mut *(user_data as *mut F);
    user_data();
}

struct InferAsync {
    // TODO
}

impl Future for InferAsync {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}
