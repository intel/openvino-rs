use crate::blob::Blob;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    ie_infer_request_free, ie_infer_request_get_blob, ie_infer_request_infer,
    ie_infer_request_set_batch, ie_infer_request_set_blob, ie_infer_request_t,
    ie_infer_request_infer_async, ie_infer_request_wait,
};

/// See
/// [`InferRequest`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1InferRequest.html).
pub struct InferRequest {
    pub(crate) instance: *mut ie_infer_request_t,
}
drop_using_function!(InferRequest, ie_infer_request_free);

unsafe impl Send for InferRequest {}
unsafe impl Sync for InferRequest {}

impl InferRequest {
    /// Set the batch size of the inference requests.
    pub fn set_batch_size(&mut self, size: usize) -> Result<()> {
        try_unsafe!(ie_infer_request_set_batch(self.instance, size))
    }

    /// Assign a [Blob] to the input (i.e. `name`) on the network.
    pub fn set_blob(&mut self, name: &str, blob: &Blob) -> Result<()> {
        try_unsafe!(ie_infer_request_set_blob(
            self.instance,
            cstr!(name),
            blob.instance
        ))
    }

    /// Retrieve a [Blob] from the output (i.e. `name`) on the network.
    pub fn get_blob(&mut self, name: &str) -> Result<Blob> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_infer_request_get_blob(
            self.instance,
            cstr!(name),
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(unsafe { Blob::from_raw_pointer(instance) })
    }

    /// Execute the inference request.
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ie_infer_request_infer(self.instance))
    }

    /// Execute the inference request asyncroneously.
    pub fn infer_async(&mut self) -> Result<()> {
        try_unsafe!(ie_infer_request_infer_async(self.instance))
    }

    /// Wait for the result of the inference asyncroneous request.
    pub fn wait(&mut self, timeout: i64) -> Result<()> {
        try_unsafe!(ie_infer_request_wait(self.instance, timeout))
    }
}
