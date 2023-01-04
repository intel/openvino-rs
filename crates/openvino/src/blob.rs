use crate::tensor_desc::TensorDesc;
use crate::{drop_using_function, try_unsafe, util::Result, InferenceError};
use openvino_sys::{
    self, ie_blob_buffer__bindgen_ty_1, ie_blob_buffer_t, ie_blob_byte_size, ie_blob_free,
    ie_blob_get_buffer, ie_blob_get_dims, ie_blob_get_layout, ie_blob_get_precision,
    ie_blob_make_memory, ie_blob_size, ie_blob_t, tensor_desc_t,
};
use std::convert::TryFrom;
use std::mem::MaybeUninit;

/// See [`Blob`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Blob.html).
pub struct Blob {
    pub(crate) instance: *mut ie_blob_t,
}
drop_using_function!(Blob, ie_blob_free);

impl Blob {
    /// Create a new [`Blob`] by copying data in to the OpenVINO-allocated memory.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bytes passed in `data` does not match the expected
    /// size of the tensor `description`.
    pub fn new(description: &TensorDesc, data: &[u8]) -> Result<Self> {
        let mut blob = Self::allocate(description)?;
        let blob_len = blob.byte_len()?;
        assert_eq!(
            blob_len,
            data.len(),
            "The data to initialize ({} bytes) must be the same as the blob size ({} bytes).",
            data.len(),
            blob_len
        );

        // Copy the incoming data into the buffer.
        let buffer = blob.buffer_mut()?;
        buffer.copy_from_slice(data);

        Ok(blob)
    }

    /// Allocate space in OpenVINO for an empty [`Blob`].
    pub fn allocate(description: &TensorDesc) -> Result<Self> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_blob_make_memory(
            std::ptr::addr_of!(description.instance),
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(Self { instance })
    }

    /// Return the tensor description of this [`Blob`].
    pub fn tensor_desc(&self) -> Result<TensorDesc> {
        let blob = self.instance as *const ie_blob_t;

        let mut layout = MaybeUninit::uninit();
        try_unsafe!(ie_blob_get_layout(blob, layout.as_mut_ptr()))?;

        let mut dimensions = MaybeUninit::uninit();
        try_unsafe!(ie_blob_get_dims(blob, dimensions.as_mut_ptr()))?;
        // Safety: this assertion is trying to avoid the improbable case where some future version
        // of the OpenVINO library returns a dimensions array with size different than the one
        // auto-generated in the bindings; see `struct dimensions` in
        // `openvino-sys/src/generated/types.rs`. It is not clear to me whether this will return the
        // statically-expected size or the dynamic size -- this is not effective in the former case.
        assert_eq!(unsafe { dimensions.assume_init() }.dims.len(), 8);

        let mut precision = MaybeUninit::uninit();
        try_unsafe!(ie_blob_get_precision(blob, precision.as_mut_ptr()))?;

        Ok(TensorDesc {
            // Safety: all reads succeeded so values must be initialized
            instance: unsafe {
                tensor_desc_t {
                    layout: layout.assume_init(),
                    dims: dimensions.assume_init(),
                    precision: precision.assume_init(),
                }
            },
        })
    }

    /// Get the number of elements contained in the [`Blob`].
    ///
    /// # Panics
    ///
    /// Panics if the returned OpenVINO size will not fit in `usize`.
    pub fn len(&self) -> Result<usize> {
        let mut size = 0;
        try_unsafe!(ie_blob_size(self.instance, std::ptr::addr_of_mut!(size)))?;
        Ok(usize::try_from(size).unwrap())
    }

    /// Get the size of the current [`Blob`] in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the returned OpenVINO size will not fit in `usize`.
    pub fn byte_len(&self) -> Result<usize> {
        let mut size = 0;
        try_unsafe!(ie_blob_byte_size(
            self.instance,
            std::ptr::addr_of_mut!(size)
        ))?;
        Ok(usize::try_from(size).unwrap())
    }

    /// Retrieve the [`Blob`]'s data as an immutable slice of bytes.
    pub fn buffer(&self) -> Result<&[u8]> {
        let mut buffer = Blob::empty_buffer();
        try_unsafe!(ie_blob_get_buffer(
            self.instance,
            std::ptr::addr_of_mut!(buffer)
        ))?;
        let size = self.byte_len()?;
        let slice = unsafe {
            std::slice::from_raw_parts(buffer.__bindgen_anon_1.buffer as *const u8, size)
        };
        Ok(slice)
    }

    /// Retrieve the [`Blob`]'s data as a mutable slice of bytes.
    pub fn buffer_mut(&mut self) -> Result<&mut [u8]> {
        let mut buffer = Blob::empty_buffer();
        try_unsafe!(ie_blob_get_buffer(
            self.instance,
            std::ptr::addr_of_mut!(buffer)
        ))?;
        let size = self.byte_len()?;
        let slice = unsafe {
            std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer.cast::<u8>(), size)
        };
        Ok(slice)
    }

    /// Retrieve the [`Blob`]'s data as an immutable slice of type `T`.
    ///
    /// # Safety
    ///
    /// This function is `unsafe`, since the values of `T` may not have been properly initialized;
    /// however, this functionality is provided as an equivalent of what C/C++ users of OpenVINO
    /// currently do to access [`Blob`]s with, e.g., floating point values:
    /// `results.buffer_as_type::<f32>()`.
    pub unsafe fn buffer_as_type<T>(&self) -> Result<&[T]> {
        let mut buffer = Blob::empty_buffer();
        InferenceError::from(ie_blob_get_buffer(
            self.instance,
            std::ptr::addr_of_mut!(buffer),
        ))?;
        // This is very unsafe, but very convenient: by allowing users to specify T, they can
        // retrieve the buffer in whatever shape they prefer. But we must ensure that they cannot
        // read too many bytes, so we manually calculate the resulting slice `size`.
        let size = self.byte_len()? / std::mem::size_of::<T>();
        let slice = std::slice::from_raw_parts(buffer.__bindgen_anon_1.buffer.cast::<T>(), size);
        Ok(slice)
    }

    /// Retrieve the [`Blob`]'s data as a mutable slice of type `T`.
    ///
    /// # Safety
    ///
    /// This function is `unsafe`, since the values of `T` may not have been properly initialized;
    /// however, this functionality is provided as an equivalent of what C/C++ users of OpenVINO
    /// currently do to access [`Blob`]s with, e.g., floating point values:
    /// `results.buffer_mut_as_type::<f32>()`.
    pub unsafe fn buffer_mut_as_type<T>(&mut self) -> Result<&mut [T]> {
        let mut buffer = Blob::empty_buffer();
        InferenceError::from(ie_blob_get_buffer(
            self.instance,
            std::ptr::addr_of_mut!(buffer),
        ))?;
        // This is very unsafe, but very convenient: by allowing users to specify T, they can
        // retrieve the buffer in whatever shape they prefer. But we must ensure that they cannot
        // read too many bytes, so we manually calculate the resulting slice `size`.
        let size = self.byte_len()? / std::mem::size_of::<T>();
        let slice =
            std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer.cast::<T>(), size);
        Ok(slice)
    }

    /// Construct a Blob from its associated pointer.
    pub(crate) unsafe fn from_raw_pointer(instance: *mut ie_blob_t) -> Self {
        Self { instance }
    }

    fn empty_buffer() -> ie_blob_buffer_t {
        ie_blob_buffer_t {
            __bindgen_anon_1: ie_blob_buffer__bindgen_ty_1 {
                buffer: std::ptr::null_mut(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layout, Precision};

    #[test]
    #[should_panic]
    fn invalid_blob_size() {
        let desc = TensorDesc::new(Layout::NHWC, &[1, 2, 2, 2], Precision::U8);
        // Blob should be 1x2x2x2 = 8 bytes but we pass in 7 bytes:
        let _ = Blob::new(&desc, &[0; 7]).unwrap();
    }

    #[test]
    fn buffer_conversion() {
        // In order to ensure runtime-linked libraries are linked with, we must:
        openvino_sys::library::load().expect("unable to find an OpenVINO shared library");

        const LEN: usize = 200 * 100;
        let desc = TensorDesc::new(Layout::HW, &[200, 100], Precision::U16);

        // Provide a u8 slice to create a u16 blob (twice as many items).
        let mut blob = Blob::new(&desc, &[0; LEN * 2]).unwrap();

        assert_eq!(blob.len().unwrap(), LEN);
        assert_eq!(
            blob.byte_len().unwrap(),
            LEN * 2,
            "we should have twice as many bytes (u16 = u8 * 2)"
        );
        assert_eq!(
            blob.buffer().unwrap().len(),
            LEN * 2,
            "we should have twice as many items (u16 = u8 * 2)"
        );
        assert_eq!(
            unsafe { blob.buffer_mut_as_type::<f32>() }.unwrap().len(),
            LEN / 2,
            "we should have half as many items (u16 = f32 / 2)"
        );
    }

    #[test]
    fn tensor_desc() {
        openvino_sys::library::load().expect("unable to find an OpenVINO shared library");

        let desc = TensorDesc::new(Layout::NHWC, &[1, 2, 2, 2], Precision::U8);
        let blob = Blob::new(&desc, &[0; 8]).unwrap();
        let desc2 = blob.tensor_desc().unwrap();

        // Both TensorDesc's should be equal
        assert_eq!(desc.layout(), desc2.layout());
        assert_eq!(desc.dims(), desc2.dims());
        assert_eq!(desc.precision(), desc2.precision());
    }
}
