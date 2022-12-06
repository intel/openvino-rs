use crate::{Layout, Precision};
use openvino_sys::{dimensions_t, tensor_desc_t};

/// See
/// [`TensorDesc`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1TensorDesc.html).
pub struct TensorDesc {
    pub(crate) instance: tensor_desc_t,
}

impl TensorDesc {
    /// Construct a new [`TensorDesc`] from its C API components.
    ///
    /// # Panics
    ///
    /// Only (currently) handles up to eight dimensions; will panic if exceeded.
    pub fn new(layout: Layout, dimensions: &[usize], precision: Precision) -> Self {
        // Setup dimensions.
        assert!(dimensions.len() < 8);
        let mut dims = [0; 8];
        dims[..dimensions.len()].copy_from_slice(dimensions);

        // Create the description structure.
        Self {
            instance: tensor_desc_t {
                layout,
                dims: dimensions_t {
                    ranks: dimensions.len(),
                    dims,
                },
                precision,
            },
        }
    }

    /// Layout of the tensor.
    pub fn layout(&self) -> Layout {
        self.instance.layout
    }

    /// Dimensions of the tensor.
    ///
    /// Length of the slice is equal to the tensor rank.
    pub fn dims(&self) -> &[usize] {
        &self.instance.dims.dims[..self.instance.dims.ranks]
    }

    /// Precision of the tensor.
    pub fn precision(&self) -> Precision {
        self.instance.precision
    }

    /// Get the number of elements described by this [`TensorDesc`].
    pub fn len(&self) -> usize {
        self.dims().iter().fold(1, |a, &b| a * b as usize)
    }
}
