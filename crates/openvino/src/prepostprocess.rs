//! See
//! [`ov_prepostprocess_c_api`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__prepostprocess__c__api.html#).
//!
//! For more information, read through the [`PrePostProcess
//! Walkthrough`](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Preprocessing_Overview.html).
//!
//! ```
//! # use openvino::{prepostprocess, Core, ElementType, Layout, Shape, Tensor, ResizeAlgorithm};
//! # use std::fs;
//! # let mut core = Core::new().expect("to instantiate the OpenVINO library");
//! # let mut model = core.read_model_from_file(
//! #     &"tests/fixtures/inception/inception.xml",
//! #     &"tests/fixtures/inception/inception.bin",
//! # ).expect("to read the model from file");
//! # let data = fs::read("tests/fixtures/inception/tensor-1x3x299x299-f32.bgr").expect("to read the tensor from file");
//! # let input_shape = Shape::new(&vec![1, 299, 299, 3]).expect("to create a new shape");
//! # let mut tensor = Tensor::new(ElementType::F32, &input_shape).expect("to create a new tensor");
//! # let buffer = tensor.get_raw_data_mut().unwrap();
//! # buffer.copy_from_slice(&data);
//! // Instantiate a new core, read in a model, and set up a tensor with input data before performing pre/post processing
//! // Pre-process the input by:
//! // - converting NHWC to NCHW
//! // - resizing the input image
//! let mut pipeline = prepostprocess::Pipeline::new(&model).expect("to create a new pipeline");
//! let input_info = pipeline.get_input_info_by_name("input").expect("to get input info by name");
//! let mut input_tensor_info = input_info.get_tensor_info().expect("to get tensor info");
//! input_tensor_info.set_from(&tensor).expect("to set tensor from");
//! input_tensor_info.set_layout(Layout::new("NHWC").expect("to create a new layout")).expect("to set layout");
//! let mut preprocess_steps = input_info.get_steps().expect("to get preprocess steps");
//! preprocess_steps.resize(ResizeAlgorithm::Linear).expect("to resize");
//! let mut model_info = input_info.get_model_info().expect("to get model info");
//! model_info.set_layout(Layout::new("NCHW").expect("to create a new layout")).expect("to set layout");
//! let new_model = pipeline.build_new_model().expect("to build new model with above prepostprocess steps");
//! ```
use crate::{
    cstr, drop_using_function, layout::Layout, try_unsafe, util::Result, ElementType, Model,
    ResizeAlgorithm, Tensor,
};
use openvino_sys::{
    ov_preprocess_input_info_free, ov_preprocess_input_info_get_model_info,
    ov_preprocess_input_info_get_preprocess_steps, ov_preprocess_input_info_get_tensor_info,
    ov_preprocess_input_info_t, ov_preprocess_input_model_info_free,
    ov_preprocess_input_model_info_set_layout, ov_preprocess_input_model_info_t,
    ov_preprocess_input_tensor_info_free, ov_preprocess_input_tensor_info_set_from,
    ov_preprocess_input_tensor_info_set_layout, ov_preprocess_input_tensor_info_t,
    ov_preprocess_output_info_free, ov_preprocess_output_info_get_tensor_info,
    ov_preprocess_output_info_t, ov_preprocess_output_set_element_type,
    ov_preprocess_output_tensor_info_free, ov_preprocess_output_tensor_info_t,
    ov_preprocess_prepostprocessor_build, ov_preprocess_prepostprocessor_create,
    ov_preprocess_prepostprocessor_free, ov_preprocess_prepostprocessor_get_input_info,
    ov_preprocess_prepostprocessor_get_input_info_by_index,
    ov_preprocess_prepostprocessor_get_input_info_by_name,
    ov_preprocess_prepostprocessor_get_output_info_by_index,
    ov_preprocess_prepostprocessor_get_output_info_by_name, ov_preprocess_prepostprocessor_t,
    ov_preprocess_preprocess_steps_convert_element_type,
    ov_preprocess_preprocess_steps_convert_layout, ov_preprocess_preprocess_steps_free,
    ov_preprocess_preprocess_steps_resize, ov_preprocess_preprocess_steps_t,
};

/// See
/// [`ov_preprocess_prepostprocessor_t`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__preprocess__prepostprocessor__t.html).
#[derive(Debug)]
pub struct Pipeline {
    ptr: *mut ov_preprocess_prepostprocessor_t,
}
drop_using_function!(Pipeline, ov_preprocess_prepostprocessor_free);
impl Pipeline {
    /// Creates a new [`Pipeline`] for the given [`Model`].
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_create(
            model.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Self { ptr })
    }

    /// Retrieves the input information by index.
    pub fn get_input_info_by_index(&self, index: usize) -> Result<InputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(ptr)
        ))?;

        Ok(InputInfo { ptr })
    }

    /// Retrieves the input information by name.
    pub fn get_input_info_by_name(&self, name: &str) -> Result<InputInfo> {
        let name = cstr!(name);
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_name(
            self.ptr,
            name.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;

        Ok(InputInfo { ptr })
    }

    /// Retrieves the output information by name.
    pub fn get_output_info_by_name(&self, name: &str) -> Result<OutputInfo> {
        let name = cstr!(name);
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_name(
            self.ptr,
            name.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(OutputInfo { ptr })
    }

    /// Retrieves the output information by index.
    pub fn get_output_info_by_index(&self, index: usize) -> Result<OutputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(ptr)
        ))?;

        Ok(OutputInfo { ptr })
    }

    /// Retrieves the input information.
    ///
    /// # Panics
    ///
    /// Panics if the returned input info is null.
    pub fn get_input_info(&self) -> Result<InputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        assert!(!ptr.is_null());
        Ok(InputInfo { ptr })
    }

    /// Builds a new model with all steps from pre/postprocessing.
    pub fn build_new_model(&self) -> Result<Model> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_build(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Model::from_ptr(ptr))
    }
}

/// See
/// [`ov_preprocess_input_info_t`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__preprocess__input__info__t.html).
pub struct InputInfo {
    ptr: *mut ov_preprocess_input_info_t,
}
drop_using_function!(InputInfo, ov_preprocess_input_info_free);

impl InputInfo {
    /// Retrieves the preprocessing model input information.
    pub fn get_model_info(&self) -> Result<InputModelInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_model_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(InputModelInfo { ptr })
    }

    /// Retrieves the input tensor information.
    pub fn get_tensor_info(&self) -> Result<InputTensorInfo> {
        let mut ptr: *mut ov_preprocess_input_tensor_info_t = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_tensor_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(InputTensorInfo { ptr })
    }

    /// Retrieves the preprocessing steps.
    pub fn get_steps(&self) -> Result<Steps> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_preprocess_steps(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Steps { ptr })
    }
}

/// See
/// [`ov_preprocess_output_info_t`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__preprocess__output__info__t.html).
pub struct OutputInfo {
    ptr: *mut ov_preprocess_output_info_t,
}
drop_using_function!(OutputInfo, ov_preprocess_output_info_free);
impl OutputInfo {
    /// Retrieves preprocess output tensor information.
    pub fn get_tensor_info(&self) -> Result<OutputTensorInfo> {
        let mut ptr: *mut ov_preprocess_output_tensor_info_t = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_output_info_get_tensor_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(OutputTensorInfo { ptr })
    }
}

/// See
/// [`ov_preprocess_input_model_info_t`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__preprocess__input__model__info__t.html).
pub struct InputModelInfo {
    ptr: *mut ov_preprocess_input_model_info_t,
}
drop_using_function!(InputModelInfo, ov_preprocess_input_model_info_free);
impl InputModelInfo {
    /// Sets the layout for the model information obj.
    pub fn set_layout(&mut self, mut layout: Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_input_model_info_set_layout(
            self.ptr,
            layout.as_mut_ptr()
        ))
    }
}

/// See
/// [`ov_preprocess_input_tensor_info_t`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__preprocess__input__tensor__info__t.html).
pub struct InputTensorInfo {
    ptr: *mut ov_preprocess_input_tensor_info_t,
}
drop_using_function!(InputTensorInfo, ov_preprocess_input_tensor_info_free);
impl InputTensorInfo {
    /// Sets the [`Layout`] for the input tensor.
    pub fn set_layout(&mut self, mut layout: Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_input_tensor_info_set_layout(
            self.ptr,
            layout.as_mut_ptr()
        ))
    }

    /// Sets the input tensor info from an existing tensor.
    pub fn set_from(&mut self, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_preprocess_input_tensor_info_set_from(
            self.ptr,
            tensor.as_ptr()
        ))
    }
}

/// See
/// [`ov_preprocess_output_tensor_info_t`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__preprocess__output__tensor__info__t.html).
pub struct OutputTensorInfo {
    ptr: *mut ov_preprocess_output_tensor_info_t,
}
drop_using_function!(OutputTensorInfo, ov_preprocess_output_tensor_info_free);
impl OutputTensorInfo {
    /// Sets the element type for output tensor info.
    pub fn set_element_type(&mut self, element_type: ElementType) -> Result<()> {
        try_unsafe!(ov_preprocess_output_set_element_type(
            self.ptr,
            element_type.into()
        ))
    }
}

/// See
/// [`ov_preprocess_preprocess_steps_t`](https://docs.openvino.ai/2024/api/c_cpp_api/structov__preprocess__preprocess__steps__t.html).
pub struct Steps {
    ptr: *mut ov_preprocess_preprocess_steps_t,
}
drop_using_function!(Steps, ov_preprocess_preprocess_steps_free);
impl Steps {
    /// Resizes the data in a [`Tensor`].
    pub fn resize(&mut self, resize_algo: ResizeAlgorithm) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_resize(
            self.ptr,
            resize_algo.into()
        ))
    }

    /// Converts the [`Layout`] of the data in a [`Tensor`].
    pub fn convert_layout(&mut self, mut new_layout: Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_convert_layout(
            self.ptr,
            new_layout.as_mut_ptr(),
        ))
    }

    /// Converts the element type of data in tensor.
    pub fn convert_element_type(&mut self, new_element_type: ElementType) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_convert_element_type(
            self.ptr,
            new_element_type.into()
        ))
    }
}
