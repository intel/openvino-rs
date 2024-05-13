//! See [`PrePostProcess`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__prepostprocess__c__api.html#).
//!
//! See [`PrePostProcess Walkthrough`](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Preprocessing_Overview.html).
//!
//! ```
//!
//! # use openvino::{prepostprocess, Core, ElementType, Layout, Shape, Tensor, ResizeAlgorithm};
//! # use std::fs;
//! # let mut core = Core::new().expect("to instantiate the OpenVINO library");
//! # let mut model = core.read_model_from_file(
//! #     &"tests/fixtures/inception/inception.xml",
//! #     &"tests/fixtures/inception/inception.bin",
//! # ).expect("to read the model from file");
//! # let data = fs::read("tests/fixtures/inception/tensor-1x3x299x299-f32.bgr").expect("to read the tensor from file");
//! # let input_shape = Shape::new(&vec![1, 299, 299, 3]).expect("to create a new shape");
//! # let tensor = Tensor::new_from_host_ptr(ElementType::F32, &input_shape, &data).expect("to create a new tensor from host pointer");
//! // Insantiate a new core, read in a model, and set up a tensor with input data before performing pre/post processing
//! // Pre-process the input by:
//! // - converting NHWC to NCHW
//! // - resizing the input image
//! let mut pre_post_process = prepostprocess::PrePostProcess::new(&model).expect("to create a new PrePostProcess instance");
//! let input_info = pre_post_process.get_input_info_by_name("input").expect("to get input info by name");
//! let mut input_tensor_info = input_info.preprocess_input_info_get_tensor_info().expect("to get tensor info");
//! input_tensor_info.preprocess_input_tensor_set_from(&tensor).expect("to set tensor from");
//! input_tensor_info.preprocess_input_tensor_set_layout(&Layout::new("NHWC").expect("to create a new layout")).expect("to set layout");
//! let mut preprocess_steps = input_info.get_preprocess_steps().expect("to get preprocess steps");
//! preprocess_steps.preprocess_steps_resize(ResizeAlgorithm::Linear).expect("to resize");
//! let model_info = input_info.get_model_info().expect("to get model info");
//! model_info.model_info_set_layout(&Layout::new("NCHW").expect("to create a new layout")).expect("to set layout");
//! let new_model = pre_post_process.build_new_model().expect("to build new model with above prepostprocess steps");
//! ```
use crate::{
    cstr, drop_using_function, layout::Layout, try_unsafe, util::Result, ElementType, Model, ResizeAlgorithm, Tensor,
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

/// See [`PrePostProcess`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__prepostprocessor__t.html).
#[derive(Debug)]
pub struct PrePostProcess {
    ptr: *mut ov_preprocess_prepostprocessor_t,
}
drop_using_function!(PrePostProcess, ov_preprocess_prepostprocessor_free);

/// See [`PreProcessInputInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__input__info__t.html).
pub struct PreProcessInputInfo {
    ptr: *mut ov_preprocess_input_info_t,
}
drop_using_function!(PreProcessInputInfo, ov_preprocess_input_info_free);

/// See [`PreprocessOutputInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__output__info__t.html).
pub struct PreProcessOutputInfo {
    ptr: *mut ov_preprocess_output_info_t,
}
drop_using_function!(PreProcessOutputInfo, ov_preprocess_output_info_free);

/// See [`PreprocessSteps`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__preprocess__steps__t.html).
pub struct PreProcessSteps {
    ptr: *mut ov_preprocess_preprocess_steps_t,
}
drop_using_function!(PreProcessSteps, ov_preprocess_preprocess_steps_free);

/// See [`PreprocessInputModelInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__input__model__info__t.html).
pub struct PreProcessInputModelInfo {
    ptr: *mut ov_preprocess_input_model_info_t,
}
drop_using_function!(
    PreProcessInputModelInfo,
    ov_preprocess_input_model_info_free
);

/// See [`PreprocessInputTensorInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__input__tensor__info__t.html).
pub struct PreProcessInputTensorInfo {
    ptr: *mut ov_preprocess_input_tensor_info_t,
}
drop_using_function!(
    PreProcessInputTensorInfo,
    ov_preprocess_input_tensor_info_free
);

/// See [`PreprocessOutputTensorInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__output__tensor__info__t.html).
pub struct PreProcessOutputTensorInfo {
    ptr: *mut ov_preprocess_output_tensor_info_t,
}
drop_using_function!(
    PreProcessOutputTensorInfo,
    ov_preprocess_output_tensor_info_free
);

impl PreProcessInputModelInfo {
    /// Sets the layout for the model information obj.
    pub fn model_info_set_layout(&self, layout: &Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_input_model_info_set_layout(
            self.ptr,
            layout.as_ptr()
        ))
    }
}

impl PreProcessInputTensorInfo {
    /// Sets the layout for the input tensor.
    pub fn preprocess_input_tensor_set_layout(&self, layout: &Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_input_tensor_info_set_layout(
            self.ptr,
            layout.as_ptr()
        ))
    }

    /// Sets the input tensor info from an existing tensor.
    pub fn preprocess_input_tensor_set_from(&mut self, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_preprocess_input_tensor_info_set_from(
            self.ptr,
            tensor.as_ptr()
        ))
    }
}

impl PrePostProcess {
    /// Creates a new `PrePostProcess` pipeline for the given model.
    pub fn new(model: &Model) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_create(
            model.as_ptr(),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(Self { ptr })
    }

    /// Retrieves the input information by index.
    pub fn get_input_info_by_index(&self, index: usize) -> Result<PreProcessInputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(ptr)
        ))?;

        Ok(PreProcessInputInfo { ptr })
    }

    /// Retrieves the input information by name.
    pub fn get_input_info_by_name(&self, name: &str) -> Result<PreProcessInputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_name(
            self.ptr,
            cstr!(name),
            std::ptr::addr_of_mut!(ptr)
        ))?;

        Ok(PreProcessInputInfo { ptr })
    }

    /// Retrieves the output information by name.
    pub fn get_output_info_by_name(&self, name: &str) -> Result<PreProcessOutputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_name(
            self.ptr,
            cstr!(name),
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(PreProcessOutputInfo { ptr })
    }

    /// Retrieves the output information by index.
    pub fn get_output_info_by_index(&self, index: usize) -> Result<PreProcessOutputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_index(
            self.ptr,
            index,
            std::ptr::addr_of_mut!(ptr)
        ))?;

        Ok(PreProcessOutputInfo { ptr })
    }

    /// Retrieves the input information.
    ///
    /// # Panics
    ///
    /// Panics if the returned input info is null.
    pub fn get_input_info(&self) -> Result<PreProcessInputInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        assert!(!ptr.is_null());
        Ok(PreProcessInputInfo { ptr })
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

impl PreProcessSteps {
    /// Resizes data in tensor.
    pub fn preprocess_steps_resize(&mut self, resize_algo: ResizeAlgorithm) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_resize(self.ptr, resize_algo as u32,))?;

        Ok(())
    }

    /// Converts the layout of data in tensor.
    pub fn preprocess_convert_layout(&self, layout: &Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_convert_layout(
            self.ptr,
            layout.as_ptr(),
        ))?;

        Ok(())
    }

    /// Converts the element type of data in tensor.
    pub fn preprocess_convert_element_type(&self, element_type: ElementType) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_convert_element_type(
            self.ptr,
            element_type as u32
        ))?;

        Ok(())
    }
}

impl PreProcessOutputTensorInfo {
    /// Sets the element type for output tensor info.
    pub fn preprocess_set_element_type(&self, element_type: ElementType) -> Result<()> {
        try_unsafe!(ov_preprocess_output_set_element_type(
            self.ptr,
            element_type as u32
        ))
    }
}

impl PreProcessOutputInfo {
    /// Retrieves preprocess output tensor information.
    pub fn get_output_info_get_tensor_info(&self) -> Result<PreProcessOutputTensorInfo> {
        let mut ptr: *mut ov_preprocess_output_tensor_info_t = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_output_info_get_tensor_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(PreProcessOutputTensorInfo { ptr })
    }
}

impl PreProcessInputInfo {
    /// Retrieves the preprocessing model input information.
    pub fn get_model_info(&self) -> Result<PreProcessInputModelInfo> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_model_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(PreProcessInputModelInfo { ptr })
    }

    /// Retrieves the input tensor information.
    pub fn preprocess_input_info_get_tensor_info(&self) -> Result<PreProcessInputTensorInfo> {
        let mut ptr: *mut ov_preprocess_input_tensor_info_t = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_tensor_info(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(PreProcessInputTensorInfo { ptr })
    }

    /// Retrieves preprocessing steps object.
    pub fn get_preprocess_steps(&self) -> Result<PreProcessSteps> {
        let mut ptr = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_preprocess_steps(
            self.ptr,
            std::ptr::addr_of_mut!(ptr)
        ))?;
        Ok(PreProcessSteps { ptr })
    }
}
