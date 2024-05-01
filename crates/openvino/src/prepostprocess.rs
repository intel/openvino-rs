//! See [`PrePostProcess`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__prepostprocess__c__api.html#).
//!
//! See [`PrePostProcess Walkthrough`](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Preprocessing_Overview.html).
//!
//! ```
//!
//! # use openvino::{prepostprocess, Core, ElementType, Layout, Shape, Tensor};
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
//! preprocess_steps.preprocess_steps_resize(0).expect("to resize");
//! let model_info = input_info.get_model_info().expect("to get model info");
//! model_info.model_info_set_layout(&Layout::new("NCHW").expect("to create a new layout")).expect("to set layout");
//! let new_model = pre_post_process.build_new_model().expect("to build new model with above prepostprocess steps");
//! ```
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
use std::ffi::CString;

use crate::{
    drop_using_function, layout::Layout, try_unsafe, util::Result, ElementType, Model, Tensor,
};

/// See [`PrePostProcess`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__prepostprocessor__t.html).
#[derive(Debug)]
pub struct PrePostProcess {
    instance: *mut ov_preprocess_prepostprocessor_t,
}
drop_using_function!(PrePostProcess, ov_preprocess_prepostprocessor_free);

/// See [`PreProcessInputInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__input__info__t.html).
pub struct PreProcessInputInfo {
    instance: *mut ov_preprocess_input_info_t,
}
drop_using_function!(PreProcessInputInfo, ov_preprocess_input_info_free);

/// See [`PreprocessOutputInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__output__info__t.html).
pub struct PreProcessOutputInfo {
    instance: *mut ov_preprocess_output_info_t,
}
drop_using_function!(PreProcessOutputInfo, ov_preprocess_output_info_free);

/// See [`PreprocessSteps`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__preprocess__steps__t.html).
pub struct PreProcessSteps {
    instance: *mut ov_preprocess_preprocess_steps_t,
}
drop_using_function!(PreProcessSteps, ov_preprocess_preprocess_steps_free);

/// See [`PreprocessInputModelInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__input__model__info__t.html).
pub struct PreProcessInputModelInfo {
    instance: *mut ov_preprocess_input_model_info_t,
}
drop_using_function!(
    PreProcessInputModelInfo,
    ov_preprocess_input_model_info_free
);

/// See [`PreprocessInputTensorInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__input__tensor__info__t.html).
pub struct PreProcessInputTensorInfo {
    instance: *mut ov_preprocess_input_tensor_info_t,
}
drop_using_function!(
    PreProcessInputTensorInfo,
    ov_preprocess_input_tensor_info_free
);

/// See [`PreprocessOutputTensorInfo`](https://docs.openvino.ai/2023.3/api/c_cpp_api/structov__preprocess__output__tensor__info__t.html).
pub struct PreProcessOutputTensorInfo {
    instance: *mut ov_preprocess_output_tensor_info_t,
}
drop_using_function!(
    PreProcessOutputTensorInfo,
    ov_preprocess_output_tensor_info_free
);

impl PreProcessInputModelInfo {
    /// Sets the layout for the model information obj.
    pub fn model_info_set_layout(&self, layout: &Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_input_model_info_set_layout(
            self.instance,
            layout.instance()
        ))?;

        Ok(())
    }
}

impl PreProcessInputTensorInfo {
    /// Sets the layout for the input tensor.
    pub fn preprocess_input_tensor_set_layout(&self, layout: &Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_input_tensor_info_set_layout(
            self.instance,
            layout.instance()
        ))?;

        Ok(())
    }

    /// Sets the input tensor info from an existing tensor.
    pub fn preprocess_input_tensor_set_from(&mut self, tensor: &Tensor) -> Result<()> {
        try_unsafe!(ov_preprocess_input_tensor_info_set_from(
            self.instance,
            tensor.instance()
        ))?;

        Ok(())
    }
}

impl PrePostProcess {
    /// Creates a new `PrePostprocess` instance for the given model.
    pub fn new(model: &Model) -> Result<Self> {
        let mut preprocess = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_create(
            model.instance(),
            std::ptr::addr_of_mut!(preprocess)
        ))?;

        Ok(Self {
            instance: preprocess,
        })
    }

    /// Retrieves the input information by index.
    pub fn get_input_info_by_index(&self, index: usize) -> Result<PreProcessInputInfo> {
        let mut input_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(input_info)
        ))?;

        Ok(PreProcessInputInfo {
            instance: input_info,
        })
    }

    /// Retrieves the input information by name.
    pub fn get_input_info_by_name(&self, name: &str) -> Result<PreProcessInputInfo> {
        let mut input_info = std::ptr::null_mut();
        let c_layout_desc = CString::new(name).unwrap();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_name(
            self.instance,
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(input_info)
        ))?;

        Ok(PreProcessInputInfo {
            instance: input_info,
        })
    }

    /// Retrieves the output information by name.
    pub fn get_output_info_by_name(&self, name: &str) -> Result<PreProcessOutputInfo> {
        let mut output_info = std::ptr::null_mut();
        let c_layout_desc = CString::new(name).unwrap();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_name(
            self.instance,
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(output_info)
        ))?;

        Ok(PreProcessOutputInfo {
            instance: output_info,
        })
    }

    /// Retrieves the output information by index.
    pub fn get_output_info_by_index(&self, index: usize) -> Result<PreProcessOutputInfo> {
        let mut output_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(output_info)
        ))?;

        Ok(PreProcessOutputInfo {
            instance: output_info,
        })
    }

    /// Retrieves the input information.
    pub fn get_input_info(&self) -> Result<PreProcessInputInfo> {
        let mut input_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info(
            self.instance,
            std::ptr::addr_of_mut!(input_info)
        ))?;
        assert!(!input_info.is_null());

        Ok(PreProcessInputInfo {
            instance: input_info,
        })
    }

    /// Builds a new model with all steps from pre/postprocessing.
    pub fn build_new_model(&self) -> Result<Model> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_build(
            self.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(Model::new_from_instance(instance))
    }
}

impl PreProcessSteps {
    /// Resizes data in tensor.
    pub fn preprocess_steps_resize(&mut self, resize_algo: u32) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_resize(
            self.instance,
            resize_algo,
        ))?;

        Ok(())
    }

    /// Converts the layout of data in tensor.
    pub fn preprocess_convert_layout(&self, layout: &Layout) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_convert_layout(
            self.instance,
            layout.instance(),
        ))?;

        Ok(())
    }

    /// Converts the element type of data in tensor.
    pub fn preprocess_convert_element_type(&self, element_type: ElementType) -> Result<()> {
        try_unsafe!(ov_preprocess_preprocess_steps_convert_element_type(
            self.instance,
            element_type as u32
        ))?;

        Ok(())
    }
}

impl PreProcessOutputTensorInfo {
    /// Sets the element type for output tensor info.
    pub fn preprocess_set_element_type(&self, element_type: ElementType) -> Result<()> {
        try_unsafe!(ov_preprocess_output_set_element_type(
            self.instance,
            element_type as u32
        ))?;

        Ok(())
    }
}

impl PreProcessOutputInfo {
    /// Retrieves preprocess output tensor information.
    pub fn get_output_info_get_tensor_info(&self) -> Result<PreProcessOutputTensorInfo> {
        let mut preprocess_output_tensor_info: *mut ov_preprocess_output_tensor_info_t =
            std::ptr::null_mut();
        try_unsafe!(ov_preprocess_output_info_get_tensor_info(
            self.instance,
            std::ptr::addr_of_mut!(preprocess_output_tensor_info)
        ))?;

        Ok(PreProcessOutputTensorInfo {
            instance: preprocess_output_tensor_info,
        })
    }
}

impl PreProcessInputInfo {
    /// Retrieves the preprocessing model input information.
    pub fn get_model_info(&self) -> Result<PreProcessInputModelInfo> {
        let mut model_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_model_info(
            self.instance,
            std::ptr::addr_of_mut!(model_info)
        ))?;

        Ok(PreProcessInputModelInfo {
            instance: model_info,
        })
    }

    /// Retrieves the input tensor information.
    pub fn preprocess_input_info_get_tensor_info(&self) -> Result<PreProcessInputTensorInfo> {
        let mut preprocess_input_tensor_info: *mut ov_preprocess_input_tensor_info_t =
            std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_tensor_info(
            self.instance,
            std::ptr::addr_of_mut!(preprocess_input_tensor_info)
        ))?;

        Ok(PreProcessInputTensorInfo {
            instance: preprocess_input_tensor_info,
        })
    }

    /// Retrieves preprocessing steps object.
    pub fn get_preprocess_steps(&self) -> Result<PreProcessSteps> {
        let mut preprocess_steps = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_preprocess_steps(
            self.instance,
            std::ptr::addr_of_mut!(preprocess_steps)
        ))?;

        Ok(PreProcessSteps {
            instance: preprocess_steps,
        })
    }
}
