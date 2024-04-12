use crate::cstr;
use std::borrow::Cow;
use std::convert::Infallible;
use std::ffi::CString;
use std::str::FromStr;

/// `DeviceType` represents accelerator devices.
#[derive(Debug)]
pub enum DeviceType<'a> {
    /// [CPU Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)
    CPU,
    /// [GPU Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)
    GPU,
    /// [NPU Device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
    NPU,
    /// [GNA Device](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_supported_plugins_GNA.html)
    #[deprecated = "Deprecated since OpenVINO 2024.0; use NPU device instead"]
    GNA,
    /// Arbitrary device.
    Other(Cow<'a, str>),
}

impl DeviceType<'_> {
    /// Creates a device type with owned string data.
    pub fn to_owned(&self) -> DeviceType<'static> {
        match self {
            DeviceType::CPU => DeviceType::CPU,
            DeviceType::GPU => DeviceType::GPU,
            DeviceType::NPU => DeviceType::NPU,
            #[allow(deprecated)]
            DeviceType::GNA => DeviceType::GNA,
            DeviceType::Other(s) => DeviceType::Other(Cow::Owned(s.clone().into_owned())),
        }
    }
}

impl<'a> From<&'a DeviceType<'a>> for &'a str {
    fn from(value: &'a DeviceType) -> Self {
        match value {
            DeviceType::CPU => "CPU",
            DeviceType::GPU => "GPU",
            DeviceType::NPU => "NPU",
            #[allow(deprecated)]
            DeviceType::GNA => "GNA",
            DeviceType::Other(s) => s,
        }
    }
}

impl<'a> From<DeviceType<'a>> for CString {
    fn from(value: DeviceType) -> Self {
        let device: &str = (&value).into();
        cstr!(device)
    }
}

impl<'a> From<&'a str> for DeviceType<'a> {
    fn from(s: &'a str) -> Self {
        match s {
            "CPU" => DeviceType::CPU,
            "GPU" => DeviceType::GPU,
            "NPU" => DeviceType::NPU,
            #[allow(deprecated)]
            "GNA" => DeviceType::GNA,
            s => DeviceType::Other(Cow::Borrowed(s)),
        }
    }
}

impl FromStr for DeviceType<'static> {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(DeviceType::from(s).to_owned())
    }
}
