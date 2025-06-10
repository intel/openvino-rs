use std::borrow::Cow;

/// See
/// [`ov_property_c_api`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__property__c__api.html).
/// `PropertyKey` represents valid configuration properties for a [`crate::Core`] instance.
#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub enum PropertyKey {
    /// A string list of supported read-only properties.
    SupportedProperties,
    /// A list of available device IDs.
    AvailableDevices,
    /// An unsigned integer value of optimal number of compiled model infer requests.
    OptimalNumberOfInferRequests,
    /// A hint for a range for number of async infer requests. If a device supports streams, the
    /// metric provides the range for number of IRs per stream.
    RangeForAsyncInferRequests,
    /// Information about a range for streams on platforms where streams are supported.
    RangeForStreams,
    /// A string value representing a full device name.
    DeviceFullName,
    /// A string list of capabilities options per device.
    DeviceCapabilities,
    /// The name of a model.
    ModelName,
    /// Information about optimal batch size for the given device and network.
    OptimalBatchSize,
    /// Maximum batch size which does not cause performance degradation due to memory swap impact.
    MaxBatchSize,
    /// Read-write property key.
    Rw(RwPropertyKey),
    /// An arbitrary key.
    Other(Cow<'static, str>),
}

/// Read-write property keys.
#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Clone)]
pub enum RwPropertyKey {
    /// The directory which will be used to store any data cached by plugins.
    CacheDir,
    /// The cache mode between `optimize_size` and `optimize_speed`. If `optimize_size` is selected,
    /// smaller cache files will be created. If `optimize_speed` is selected, loading time will
    /// decrease but the cache file size will increase.
    CacheMode,
    /// The number of executor logical partitions.
    NumStreams,
    /// The maximum number of threads that can be used for inference tasks.
    InferenceNumThreads,
    /// High-level OpenVINO hint for using CPU pinning to bind CPU threads to processors during inference.
    HintEnableCpuPinning,
    /// High-level OpenVINO hint for using hyper threading processors during CPU inference.
    HintEnableHyperThreading,
    /// High-level OpenVINO Performance Hints.
    HintPerformanceMode,
    /// High-level OpenVINO Hints for the type of CPU core used during inference.
    HintSchedulingCoreType,
    /// Hint for device to use specified precision for inference.
    HintInferencePrecision,
    /// Backs the performance hints by giving additional information on how many inference requests
    /// the application will be keeping in flight usually this value comes from the actual use-case
    /// (e.g. number of video-cameras, or other sources of inputs)
    HintNumRequests,
    /// Desirable log level.
    LogLevel,
    /// High-level OpenVINO model priority hint.
    HintModelPriority,
    /// Performance counters.
    EnableProfiling,
    /// Device priorities configuration, with comma-separated devices listed in the desired
    /// priority.
    DevicePriorities,
    /// A high-level OpenVINO execution hint. Unlike low-level properties that are individual
    /// (per-device), the hints are something that every device accepts and turns into
    /// device-specific settings, the execution mode hint controls preferred optimization targets
    /// (performance or accuracy) for a given model.
    ///
    /// It can be set to be below value:
    /// - `"PERFORMANCE"`: optimize for max performance
    /// - `"ACCURACY"`: optimize for max accuracy
    HintExecutionMode,
    /// Whether to force terminate TBB when OV Core is destroyed.
    ForceTbbTerminate,
    /// Configure `mmap()` use for model read.
    EnableMmap,
    /// ?
    AutoBatchTimeout,
    /// An arbitrary key.
    Other(Cow<'static, str>),
}

impl AsRef<str> for PropertyKey {
    fn as_ref(&self) -> &str {
        match self {
            PropertyKey::SupportedProperties => "SUPPORTED_PROPERTIES",
            PropertyKey::AvailableDevices => "AVAILABLE_DEVICES",
            PropertyKey::OptimalNumberOfInferRequests => "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
            PropertyKey::RangeForAsyncInferRequests => "RANGE_FOR_ASYNC_INFER_REQUESTS",
            PropertyKey::RangeForStreams => "RANGE_FOR_STREAMS",
            PropertyKey::DeviceFullName => "FULL_DEVICE_NAME",
            PropertyKey::DeviceCapabilities => "OPTIMIZATION_CAPABILITIES",
            PropertyKey::ModelName => "NETWORK_NAME",
            PropertyKey::OptimalBatchSize => "OPTIMAL_BATCH_SIZE",
            PropertyKey::MaxBatchSize => "MAX_BATCH_SIZE",
            PropertyKey::Rw(rw) => rw.as_ref(),
            PropertyKey::Other(s) => s,
        }
    }
}

impl AsRef<str> for RwPropertyKey {
    fn as_ref(&self) -> &str {
        match self {
            RwPropertyKey::CacheDir => "CACHE_DIR",
            RwPropertyKey::CacheMode => "CACHE_MODE",
            RwPropertyKey::NumStreams => "NUM_STREAMS",
            RwPropertyKey::InferenceNumThreads => "INFERENCE_NUM_THREADS",
            RwPropertyKey::HintEnableCpuPinning => "ENABLE_CPU_PINNING",
            RwPropertyKey::HintEnableHyperThreading => "ENABLE_HYPER_THREADING",
            RwPropertyKey::HintPerformanceMode => "PERFORMANCE_HINT",
            RwPropertyKey::HintSchedulingCoreType => "SCHEDULING_CORE_TYPE",
            RwPropertyKey::HintInferencePrecision => "INFERENCE_PRECISION_HINT",
            RwPropertyKey::HintNumRequests => "PERFORMANCE_HINT_NUM_REQUESTS",
            RwPropertyKey::LogLevel => "LOG_LEVEL",
            RwPropertyKey::HintModelPriority => "MODEL_PRIORITY",
            RwPropertyKey::EnableProfiling => "PERF_COUNT",
            RwPropertyKey::DevicePriorities => "MULTI_DEVICE_PRIORITIES",
            RwPropertyKey::HintExecutionMode => "EXECUTION_MODE_HINT",
            RwPropertyKey::ForceTbbTerminate => "FORCE_TBB_TERMINATE",
            RwPropertyKey::EnableMmap => "ENABLE_MMAP",
            RwPropertyKey::AutoBatchTimeout => "AUTO_BATCH_TIMEOUT",
            RwPropertyKey::Other(s) => s,
        }
    }
}

impl From<RwPropertyKey> for PropertyKey {
    fn from(key: RwPropertyKey) -> Self {
        PropertyKey::Rw(key)
    }
}
