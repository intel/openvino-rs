/// `ElementType` represents the type of elements that a tensor can hold.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum ElementType {
    /// An undefined element type.
    Undefined = openvino_sys::ov_element_type_e_UNDEFINED,
    /// A dynamic element type.
    Dynamic = openvino_sys::ov_element_type_e_DYNAMIC,
    /// A boolean element type.
    Boolean = openvino_sys::ov_element_type_e_OV_BOOLEAN,
    /// A Bf16 element type.
    Bf16 = openvino_sys::ov_element_type_e_BF16,
    /// A F16 element type.
    F16 = openvino_sys::ov_element_type_e_F16,
    /// A F32 element type.
    F32 = openvino_sys::ov_element_type_e_F32,
    /// A F64 element type.
    F64 = openvino_sys::ov_element_type_e_F64,
    /// A 4-bit integer element type.
    I4 = openvino_sys::ov_element_type_e_I4,
    /// An 8-bit integer element type.
    I8 = openvino_sys::ov_element_type_e_I8,
    /// A 16-bit integer element type.
    I16 = openvino_sys::ov_element_type_e_I16,
    /// A 32-bit integer element type.
    I32 = openvino_sys::ov_element_type_e_I32,
    /// A 64-bit integer element type.
    I64 = openvino_sys::ov_element_type_e_I64,
    /// An 1-bit unsigned integer element type.
    U1 = openvino_sys::ov_element_type_e_U1,
    /// An 4-bit unsigned integer element type.
    U4 = openvino_sys::ov_element_type_e_U4,
    /// An 8-bit unsigned integer element type.
    U8 = openvino_sys::ov_element_type_e_U8,
    /// A 16-bit unsigned integer element type.
    U16 = openvino_sys::ov_element_type_e_U16,
    /// A 32-bit unsigned integer element type.
    U32 = openvino_sys::ov_element_type_e_U32,
    /// A 64-bit unsigned integer element type.
    U64 = openvino_sys::ov_element_type_e_U64,
    /// NF4 element type.
    NF4 = openvino_sys::ov_element_type_e_NF4,
}
