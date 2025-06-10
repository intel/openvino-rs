use openvino_sys::ov_element_type_e;
use std::fmt;

/// See
/// [`ov_element_type_e`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__base__c__api.html#_CPPv417ov_element_type_e).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ElementType {
    /// A dynamic element type.
    Dynamic,
    /// A boolean element type.
    Boolean,
    /// A Bf16 element type.
    Bf16,
    /// A F16 element type.
    F16,
    /// A F32 element type.
    F32,
    /// A F64 element type.
    F64,
    /// A 4-bit integer element type.
    I4,
    /// An 8-bit integer element type.
    I8,
    /// A 16-bit integer element type.
    I16,
    /// A 32-bit integer element type.
    I32,
    /// A 64-bit integer element type.
    I64,
    /// A 1-bit unsigned integer element type.
    U1,
    /// A 2-bit unsigned integer element type.
    U2,
    /// A 3-bit unsigned integer element type.
    U3,
    /// A 4-bit unsigned integer element type.
    U4,
    /// A 6-bit unsigned integer element type.
    U6,
    /// An 8-bit unsigned integer element type.
    U8,
    /// A 16-bit unsigned integer element type.
    U16,
    /// A 32-bit unsigned integer element type.
    U32,
    /// A 64-bit unsigned integer element type.
    U64,
    /// NF4 element type.
    NF4,
    /// F8E4M3 element type.
    F8E4M3,
    /// F8E5M3 element type.
    F8E5M3,
    /// A string element type.
    String,
    /// F4E2M1 element type.
    F4E2M1,
    /// F8E8M0 element type.
    F8E8M0,
}

impl From<ov_element_type_e> for ElementType {
    fn from(ty: ov_element_type_e) -> Self {
        match ty {
            ov_element_type_e::DYNAMIC => Self::Dynamic,
            ov_element_type_e::OV_BOOLEAN => Self::Boolean,
            ov_element_type_e::BF16 => Self::Bf16,
            ov_element_type_e::F16 => Self::F16,
            ov_element_type_e::F32 => Self::F32,
            ov_element_type_e::F64 => Self::F64,
            ov_element_type_e::I4 => Self::I4,
            ov_element_type_e::I8 => Self::I8,
            ov_element_type_e::I16 => Self::I16,
            ov_element_type_e::I32 => Self::I32,
            ov_element_type_e::I64 => Self::I64,
            ov_element_type_e::U1 => Self::U1,
            ov_element_type_e::U2 => Self::U2,
            ov_element_type_e::U3 => Self::U3,
            ov_element_type_e::U4 => Self::U4,
            ov_element_type_e::U6 => Self::U6,
            ov_element_type_e::U8 => Self::U8,
            ov_element_type_e::U16 => Self::U16,
            ov_element_type_e::U32 => Self::U32,
            ov_element_type_e::U64 => Self::U64,
            ov_element_type_e::NF4 => Self::NF4,
            ov_element_type_e::F8E4M3 => Self::F8E4M3,
            ov_element_type_e::F8E5M3 => Self::F8E5M3,
            ov_element_type_e::STRING => Self::String,
            ov_element_type_e::F4E2M1 => Self::F4E2M1,
            ov_element_type_e::F8E8M0 => Self::F8E8M0,
        }
    }
}

impl From<ElementType> for ov_element_type_e {
    fn from(ty: ElementType) -> ov_element_type_e {
        match ty {
            ElementType::Dynamic => ov_element_type_e::DYNAMIC,
            ElementType::Boolean => ov_element_type_e::OV_BOOLEAN,
            ElementType::Bf16 => ov_element_type_e::BF16,
            ElementType::F16 => ov_element_type_e::F16,
            ElementType::F32 => ov_element_type_e::F32,
            ElementType::F64 => ov_element_type_e::F64,
            ElementType::I4 => ov_element_type_e::I4,
            ElementType::I8 => ov_element_type_e::I8,
            ElementType::I16 => ov_element_type_e::I16,
            ElementType::I32 => ov_element_type_e::I32,
            ElementType::I64 => ov_element_type_e::I64,
            ElementType::U1 => ov_element_type_e::U1,
            ElementType::U2 => ov_element_type_e::U2,
            ElementType::U3 => ov_element_type_e::U3,
            ElementType::U4 => ov_element_type_e::U4,
            ElementType::U6 => ov_element_type_e::U6,
            ElementType::U8 => ov_element_type_e::U8,
            ElementType::U16 => ov_element_type_e::U16,
            ElementType::U32 => ov_element_type_e::U32,
            ElementType::U64 => ov_element_type_e::U64,
            ElementType::NF4 => ov_element_type_e::NF4,
            ElementType::F8E4M3 => ov_element_type_e::F8E4M3,
            ElementType::F8E5M3 => ov_element_type_e::F8E5M3,
            ElementType::String => ov_element_type_e::STRING,
            ElementType::F4E2M1 => ov_element_type_e::F4E2M1,
            ElementType::F8E8M0 => ov_element_type_e::F8E8M0,
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dynamic => write!(f, "Dynamic"),
            Self::Boolean => write!(f, "Boolean"),
            Self::Bf16 => write!(f, "BF16"),
            Self::F16 => write!(f, "F16"),
            Self::F32 => write!(f, "F32"),
            Self::F64 => write!(f, "F64"),
            Self::I4 => write!(f, "I4"),
            Self::I8 => write!(f, "I8"),
            Self::I16 => write!(f, "I16"),
            Self::I32 => write!(f, "I32"),
            Self::I64 => write!(f, "I64"),
            Self::U1 => write!(f, "U1"),
            Self::U2 => write!(f, "U2"),
            Self::U3 => write!(f, "U3"),
            Self::U4 => write!(f, "U4"),
            Self::U6 => write!(f, "U6"),
            Self::U8 => write!(f, "U8"),
            Self::U16 => write!(f, "U16"),
            Self::U32 => write!(f, "U32"),
            Self::U64 => write!(f, "U64"),
            Self::NF4 => write!(f, "NF4"),
            Self::F8E4M3 => write!(f, "F8E4M3"),
            Self::F8E5M3 => write!(f, "F8E5M3"),
            Self::String => write!(f, "String"),
            Self::F4E2M1 => write!(f, "F4E2M1"),
            Self::F8E8M0 => write!(f, "F8E8M0"),
        }
    }
}
