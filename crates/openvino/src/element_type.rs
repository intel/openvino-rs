use openvino_sys::{
    ov_element_type_e_BF16, ov_element_type_e_DYNAMIC, ov_element_type_e_F16,
    ov_element_type_e_F32, ov_element_type_e_F64, ov_element_type_e_F8E4M3,
    ov_element_type_e_F8E5M3, ov_element_type_e_I16, ov_element_type_e_I32, ov_element_type_e_I4,
    ov_element_type_e_I64, ov_element_type_e_I8, ov_element_type_e_NF4,
    ov_element_type_e_OV_BOOLEAN, ov_element_type_e_U1, ov_element_type_e_U16,
    ov_element_type_e_U32, ov_element_type_e_U4, ov_element_type_e_U64, ov_element_type_e_U8,
    ov_element_type_e_UNDEFINED,
};

use std::convert::TryFrom;
use std::error::Error;
use std::fmt;

/// `ElementType` represents the type of elements that a tensor can hold. See [`ElementType`](https://docs.openvino.ai/2024/api/c_cpp_api/group__ov__base__c__api.html#_CPPv417ov_element_type_e).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ElementType {
    /// An undefined element type.
    Undefined = ov_element_type_e_UNDEFINED,
    /// A dynamic element type.
    Dynamic = ov_element_type_e_DYNAMIC,
    /// A boolean element type.
    Boolean = ov_element_type_e_OV_BOOLEAN,
    /// A Bf16 element type.
    Bf16 = ov_element_type_e_BF16,
    /// A F16 element type.
    F16 = ov_element_type_e_F16,
    /// A F32 element type.
    F32 = ov_element_type_e_F32,
    /// A F64 element type.
    F64 = ov_element_type_e_F64,
    /// A 4-bit integer element type.
    I4 = ov_element_type_e_I4,
    /// An 8-bit integer element type.
    I8 = ov_element_type_e_I8,
    /// A 16-bit integer element type.
    I16 = ov_element_type_e_I16,
    /// A 32-bit integer element type.
    I32 = ov_element_type_e_I32,
    /// A 64-bit integer element type.
    I64 = ov_element_type_e_I64,
    /// An 1-bit unsigned integer element type.
    U1 = ov_element_type_e_U1,
    /// An 4-bit unsigned integer element type.
    U4 = ov_element_type_e_U4,
    /// An 8-bit unsigned integer element type.
    U8 = ov_element_type_e_U8,
    /// A 16-bit unsigned integer element type.
    U16 = ov_element_type_e_U16,
    /// A 32-bit unsigned integer element type.
    U32 = ov_element_type_e_U32,
    /// A 64-bit unsigned integer element type.
    U64 = ov_element_type_e_U64,
    /// NF4 element type.
    NF4 = ov_element_type_e_NF4,
    /// F8E4M3 element type.
    F8E4M3 = ov_element_type_e_F8E4M3,
    /// F8E5M3 element type.
    F8E5M3 = ov_element_type_e_F8E5M3,
}

/// Error returned when attempting to create an [`ElementType`] from an illegal `u32` value.
#[derive(Debug)]
pub struct IllegalValueError(u32);

impl Error for IllegalValueError {}

impl fmt::Display for IllegalValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "illegal value: {}", self.0)
    }
}

impl TryFrom<u32> for ElementType {
    type Error = IllegalValueError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        #[allow(non_upper_case_globals)]
        match value {
            ov_element_type_e_UNDEFINED => Ok(Self::Undefined),
            ov_element_type_e_DYNAMIC => Ok(Self::Dynamic),
            ov_element_type_e_OV_BOOLEAN => Ok(Self::Boolean),
            ov_element_type_e_BF16 => Ok(Self::Bf16),
            ov_element_type_e_F16 => Ok(Self::F16),
            ov_element_type_e_F32 => Ok(Self::F32),
            ov_element_type_e_F64 => Ok(Self::F64),
            ov_element_type_e_I4 => Ok(Self::I4),
            ov_element_type_e_I8 => Ok(Self::I8),
            ov_element_type_e_I16 => Ok(Self::I16),
            ov_element_type_e_I32 => Ok(Self::I32),
            ov_element_type_e_I64 => Ok(Self::I64),
            ov_element_type_e_U1 => Ok(Self::U1),
            ov_element_type_e_U4 => Ok(Self::U4),
            ov_element_type_e_U8 => Ok(Self::U8),
            ov_element_type_e_U16 => Ok(Self::U16),
            ov_element_type_e_U32 => Ok(Self::U32),
            ov_element_type_e_U64 => Ok(Self::U64),
            ov_element_type_e_NF4 => Ok(Self::NF4),
            ov_element_type_e_F8E4M3 => Ok(Self::F8E4M3),
            ov_element_type_e_F8E5M3 => Ok(Self::F8E5M3),
            _ => Err(IllegalValueError(value)),
        }
    }
}

impl From<ElementType> for u32 {
    fn from(value: ElementType) -> Self {
        value as Self
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undefined => write!(f, "Undefined"),
            Self::Dynamic => write!(f, "Dynamic"),
            Self::Boolean => write!(f, "Boolean"),
            Self::Bf16 => write!(f, "Bf16"),
            Self::F16 => write!(f, "F16"),
            Self::F32 => write!(f, "F32"),
            Self::F64 => write!(f, "F64"),
            Self::I4 => write!(f, "I4"),
            Self::I8 => write!(f, "I8"),
            Self::I16 => write!(f, "I16"),
            Self::I32 => write!(f, "I32"),
            Self::I64 => write!(f, "I64"),
            Self::U1 => write!(f, "U1"),
            Self::U4 => write!(f, "U4"),
            Self::U8 => write!(f, "U8"),
            Self::U16 => write!(f, "U16"),
            Self::U32 => write!(f, "U32"),
            Self::U64 => write!(f, "U64"),
            Self::NF4 => write!(f, "NF4"),
            Self::F8E4M3 => write!(f, "F8E4M3"),
            Self::F8E5M3 => write!(f, "F8E5M3"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto as _;

    #[test]
    fn try_from_u32() {
        assert_eq!(ElementType::Undefined, 0u32.try_into().unwrap());
        let last: u32 = ElementType::F8E5M3.into();
        let result: Result<ElementType, _> = (last + 1).try_into();
        assert!(result.is_err());
    }
}
