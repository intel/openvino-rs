use std::convert::TryFrom;
use std::error::Error;
use std::fmt;

/// `ElementType` represents the type of elements that a tensor can hold. See [`ElementType`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__base__c__api.html#_CPPv417ov_element_type_e).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ElementType {
    /// An undefined element type.
    Undefined = 0,
    /// A dynamic element type.
    Dynamic = 1,
    /// A boolean element type.
    Boolean = 2,
    /// A Bf16 element type.
    Bf16 = 3,
    /// A F16 element type.
    F16 = 4,
    /// A F32 element type.
    F32 = 5,
    /// A F64 element type.
    F64 = 6,
    /// A 4-bit integer element type.
    I4 = 7,
    /// An 8-bit integer element type.
    I8 = 8,
    /// A 16-bit integer element type.
    I16 = 9,
    /// A 32-bit integer element type.
    I32 = 10,
    /// A 64-bit integer element type.
    I64 = 11,
    /// An 1-bit unsigned integer element type.
    U1 = 12,
    /// An 4-bit unsigned integer element type.
    U4 = 13,
    /// An 8-bit unsigned integer element type.
    U8 = 14,
    /// A 16-bit unsigned integer element type.
    U16 = 15,
    /// A 32-bit unsigned integer element type.
    U32 = 16,
    /// A 64-bit unsigned integer element type.
    U64 = 17,
    /// NF4 element type.
    NF4 = 18,
    /// F8E4M3 element type.
    F8E4M3 = 19,
    /// F8E5M3 element type.
    F8E5M3 = 20,
}

/// Error returned when attempting to create an [ElementType] from an illegal `u32` value.
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
        match value {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Dynamic),
            2 => Ok(Self::Boolean),
            3 => Ok(Self::Bf16),
            4 => Ok(Self::F16),
            5 => Ok(Self::F32),
            6 => Ok(Self::F64),
            7 => Ok(Self::I4),
            8 => Ok(Self::I8),
            9 => Ok(Self::I16),
            10 => Ok(Self::I32),
            11 => Ok(Self::I64),
            12 => Ok(Self::U1),
            13 => Ok(Self::U4),
            14 => Ok(Self::U8),
            15 => Ok(Self::U16),
            16 => Ok(Self::U32),
            17 => Ok(Self::U64),
            18 => Ok(Self::NF4),
            19 => Ok(Self::F8E4M3),
            20 => Ok(Self::F8E5M3),
            _ => Err(IllegalValueError(value)),
        }
    }
}

impl From<ElementType> for u32 {
    fn from(value: ElementType) -> Self {
        match value {
            ElementType::Undefined => 0,
            ElementType::Dynamic => 1,
            ElementType::Boolean => 2,
            ElementType::Bf16 => 3,
            ElementType::F16 => 4,
            ElementType::F32 => 5,
            ElementType::F64 => 6,
            ElementType::I4 => 7,
            ElementType::I8 => 8,
            ElementType::I16 => 9,
            ElementType::I32 => 10,
            ElementType::I64 => 11,
            ElementType::U1 => 12,
            ElementType::U4 => 13,
            ElementType::U8 => 14,
            ElementType::U16 => 15,
            ElementType::U32 => 16,
            ElementType::U64 => 17,
            ElementType::NF4 => 18,
            ElementType::F8E4M3 => 19,
            ElementType::F8E5M3 => 20,
        }
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
    use openvino_sys::*;
    use std::convert::TryInto as _;

    #[test]
    fn check_discriminant_values() {
        assert_eq!(ov_element_type_e_UNDEFINED, ElementType::Undefined as u32);
        assert_eq!(ov_element_type_e_U1, ElementType::U1 as u32);
        assert_eq!(ov_element_type_e_U4, ElementType::U4 as u32);
        assert_eq!(ov_element_type_e_U8, ElementType::U8 as u32);
        assert_eq!(ov_element_type_e_U16, ElementType::U16 as u32);
        assert_eq!(ov_element_type_e_U32, ElementType::U32 as u32);
        assert_eq!(ov_element_type_e_U64, ElementType::U64 as u32);
        assert_eq!(ov_element_type_e_I4, ElementType::I4 as u32);
        assert_eq!(ov_element_type_e_I8, ElementType::I8 as u32);
        assert_eq!(ov_element_type_e_I16, ElementType::I16 as u32);
        assert_eq!(ov_element_type_e_I32, ElementType::I32 as u32);
        assert_eq!(ov_element_type_e_I64, ElementType::I64 as u32);
        assert_eq!(ov_element_type_e_F16, ElementType::F16 as u32);
        assert_eq!(ov_element_type_e_F32, ElementType::F32 as u32);
        assert_eq!(ov_element_type_e_F64, ElementType::F64 as u32);
        assert_eq!(ov_element_type_e_OV_BOOLEAN, ElementType::Boolean as u32);
        assert_eq!(ov_element_type_e_DYNAMIC, ElementType::Dynamic as u32);
        assert_eq!(ov_element_type_e_NF4, ElementType::NF4 as u32);
        assert_eq!(ov_element_type_e_F8E4M3, ElementType::F8E4M3 as u32);
        assert_eq!(ov_element_type_e_F8E5M3, ElementType::F8E5M3 as u32);
    }

    #[test]
    fn try_from_u32() {
        assert_eq!(ElementType::Undefined, 0u32.try_into().unwrap());
        let last: u32 = ElementType::F8E5M3.into();
        let result: Result<ElementType, _> = (last + 1).try_into();
        assert!(result.is_err());
    }
}
