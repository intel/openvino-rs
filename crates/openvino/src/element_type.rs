/// `ElementType` represents the type of elements that a tensor can hold. See [`ElementType`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__base__c__api.html#_CPPv417ov_element_type_e).
#[derive(Debug)]
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvino_sys::*;

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
    }
}
