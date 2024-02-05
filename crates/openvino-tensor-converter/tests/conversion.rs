use env_logger;
use openvino_tensor_converter::{convert, Dimensions, Precision};

#[test]
fn same_result_twice_u8() {
    let input = "tests/test.jpg";
    let dimensions = Dimensions::new(227, 227, 3, Precision::U8);

    let first = convert(input, &dimensions).unwrap();
    let second = convert(input, &dimensions).unwrap();
    assert_same_bytes(&first, &second);
}

#[test]
fn same_result_twice_fp32() {
    env_logger::init();
    let input = "tests/test.jpg";
    let dimensions = Dimensions::new(227, 227, 3, Precision::FP32);

    let first = convert(input, &dimensions).unwrap();
    let second = convert(input, &dimensions).unwrap();
    assert_same_bytes(&first, &second);
}

fn assert_same_bytes(a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        if a != b {
            panic!("First not-equal byte at index {}: {} != {}", i, a, b);
        }
    }
}
