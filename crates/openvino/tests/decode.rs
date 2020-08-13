//! These tests exemplify how to use the `opencv` library to read in images for use in
//! `openvino-rs`. Though not directly

mod fixture;
use fixture::Fixture;
use opencv;
use opencv::core::MatTrait;

#[test]
fn read_image() {
    let mat = opencv::imgcodecs::imread(
        &*Fixture::image().to_string_lossy(),
        opencv::imgcodecs::IMREAD_COLOR,
    )
    .unwrap();

    assert_eq!(mat.channels().unwrap(), 3);
    assert_eq!(mat.typ().unwrap(), opencv::core::CV_8UC3);
}

#[test]
fn decode_image() {
    let bytes = std::fs::read(Fixture::image()).unwrap();
    let mut bytes_mat = opencv::core::Mat::from_slice::<u8>(&bytes).unwrap();
    let mat = opencv::imgcodecs::imdecode(&mut bytes_mat, opencv::imgcodecs::IMREAD_COLOR).unwrap();

    assert_eq!(mat.channels().unwrap(), 3);
    assert_eq!(mat.typ().unwrap(), opencv::core::CV_8UC3);
}
