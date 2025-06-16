//! An experimental library demonstrating how to use `OpenCV` in Rust to convert images into
//! OpenVINO-compatible tensors.
//!
//! > WARNING: this is still experimental--no correctness guarantees!

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]

use core::{fmt, slice};
use log::info;
use opencv::core::{MatTraitConst, Scalar_};
use std::convert::TryInto;
use std::{num::ParseIntError, path::Path, str::FromStr};

/// Convert an image from NHWC format to NCHW format.
fn nhwc_to_nchw(data: &[u8], dimensions: &Dimensions) -> Vec<u8> {
    let mut nchw_data = vec![0; data.len()];
    let (height, width, channels) = (
        dimensions.height as usize,
        dimensions.width as usize,
        dimensions.channels as usize,
    );
    assert_eq!(
        data.len(),
        height * width * channels * dimensions.precision.bytes()
    );
    for h in 0..height {
        for w in 0..width {
            for c in 0..channels {
                let nhwc_index =
                    (h * width * channels + w * channels + c) * dimensions.precision.bytes();
                let nchw_index =
                    (c * height * width + h * width + w) * dimensions.precision.bytes();
                for b in 0..dimensions.precision.bytes() {
                    nchw_data[nchw_index + b] = data[nhwc_index + b];
                }
            }
        }
    }
    nchw_data
}

/// Convert an image a path to a resized sequence of bytes.
///
/// # Errors
///
/// This function will return an error if the path is not a valid file, if the path cannot be
/// converted to a string, or if the conversion fails for any other reason.
pub fn convert<P: AsRef<Path>>(
    path: P,
    dimensions: &Dimensions,
    format: &str,
) -> Result<Vec<u8>, ConversionError> {
    let path = path.as_ref();
    info!("Converting {} to {:?}", path.display(), dimensions);
    if !path.is_file() {
        return Err(ConversionError("The path is not a valid file.".to_string()));
    }

    // Decode the source image. This uses the default flags (see
    // https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) to match what
    // OpenVINO's wrapper does (see
    // https://github.com/openvinotoolkit/openvino/blob/7566e8202fa6c00f27de27889e7bf99d7ddf2636/inference-engine/ie_bridges/c/samples/common/opencv_c_wraper.cpp#L25).
    let path_as_str = path
        .to_str()
        .ok_or(ConversionError("Unable to stringify the path.".to_string()))?;
    let src = opencv::imgcodecs::imread(path_as_str, opencv::imgcodecs::IMREAD_COLOR)?;
    info!("The input image has size = {:?}, channels = {}, type = {}, total items = {}, item size (bytes) = {}", src.size()?, src.channels(), src.typ(), src.total(), src.elem_size1());

    // Create a destination Mat of the right shape, filling it with 0s (see
    // https://docs.rs/opencv/0.46.3/opencv/core/struct.Mat.html#method.new_rows_cols_with_default).
    let mut resized = opencv::core::Mat::new_rows_cols_with_default(
        dimensions.height,
        dimensions.width,
        dimensions.as_type(),
        Scalar_::all(0.0),
    )?;
    info!("Before resizing, the `resize` image has size = {:?}, channels = {}, type = {}, total items = {}, item size (bytes) = {}", resized.size(), resized.channels(), resized.typ(), resized.total(), resized.elem_size1());

    // Resize the `src` Mat into the `dst` Mat using bilinear interpolation (see
    // https://docs.rs/opencv/0.46.3/opencv/imgproc/fn.resize.html).
    let dst_size = resized.size()?;
    opencv::imgproc::resize(
        &src,
        &mut resized,
        dst_size,
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    info!("After resizing, the `resize` image has size = {:?}, channels = {}, type = {}, total items = {}, item size (bytes) = {}", resized.size(), resized.channels(), resized.typ(), resized.total(), resized.elem_size1());

    // Because `imgproc::resize` can alter the depth/precision of our destination image, we convert the `resized` image
    // to the appropriate `Precision`.
    let mut dst = opencv::core::Mat::new_rows_cols_with_default(
        dimensions.height,
        dimensions.width,
        dimensions.as_type(),
        Scalar_::all(0.0),
    )?;
    // The alpha/beta values are the defaults from C++.
    resized.convert_to(&mut dst, dimensions.as_type(), 1.0, 0.0)?;
    info!("After conversion, the `dst` image has size = {:?}, channels = {}, type = {}, total items = {}, item size (bytes) = {}", dst.size(), dst.channels(), dst.typ(), dst.total(), dst.elem_size1());

    // Copy the bytes of the Mat out to a Vec<u8>.
    let dst_slice = unsafe { slice::from_raw_parts(dst.data(), dimensions.bytes()) };
    let nhwc_data = dst_slice.to_vec();
    match format {
        "nchw" => Ok(nhwc_to_nchw(&nhwc_data, dimensions)),
        "nhwc" => Ok(nhwc_data),
        _ => Err(ConversionError("Invalid format specified.".to_string())),
    }
}

/// Container for the reasons a conversion can fail.
#[derive(Debug)]
pub struct ConversionError(String);
impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl From<opencv::Error> for ConversionError {
    fn from(e: opencv::Error) -> Self {
        Self(e.message)
    }
}
impl From<ParseIntError> for ConversionError {
    fn from(e: ParseIntError) -> Self {
        Self(format!("parsing error: {e}"))
    }
}

/// Define the dimensions and pixel precision of an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dimensions {
    height: i32,
    width: i32,
    channels: i32,
    precision: Precision,
}
impl Dimensions {
    /// Construct a new dimensions object.
    #[must_use]
    pub fn new(height: i32, width: i32, channels: i32, precision: Precision) -> Self {
        Self {
            height,
            width,
            channels,
            precision,
        }
    }

    /// Return the number of bytes that the dimensions should occupy.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of items overflows a `usize`.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let num_items: usize = (self.height * self.width * self.channels)
            .try_into()
            .expect("overflow in number of items");
        num_items * self.precision.bytes()
    }

    /// See `OpenCV`'s [basic structures] for a description of the various primitive types.
    ///
    /// [basic structures]: https://docs.opencv.org/2.4/modules/core/doc/basic_structures.html
    fn as_type(&self) -> i32 {
        use Precision::{FP32, U8};
        match (self.precision, self.channels) {
            (FP32, 3) => opencv::core::CV_32FC3,
            (U8, 3) => opencv::core::CV_8UC3,
            _ => unimplemented!(),
        }
    }
}
impl FromStr for Dimensions {
    type Err = ConversionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.trim().split('x').collect();
        if parts.len() != 4 {
            return Err(ConversionError("Not enough parts in dimension string; should be [height]x[width]x[channels]x[precision]".to_string()));
        }
        let height = i32::from_str(parts[0])?;
        let width = i32::from_str(parts[1])?;
        let channels = i32::from_str(parts[2])?;
        let precision = Precision::from_str(parts[3])?;
        Ok(Self {
            height,
            width,
            channels,
            precision,
        })
    }
}

/// Distinguish the precision of each pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Each pixel is an 8-bit value.
    U8,
    /// Each pixel is a 32-bit floating point value.
    FP32,
}
impl Precision {
    /// Return the number of bytes occupied by the precision.
    #[must_use]
    pub fn bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::FP32 => 4,
        }
    }
}
impl FromStr for Precision {
    type Err = ConversionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "u8" => Ok(Self::U8),
            "fp32" => Ok(Self::FP32),
            _ => Err(ConversionError(format!("unrecognized precision: {s}"))),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn parse() {
        assert_eq!(
            Dimensions::from_str("100x20x3xfp32").unwrap(),
            Dimensions::new(100, 20, 3, Precision::FP32)
        );
    }
}
