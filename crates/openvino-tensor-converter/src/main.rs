use openvino_tensor_converter::{convert, Dimensions};
use std::{fs, path::PathBuf, str::FromStr};
use structopt::StructOpt;

fn main() {
    env_logger::init();
    let options = Options::from_args();
    let dimensions = Dimensions::from_str(&options.dimensions).expect("Failed to parse dimensions");
    let tensor_data = convert(options.input, &dimensions).expect("Failed to convert image");
    fs::write(options.output, tensor_data).expect("Failed to write tensor")
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "tensor-converter",
    about = "Decode and resize images into valid OpenVINO tensors."
)]
struct Options {
    /// Input file.
    #[structopt(name = "INPUT FILE", parse(from_os_str))]
    input: PathBuf,

    /// Output file.
    #[structopt(name = "OUTPUT FILE", parse(from_os_str))]
    output: PathBuf,

    /// The dimensions of the output file as "[height]x[width]x[channels]x[precision]"; e.g. 300x300x3xfp32.
    #[structopt(name = "OUTPUT DIMENSIONS")]
    dimensions: String,
}
