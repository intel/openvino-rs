//! This test originates from some strange segfaults that we observed while using the OpenVINO C
//! library. Because OpenVINO is taking a pointer to a tensor when constructing a model, we want to
//! be sure that we do the right thing on this side of the FFI boundary.

mod fixtures;

use fixtures::mobilenet as fixture;
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
use std::fs;

#[test]
fn memory_safety() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let xml = fs::read_to_string(fixture::graph())?;
    let weights = fs::read(fixture::weights())?;

    // Copy the fixture weights into a tensor. Once we're done here we want to get rid of the
    // original weights buffer as a sanity check.
    let shape = Shape::new(&[1, weights.len() as i64])?;
    let mut weights_tensor = Tensor::new(ElementType::U8, &shape)?;
    weights_tensor.get_raw_data_mut()?.copy_from_slice(&weights);
    drop(weights);

    // Now create a model from a reference to the weights tensor. We observed segfault crashes when
    // passing weights by value but not by reference.
    let model = core.read_model_from_buffer(xml.as_bytes(), Some(&weights_tensor))?;
    drop(weights_tensor);

    // Here we double-check that the model is usable. Though it has captured a reference to the
    // `weights_tensor` and that tensor has been dropped, whatever OpenVINO is doing internally must
    // be safe enough. See
    // https://github.com/openvinotoolkit/openvino/blob/d840d86905f013d95cccbafaa0ddff266e250f75/src/inference/src/model_reader.cpp#L178.
    assert_eq!(model.get_inputs_len()?, 1);
    assert!(core.compile_model(&model, DeviceType::CPU).is_ok());
    Ok(())
}

fn relu_model(core: &mut Core, port_count: usize) -> anyhow::Result<openvino::CompiledModel> {
    let mut layers = String::new();
    let mut edges = String::new();
    for index in 0..port_count {
        let input_id = index * 3;
        let relu_id = input_id + 1;
        let result_id = input_id + 2;
        layers.push_str(&format!(
            r#"
            <layer id="{input_id}" name="input_{index}" type="Parameter" version="opset1">
                <data shape="4" element_type="f32"/>
                <output><port id="0" precision="FP32" names="input_{index}"><dim>4</dim></port></output>
            </layer>
            <layer id="{relu_id}" name="relu_{index}" type="ReLU" version="opset1">
                <input><port id="0" precision="FP32"><dim>4</dim></port></input>
                <output><port id="1" precision="FP32" names="output_{index}"><dim>4</dim></port></output>
            </layer>
            <layer id="{result_id}" name="result_{index}" type="Result" version="opset1">
                <input><port id="0" precision="FP32"><dim>4</dim></port></input>
            </layer>"#
        ));
        edges.push_str(&format!(
            r#"
            <edge from-layer="{input_id}" from-port="0" to-layer="{relu_id}" to-port="0"/>
            <edge from-layer="{relu_id}" from-port="1" to-layer="{result_id}" to-port="0"/>"#
        ));
    }
    let xml = format!(
        r#"<net name="relu" version="11"><layers>{layers}</layers><edges>{edges}</edges></net>"#
    );
    let model = core.read_model_from_buffer(xml.as_bytes(), None)?;
    Ok(core.compile_model(&model, DeviceType::CPU)?)
}

fn assert_zeroed(tensor: Tensor) -> anyhow::Result<()> {
    let data = tensor.get_raw_data()?;
    assert_eq!(data.len(), 4 * std::mem::size_of::<f32>());
    assert!(data.iter().all(|byte| *byte == 0));
    Ok(())
}

#[test]
fn request_tensors_are_initialized() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    for port_count in [1, 2] {
        let mut model = relu_model(&mut core, port_count)?;
        assert_eq!(model.get_input_size()?, port_count);
        assert_eq!(model.get_output_size()?, port_count);
        let mut request = model.create_infer_request()?;

        for index in 0..port_count {
            assert_zeroed(request.get_tensor(&format!("input_{index}"))?)?;
            assert_zeroed(request.get_tensor(&format!("output_{index}"))?)?;
            assert_zeroed(request.get_output_tensor_by_index(index)?)?;
        }
        if port_count == 1 {
            assert_zeroed(request.get_input_tensor()?)?;
            assert_zeroed(request.get_output_tensor()?)?;
        }

        request.infer()?;
        for index in 0..port_count {
            assert_zeroed(request.get_output_tensor_by_index(index)?)?;
        }

        for value in [2.0_f32, 5.0_f32] {
            for index in 0..port_count {
                let mut input = request.get_tensor(&format!("input_{index}"))?;
                input.get_data_mut::<f32>()?.copy_from_slice(&[
                    -value,
                    value + index as f32,
                    value + 1.0,
                    -1.0,
                ]);
            }
            request.infer()?;
            for index in 0..port_count {
                let expected = [0.0, value + index as f32, value + 1.0, 0.0];
                assert_eq!(
                    request
                        .get_output_tensor_by_index(index)?
                        .get_data::<f32>()?,
                    &expected
                );
                assert_eq!(
                    request
                        .get_tensor(&format!("output_{index}"))?
                        .get_data::<f32>()?,
                    &expected
                );
            }
            if port_count == 1 {
                assert_eq!(
                    request.get_output_tensor()?.get_data::<f32>()?,
                    &[0.0, value, value + 1.0, 0.0]
                );
            }
        }
    }
    Ok(())
}
