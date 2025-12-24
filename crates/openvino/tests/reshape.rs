mod fixtures;

use fixtures::alexnet as fixture;
use openvino::{Core, Dimension, PartialShape};

#[test]
fn test_reshape_single_input() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let mut model = core.read_model_from_file(
        &fixture::graph().to_string_lossy(),
        &fixture::weights().to_string_lossy(),
    )?;

    let input_node = model.get_input_by_index(0)?;
    let initial_shape = input_node.get_shape()?;
    let dims = initial_shape.get_dimensions().to_vec();
    assert_eq!(initial_shape.get_dimensions(), &[1, 3, 227, 227]);

    let mut new_dims = dims.clone();
    new_dims[0] = 2; // Change batch size
    let new_pshape = PartialShape::new_static(new_dims.len() as i64, &new_dims)?;

    model.reshape_single_input(&new_pshape)?;

    let updated_node = model.get_input_by_index(0)?;
    let final_shape = updated_node.get_shape()?;
    assert_eq!(final_shape.get_dimensions(), &[2, 3, 227, 227]);

    assert!(
        !model.is_dynamic(),
        "Model should not be dynamic before reshaping batch to -1"
    );

    let new_dims = vec![
        Dimension::new(-1, -1),
        Dimension::new(dims[1], dims[1]),
        Dimension::new(dims[2], dims[2]),
        Dimension::new(dims[3], dims[3]),
    ];
    let new_pshape = PartialShape::new(dims.len() as i64, &new_dims)?;

    model.reshape_single_input(&new_pshape)?;

    assert!(
        model.is_dynamic(),
        "Model should be dynamic after reshaping batch to -1"
    );

    let final_pshape = model.get_input_by_index(0)?.get_partial_shape()?;
    assert!(final_pshape.get_dimensions()[0].is_dynamic());

    Ok(())
}

#[test]
fn test_reshape_input_by_name() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let mut model = core.read_model_from_file(
        &fixture::graph().to_string_lossy(),
        &fixture::weights().to_string_lossy(),
    )?;

    let input_node = model.get_input_by_index(0)?;
    let name = input_node.get_name()?;
    let dims = input_node.get_shape()?.get_dimensions().to_vec();

    let mut new_dims = dims.clone();
    new_dims[0] = 2; // Change batch size
    let new_pshape = PartialShape::new_static(new_dims.len() as i64, &new_dims)?;

    model.reshape_input_by_name(&name, &new_pshape)?;

    let updated_node = model.get_input_by_index(0)?;
    let final_shape = updated_node.get_shape()?;
    assert_eq!(final_shape.get_dimensions(), &[2, 3, 227, 227]);

    assert!(
        !model.is_dynamic(),
        "Model should not be dynamic before reshaping batch to -1"
    );

    let new_dims = vec![
        Dimension::new(-1, -1),
        Dimension::new(dims[1], dims[1]),
        Dimension::new(dims[2], dims[2]),
        Dimension::new(dims[3], dims[3]),
    ];
    let new_pshape = PartialShape::new(dims.len() as i64, &new_dims)?;

    model.reshape_input_by_name(&name, &new_pshape)?;
    assert!(
        model.is_dynamic(),
        "Model should be dynamic after reshaping batch to -1"
    );

    let final_pshape = model.get_input_by_index(0)?.get_partial_shape()?;
    assert!(final_pshape.get_dimensions()[0].is_dynamic());

    Ok(())
}

#[test]
fn test_reshape_multi() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let mut model = core.read_model_from_file(
        &fixture::graph().to_string_lossy(),
        &fixture::weights().to_string_lossy(),
    )?;

    let input_node = model.get_input_by_index(0)?;
    let name = input_node.get_name()?;
    let dims = input_node.get_shape()?.get_dimensions().to_vec();

    let mut new_dims = dims.clone();
    new_dims[0] = 3;

    let new_pshape = PartialShape::new_static(new_dims.len() as i64, &new_dims)?;

    // Test passing as a slice of tuples
    model.reshape(&[(&name, &new_pshape)])?;

    assert_eq!(
        model.get_input_by_index(0)?.get_shape()?.get_dimensions()[0],
        3
    );

    assert!(
        !model.is_dynamic(),
        "Model should not be dynamic before reshaping batch to -1"
    );

    let new_dims = vec![
        Dimension::new(-1, -1),
        Dimension::new(dims[1], dims[1]),
        Dimension::new(dims[2], dims[2]),
        Dimension::new(dims[3], dims[3]),
    ];
    let new_pshape = PartialShape::new(dims.len() as i64, &new_dims)?;

    model.reshape(&[(&name, &new_pshape)])?;
    assert!(
        model.is_dynamic(),
        "Model should be dynamic after reshaping batch to -1"
    );

    let final_pshape = model.get_input_by_index(0)?.get_partial_shape()?;
    assert!(final_pshape.get_dimensions()[0].is_dynamic());

    Ok(())
}

#[test]
fn test_reshape_by_port_indexes() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let mut model = core.read_model_from_file(
        &fixture::graph().to_string_lossy(),
        &fixture::weights().to_string_lossy(),
    )?;

    let input_node = model.get_input_by_index(0)?;
    let dims = input_node.get_shape()?.get_dimensions().to_vec();

    let mut new_dims = dims.clone();
    new_dims[0] = 3;

    let new_pshape = PartialShape::new_static(new_dims.len() as i64, &new_dims)?;

    // Test passing as a slice of tuples
    model.reshape_by_port_indexes(&[(0, &new_pshape)])?;

    assert_eq!(
        model.get_input_by_index(0)?.get_shape()?.get_dimensions()[0],
        3
    );

    assert!(
        !model.is_dynamic(),
        "Model should not be dynamic before reshaping batch to -1"
    );

    let new_dims = vec![
        Dimension::new(-1, -1),
        Dimension::new(dims[1], dims[1]),
        Dimension::new(dims[2], dims[2]),
        Dimension::new(dims[3], dims[3]),
    ];
    let new_pshape = PartialShape::new(dims.len() as i64, &new_dims)?;

    model.reshape_by_port_indexes(&[(0, &new_pshape)])?;
    assert!(
        model.is_dynamic(),
        "Model should be dynamic after reshaping batch to -1"
    );

    let final_pshape = model.get_input_by_index(0)?.get_partial_shape()?;
    assert!(final_pshape.get_dimensions()[0].is_dynamic());

    Ok(())
}

#[test]
fn test_reshape_by_ports() -> anyhow::Result<()> {
    let mut core = Core::new()?;
    let mut model = core.read_model_from_file(
        &fixture::graph().to_string_lossy(),
        &fixture::weights().to_string_lossy(),
    )?;

    let input_node = model.get_input_by_index(0)?;
    let dims = input_node.get_shape()?.get_dimensions().to_vec();

    let mut new_dims = dims.clone();
    new_dims[0] = 3;

    let new_pshape = PartialShape::new_static(new_dims.len() as i64, &new_dims)?;

    // Test passing as a slice of tuples
    model.reshape_by_ports(&[(&input_node, &new_pshape)])?;

    assert_eq!(
        model.get_input_by_index(0)?.get_shape()?.get_dimensions()[0],
        3
    );

    assert!(
        !model.is_dynamic(),
        "Model should not be dynamic before reshaping batch to -1"
    );

    let new_dims = vec![
        Dimension::new(-1, -1),
        Dimension::new(dims[1], dims[1]),
        Dimension::new(dims[2], dims[2]),
        Dimension::new(dims[3], dims[3]),
    ];
    let new_pshape = PartialShape::new(dims.len() as i64, &new_dims)?;

    model.reshape_by_ports(&[(&input_node, &new_pshape)])?;
    assert!(
        model.is_dynamic(),
        "Model should be dynamic after reshaping batch to -1"
    );

    let final_pshape = model.get_input_by_index(0)?.get_partial_shape()?;
    assert!(final_pshape.get_dimensions()[0].is_dynamic());

    Ok(())
}
