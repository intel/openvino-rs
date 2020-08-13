# OpenVINO Test Fixture

In order to test the use of `openvino-rs` in the real world, here we include the necessary files for performing the
classification in [classify.rs](../classify.rs). The following list describes how these files were collected:

- Download a SSD MobileNet pre-trained model from the TensorFlow [model zoo] and extract it somewhere:
  ```shell script
  wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
  tar xzvf ssd_*.tar.gz
  ln -s ssd_inception_v2_coco_2018_01_28 model
  ```
[model zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

- Using OpenVINO's [model-optimizer documentation], convert the model into OpenVINO IR (i.e. `frozen_inference_graph*`):
  ```shell script
  pip install --user -r ../../upstream/model-optimizer/requirements_tf.txt 
  ../../upstream/model-optimizer/mo_tf.py --input_model model/frozen_inference_graph.pb --transformations_config ../../upstream/model-optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config model/pipeline.config 
  ```

[model-optimizer documentation]: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html

- Download some test images from the COCO data set:
  ```shell script
  wget http://images.cocodataset.org/zips/val2017.zip
  unzip -Z1 val2017.zip | head -n 20 | xargs unzip val2017.zip
  ```

- Optionally, build the OpenVINO example classifier (i.e. `hello_classification`) and verify that the results match
those in `example.rs`:
  ```shell script
  ../../upstream/bin/intel64/Release/hello_classification frozen_inference_graph.xml val2017/000000231527.jpg CPU
  ```
