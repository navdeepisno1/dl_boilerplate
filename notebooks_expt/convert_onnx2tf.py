import onnx2tf

onnx2tf.convert(
    input_onnx_file_path="tiny_vit.onnx",
    output_folder_path="tiny_vit",
    # copy_onnx_input_output_names_to_tflite=True,
    output_dynamic_range_quantized_tflite=True,    
    output_signaturedefs = True
)