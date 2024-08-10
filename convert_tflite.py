import argparse
import kaggle_download
import iree.compiler.tflite as iree_tflite_compile
import iree.compiler
import iree.runtime as iree_rt
import tensorflow as tf
import numpy as np
import os


def setup_gdscript(model, url):
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    types = {
        np.int32: "PackedInt32Array",
        np.int64: "PackedInt64Array",
        np.byte: "PackedByteArray",
        np.float32: "PackedFloat32Array",
        np.float64: "PackedFloat64Array",
    }
    convert_types = {
        np.int32: ".to_int32_array()",
        np.int64: ".to_int64_array()",
        np.byte: "",
        np.float32: ".to_float32_array()",
        np.float64: ".to_float64_array()",
    }
    inputs_with_type = ""
    inputs = ""
    output = ""
    output_convert = ""
    for input_detail in input_details:
        inputs_with_type += f"{input_detail['name']}: {types[input_detail['dtype']]},"
        inputs += f"{input_detail['name']},"
    # TODO Add support for multiple outputs
    for output_detail in output_details:
        output += f"{types[output_detail['dtype']]}"
        output_convert += f"{convert_types[output_detail['dtype']]}"
        break
    gdscript_file = f"""extends Node
class_name IREEModule_{url}

var module = load("res://addons/iree-zoo/{url}")

func {url}({inputs_with_type}) -> {output}:\n
    var module : IREEModule = null
    match OS.get_name():
        "Windows", "Linux", "FreeBSD", "NetBSD", "OpenBSD", "BSD":
            module = load("res://addons/iree-zoo/{url}/iree.vulkan-spirv.vmfb")
        "macOS", "iOS":
            module = load("res://addons/iree-zoo/{url}/iree.metal-spirv.vmfb")
        "Android":
            module = load("res://addons/iree-zoo/{url}/iree.metal-llvm.vmfb")
        _:
            assert(false, "Unsupported platform.")
    var output_tensor := (await module.call_module("module.main", [{inputs}]).completed as Array).front() as IREETensor
    return output_tensor.get_data(){output_convert}
"""
    with open(f"build/addons/iree-zoo/{url}/{url}.gd", "w") as f:
        f.write(gdscript_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter model url")
    parser.add_argument("url", type=str, help="The model url from kaggle")
    args = parser.parse_args()
    path = kaggle_download.model_download(args.url)
    targets = ['llvm-cpu', 'metal-spirv', 'vulkan-spirv', 'webgpu-spirv']
    print("Building for: ", targets)
    # Find .tflite file
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tflite"):
                path = os.path.join(root, file)
                break
    print("Path to model file:", path)
    folder_name = args.url.replace('/', '_').replace('-', '_')
    os.makedirs(f"build/addons/iree-zoo/{folder_name}", exist_ok=True)
    # First generate inputs/outputs
    setup_gdscript(path, folder_name)
    for target in targets:
        try:
            print("Compiling for: ", target)
            iree_tflite_compile.compile_file(
                path,
                input_type="tosa",
                output_file=f"build/addons/iree-zoo/{folder_name}/iree.{target}.vmfb",
                target_backends=[target],
                import_only=False,
            )
            print("Compiled for: ", target)
        except Exception as e:
            print(e)
