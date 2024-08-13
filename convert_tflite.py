import argparse
import kaggle_download
import iree.compiler.tflite as iree_tflite_compile
import iree.compiler
import iree.runtime as iree_rt
import tensorflow as tf
import numpy as np
import os


def setup_gdscript(model, url, original_url):
	interpreter = tf.lite.Interpreter(model_path=model)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	inputs = ""
	outputs = ""
	for input_detail in input_details:
		inputs += f"## {input_detail['name']}: {input_detail['dtype']} {input_detail['shape']}\n"
	for output_detail in output_details:
		outputs += f"\n## {output_detail['name']}: {output_detail['dtype']} {output_detail['shape']}"
	gdscript_file = f"""@icon("res://addons/iree-gd/logo.svg")
extends IREERunner
class_name IREEModule_{url}

func _load_module() -> IREEModule:
	match OS.get_name():
		"Windows", "Linux", "FreeBSD", "NetBSD", "OpenBSD", "BSD":
			return IREEModule.new().load("res://addons/iree-zoo/{original_url}/iree.vulkan-spirv.vmfb")
		"macOS", "iOS":
			return IREEModule.new().load("res://addons/iree-zoo/{original_url}/iree.metal-spirv.vmfb")
		"Android":
			return IREEModule.new().load("res://addons/iree-zoo/{original_url}/iree.llvm-cpu.vmfb")
		_:
			assert(false, "Unsupported platform.")
	return null

## INPUTS
{inputs}## ---
## OUTPUTS{outputs}
func main(inputs: Array[IREETensor]) -> IREEResult:
	return run("module.main", inputs)
"""
	with open(f"build/{url}.gd", "w") as f:
		f.write(gdscript_file)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Enter model url")
	parser.add_argument("url", type=str, help="The model url from kaggle")
	args = parser.parse_args()
	path = kaggle_download.model_download(args.url)
	targets = ['llvm-cpu', 'metal-spirv', 'vulkan-spirv']
	print("Building for: ", targets)
	# Find .tflite file
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(".tflite"):
				path = os.path.join(root, file)
				break
	print("Path to model file:", path)
	folder_name = args.url.replace('/', '_').replace('-', '_')
	with open(f"model_name.txt", "w") as f:
		f.write(folder_name)
	os.makedirs(f"build", exist_ok=True)
	# First generate inputs/outputs
	setup_gdscript(path, folder_name, args.url)
	for target in targets:
		print("Compiling for: ", target)
		iree_tflite_compile.compile_file(
			path,
			input_type="tosa",
			output_file=f"build/iree.{target}.vmfb",
			target_backends=[target],
			import_only=False,
		)
		print("Compiled for: ", target)
