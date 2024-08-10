import argparse
import kaggle_download
import iree.compiler.tflite as iree_tflite_compile
import iree.compiler
import iree.runtime as iree_rt
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter model url")
    parser.add_argument("url", type=str, help="The model url from kaggle")
    args = parser.parse_args()
    path = kaggle_download.model_download(args.url)
    targets = iree.compiler.query_available_targets()
    print("Building for: ", targets)
    # Find .tflite file
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tflite"):
                path = os.path.join(root, file)
                break
    print("Path to model file:", path)
    
    for target in targets:
        iree_tflite_compile.compile_file(
            path,
            output_file=f"iree.{target}.vmfb",
            target_backends=[target],
            import_only=False,
        )
