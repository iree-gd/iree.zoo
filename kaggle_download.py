import kagglehub
import argparse

def model_download(url):
    # Download latest version
    path = kagglehub.model_download(url)
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter model url")
    parser.add_argument("url", type=str, help="The model url from kaggle")
    args = parser.parse_args()
    path = model_download(args.url)

    print("Path to model files:", path)
