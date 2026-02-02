import os
import argparse
from roboflow import Roboflow

def download_dataset(api_key):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("shiv-nibsr").project("indian-bovine-breed-recognition")
    dataset = project.version(1).download("folder")
    
    print(f"Dataset downloaded to: {dataset.location}")
    print("Please move the 'train' and 'valid' folders into the project's 'data/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Bovine Dataset from Roboflow")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API Key")
    args = parser.parse_args()
    
    download_dataset(args.api_key)
