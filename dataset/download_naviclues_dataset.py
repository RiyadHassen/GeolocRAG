from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Naviclues dataset")

    parser.add_argument("--download_path", type=str, required= True)

    args = parser.parse_args()

    download_path = args.download_path
    # Set local storage folder
    LOCAL_DIR = download_path
    os.makedirs(LOCAL_DIR, exist_ok=True)

    REPO = "huggingCode11/NAVICLUES"


    files = list_repo_files(repo_id=REPO, repo_type="dataset")

    print(f"Found {len(files)} files in NAVICLUES repo.")

    for f in files:
        local_path = os.path.join(LOCAL_DIR, f.replace("/", "_"))  # flatten folders
        if not os.path.exists(local_path):
            print(f"Downloading {f} -> {local_path}")
            hf_hub_download(repo_id = REPO, filename= f, repo_type="dataset", local_dir=LOCAL_DIR, local_dir_use_symlinks=False)
        else:
            print(f"Already exists: {local_path}")
