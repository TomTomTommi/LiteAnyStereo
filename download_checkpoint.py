import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


# Download checkpoints files from huggingface
def download(repo_id, dest_dir, files):
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    for filename in files:
        print("Downloading", filename)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir_path)
    print("Done. Files in", dest_dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Download checkpoint files from Hugging Face.")
    parser.add_argument("--repo_id", default="tomtomtommi/LiteAnyStereoV2", help="Hugging Face repo id")
    parser.add_argument("--dest_dir", default="./checkpoints", help="download destination")
    parser.add_argument("--files", nargs="+", default=["LAS2_S.pth", "LAS2_M.pth", "LAS2_L.pth", "LAS2_H.pth"],
                         help="filenames to download")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    download(repo_id=args.repo_id,
              dest_dir=args.dest_dir,
              files=args.files)