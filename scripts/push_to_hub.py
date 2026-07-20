#!/usr/bin/env python3
"""Standalone push of a trained adapter/model dir to the Hugging Face Hub.

Usage: python scripts/push_to_hub.py <local_dir> <repo_id>
Reads HF_TOKEN from .env (never hard-code tokens). The notebooks push
automatically after training; this script exists for retries after network
failures without re-running training.
"""
import sys, os
from dotenv import load_dotenv
from huggingface_hub import HfApi

def main(local_dir, repo_id):
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN missing — copy .env.example to .env and fill it in"
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=False)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id)
    print(f"pushed {local_dir} -> https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
