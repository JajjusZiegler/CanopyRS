#!/usr/bin/env python3
"""
CanopyRS — SAM3 HuggingFace login + weight download
Usage:  HF_TOKEN=hf_xxx conda run -n canopyrs_env python setup_sam3_login.py
"""
import os
import sys

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    # Fall back to interactive prompt if running in a real TTY
    try:
        import getpass
        token = getpass.getpass("HuggingFace token: ").strip()
    except Exception:
        sys.exit("ERROR: Set HF_TOKEN env var or run in an interactive terminal.")

if not token:
    sys.exit("ERROR: No token provided.")

# ── Login ─────────────────────────────────────────────────────────────────────
from huggingface_hub import login, model_info, snapshot_download, HfApi

print("Logging in to HuggingFace...")
login(token=token, add_to_git_credential=False)
print("  Login OK")

# ── Verify access ─────────────────────────────────────────────────────────────
MODEL_ID = "facebook/sam3"
print(f"Checking access to {MODEL_ID}...")
try:
    info = model_info(MODEL_ID)
    print(f"  Access GRANTED (modelId={info.modelId})")
except Exception as e:
    err = str(e)
    if "403" in err or "gated" in err.lower() or "access" in err.lower():
        print(f"\n  Access DENIED.")
        print(f"  Request access at: https://huggingface.co/facebook/sam3")
        print(f"  Accept Meta's licence, then re-run this script.")
        sys.exit(1)
    else:
        print(f"  Unexpected error: {err}")
        sys.exit(1)

# ── Download weights ──────────────────────────────────────────────────────────
print(f"Downloading {MODEL_ID} weights (~5-10 GB, please wait)...")
path = snapshot_download(repo_id=MODEL_ID)
print(f"  Cached at: {path}")

# ── Verify load ───────────────────────────────────────────────────────────────
print("Verifying SAM3 can be loaded by transformers...")
from transformers import Sam3TrackerProcessor, Sam3TrackerModel
proc = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
print("  Processor: OK")
# Load model to CPU only for verification (no GPU needed here)
model = Sam3TrackerModel.from_pretrained(MODEL_ID)
print("  Model: OK")
del model, proc

api = HfApi()
user = api.whoami()
print(f"\n  Logged in as : {user['name']}")
print("\nSAM3 setup complete! You can now run inference with SAM3 presets.")
