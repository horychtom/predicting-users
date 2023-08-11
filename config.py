import json
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="local.env")

# GCP BUCKET
with open("gcp-service-account.json") as json_file:
    GCP_STORAGE_SERVICE_ACCOUNT = json.load(json_file)

GCP_STORAGE_SERVICE_ACCOUNT["private_key"] = os.getenv(
    "GCP_PRIV", "").replace("\\n", "\n")
GCP_BUCKET_NAME = os.getenv("BUCKET_NAME")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
