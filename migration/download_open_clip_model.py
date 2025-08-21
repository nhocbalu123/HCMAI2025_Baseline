from huggingface_hub import snapshot_download
from app.core.settings import AppSettings
import os


app_setting = AppSettings()

# Download the exact same repository that hf-hub: format uses
print("REPO ID: ", app_setting.MODEL_NAME.rsplit(':', maxsplit=1)[-1])
snapshot_download(
    repo_id=app_setting.MODEL_NAME.rsplit(':', maxsplit=1)[-1],
    cache_dir=os.getenv("HF_HOME")
)

print("✓ Pre-download completed!")
