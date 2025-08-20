from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup",
    cache_dir="./hf_cache"
)
