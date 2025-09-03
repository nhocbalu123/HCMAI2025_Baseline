## Project Instruction - Development:

### Step 1: Create folders and set permissions

```sh
make pre_setup
```

### Step 2: Download the dataset

2.1. Copy unilm into root project
[Project Link](https://github.com/microsoft/unilm/tree/master)

2.2. Download pretrained weights for retrieval
[beit3_large_itc_patch16_224](https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_f30k_retrieval.pth)

2.4. Download beit3.spm tokenizer

[beit3.spm](https://github.com/addf400/files/releases/download/beit3/beit3.spm)

Then copy pretrained weights and tokenizer to **/checkpoints**

2.3. Download all keyframe embeddings:

[embeddings](https://pub-6dc786c2b53e460d9ef9948fd14a8a9a.r2.dev/combined_embeddings/combined_30082025-1756585172_embeddings.npy)
[embeddings_metadata](https://pub-6dc786c2b53e460d9ef9948fd14a8a9a.r2.dev/combined_embeddings/combined_30082025-1756585172_metadata.npy)

Then copy them into **/migration_data**

**Directory Tree**

```
.
├── Dockerfile
├── Makefile
├── README.md
├── README_v2.md
├── README_v3.md
├── app
├── app_entrypoint.sh
├── checkpoints
├── dev_setup.sh
├── docker-compose.yml
├── gui
├── hf_cache
├── logs
├── migration
├── migration_data
├── pyproject.toml
├── unilm
└── uv.lock
```

### Step 3: Build base image

```sh
make build
```

### Step 4: Start docker compose in detached mode

```sh
make up

# or
docker compose up -d
```

The progress will be:

+ The `app_migration` service will make `id2index.json`, do migration and download the model for embedding.
At the first time, this process will take a lot of time.
About 2 to 4 minutes for indexing embedding.
About 10 to 20 minutes depends on your network for downloading open-clip model and prepare preprocessing pipeline.

+ The `app` service will run as backend for `app_gui` service.
Since this service will load the model into open-clip, so it will take about 5 minutes to start to loading the model and tokenizer.
Trying to find the way to make it more efficient for reducing start-up time

+ The `.migration_locked` in `migration` folder is used for making sure that the `app_migration` run once. Remove it to run all `app_migration` again.

### Stop and remove all docker volumes:

If you do clean up all stuffs, you must remove `migration/.migration_locked` so that it will start again.

The downloaded model won't start again since it checks for cache dir before downloading.

```sh
make quick_clean
```
