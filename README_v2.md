## Project Instruction - Development:

### Step 1: Create folders and set permissions

```sh
make pre_setup
```

### Step 2: Download the dataset

2.1. [Embedding data and keys](https://www.kaggle.com/datasets/anhnguynnhtinh/embedding-data)
Download file: `CLIP_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup_clip_embeddings.pt`
Download file: `global2imgpath.json`
Then copy them to `migration_date` folder 


2.2 [Keyframes](https://www.kaggle.com/datasets/anhnguynnhtinh/aic-keyframe-batch-one)
Download all or one of them, then unzip.
Copy all Folders with name format like: L##, eg: L01, L02 to `data_collection/keyframe`

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
