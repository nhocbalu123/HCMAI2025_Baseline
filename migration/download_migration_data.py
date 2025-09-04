import os
import time
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from minio import Minio


class MigartionDataDownloader:
    """
    Utility to pull per-video embedding + metadata files from Cloudflare R2
    and combine them into a single embedding matrix and aligned metadata list.
    """

    def __init__(self, local_dir: str, pattern: str = "_embeddings.npy",
                 cloudflare_s3_url: str = None,
                 cloudflare_access_key: str = None,
                 cloudflare_secret_key: str = None,
                 bucket: str = None):
        """
        Args:
            local_dir: Directory containing *_embeddings.npy and *_metadata.npy files
            pattern: Suffix for embedding files (default: "_embeddings.npy")
            cloudflare_s3_url: Cloudflare R2 endpoint (e.g., "https://<accountid>.r2.cloudflarestorage.com")
            cloudflare_access_key: R2 Access Key
            cloudflare_secret_key: R2 Secret Key
            bucket: R2 bucket name
        """
        self.local_dir = local_dir
        self.pattern = pattern
        self.embeddings_matrix: np.ndarray = np.array([])
        self.metadata_list: List[Dict] = []
        self.bucket = bucket

        os.makedirs(self.local_dir, exist_ok=True)

        # Init Minio client if credentials provided
        self._s3_client = None
        if all([cloudflare_s3_url, cloudflare_access_key, cloudflare_secret_key]):
            self._s3_client = Minio(
                endpoint=cloudflare_s3_url.replace("https://", ""),
                access_key=cloudflare_access_key,
                secret_key=cloudflare_secret_key,
                secure=True
            )

    def pull_from_r2(self, namespace: str) -> None:
        """
        Pull all *_embeddings.npy and *_metadata.npy files from a given R2 namespace.
        Files will be downloaded into self.local_dir.
        """
        if not self._s3_client:
            raise RuntimeError("Minio client not initialized. Provide credentials in __init__().")

        objects = self._s3_client.list_objects(self.bucket, prefix=namespace, recursive=True)

        for obj in objects:
            if obj.object_name.endswith("_embeddings.npy") or obj.object_name.endswith("_metadata.npy"):
                local_path = os.path.join(self.local_dir, os.path.basename(obj.object_name))
                if not os.path.exists(local_path):  # avoid re-download
                    self._s3_client.fget_object(self.bucket, obj.object_name, local_path)
                    print(f"⬇️ Downloaded {obj.object_name} -> {local_path}")


if __name__ == "__main__":
    print(os.environ.get("CLOUDFLARE_S3_URL"))
    print(os.getenv("CLOUDFLARE_S3_URL", "==="))
    combiner = MigartionDataDownloader(
        local_dir="migration_data",
        cloudflare_s3_url=os.getenv("CLOUDFLARE_S3_URL"),
        cloudflare_access_key=os.getenv("CLOUDFLARE_ACCESS_KEY_ID"),
        cloudflare_secret_key=os.getenv("CLOUDFLARE_SECRET_KEY"),
        bucket=os.getenv("BUCKET_NAME")
    )

    # 1. Pull all per-video embeddings + metadata from namespace
    combiner.pull_from_r2(namespace="combined_embeddings/")
