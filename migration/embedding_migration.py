import torch
import numpy as np
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from typing import Optional
from tqdm import tqdm
import argparse

import sys
import os
ROOT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_FOLDER)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from app.core.settings import KeyFrameIndexMilvusSetting


class MilvusEmbeddingInjector:
    def __init__(
        self,
        setting: KeyFrameIndexMilvusSetting,
        collection_name: str,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        db_name: str = "default",
        alias: str = "default"
        
    ):
        self.setting = setting
        self.collection_name = collection_name
        self.alias = alias
        
        self._connect(host, port, user, password, db_name, alias)
        
    def _connect(self, host: str, port: str, user: str, password: str, db_name: str, alias: str):
        
        if connections.has_connection(alias):
            connections.remove_connection(alias)
        
        conn_params = {
            "host": host,
            "port": port,
            "db_name": db_name
        }
        
        if user and password:
            conn_params["user"] = user
            conn_params["password"] = password
        
        connections.connect(alias=alias, **conn_params)
        print(f"Connected to Milvus at {host}:{port}")
        
    
    
    def create_collection(self, embedding_dim: int, index_params: Optional[dict] = None):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ]
        
        schema = CollectionSchema(fields, f"Collection for {self.collection_name} embeddings")
        
        collection = Collection(self.collection_name, schema, using=self.alias)
        print(f"Created collection '{self.collection_name}' with dimension {embedding_dim}")
        
        if index_params is None:
            index_params = {
                "metric_type": self.setting.METRIC_TYPE,
                "index_type": self.setting.INDEX_TYPE,
            }
        
        collection.create_index("embedding", index_params)
        print("Created index for embedding field")
        
        return collection
    
    def inject_embeddings(
        self, 
        embedding_file_path: str, 
        batch_size: int = 10000,
    ):
        print(f"Loading embeddings from {embedding_file_path}")
        embeddings = torch.load(
            embedding_file_path,
            weights_only=False,
            map_location=DEVICE
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        num_vectors, embedding_dim = embeddings.shape
        print(f"Loaded {num_vectors} embeddings with dimension {embedding_dim}")
        
    
        
        if utility.has_collection(self.collection_name, using=self.alias):
            print(f"Dropping existing collection '{self.collection_name}' before creation...")
            utility.drop_collection(self.collection_name, using=self.alias)

        collection = self.create_collection(embedding_dim)
     
      
        
        print(f"Inserting {num_vectors} embeddings in batches of {batch_size}")
        
        for i in tqdm(range(0, num_vectors, batch_size), desc="Inserting batches"):
            end_idx = min(i + batch_size, num_vectors)
            batch_embeddings = embeddings[i:end_idx].tolist()

            batch_ids = list(range(i, end_idx))
            entities = [batch_ids, batch_embeddings]
            collection.insert(entities)
        
        collection.flush()
        print("Data flushed to disk")
        
        collection.load()
        print("Collection loaded for search")
        
        return collection
    

    def inject_embeddings_with_metadata(
        self,
        embedding_file_path: str,
        metadata_file_path: str,
        batch_size: int = 10000,
    ):
        """
        Inject embeddings + metadata into Milvus.
        Optimized to build NumPy arrays in a single pass.
        """
        print(f"Loading embeddings from {embedding_file_path}")
        embeddings = np.load(embedding_file_path)
        metadata = np.load(metadata_file_path, allow_pickle=True).tolist()

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        num_vectors, embedding_dim = embeddings.shape
        print(f"Loaded {num_vectors} embeddings with dimension {embedding_dim}")

        # Drop existing collection if it exists
        if utility.has_collection(self.collection_name, using=self.alias):
            print(f"Dropping existing collection '{self.collection_name}' before creation...")
            utility.drop_collection(self.collection_name, using=self.alias)

        # Create collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="parent_namespace", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="video_namespace", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="frame_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="frame_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="video_frame_index", dtype=DataType.INT64),
            FieldSchema(name="global_index", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, f"Collection for {self.collection_name} embeddings")
        collection = Collection(self.collection_name, schema, using=self.alias)

        index_params = {
            "metric_type": self.setting.METRIC_TYPE,
            "index_type": self.setting.INDEX_TYPE,
        }
        collection.create_index("embedding", index_params)

        print(f"Inserting {num_vectors} embeddings + metadata in batches of {batch_size}")

        # === Single-pass extraction of metadata ===
        ids, parent, video, frame_id, frame_path, video_index, global_index = zip(*[
            (
                m["global_index"],
                m["parent_namespace"],
                m["video_namespace"],
                m["frame_id"],
                m["frame_path"],
                m["video_frame_index"],
                m["global_index"],
            )
            for m in metadata
        ])

        all_ids = np.array(ids, dtype=np.int64)
        all_parent = np.array(parent)
        all_video = np.array(video)
        all_frame_id = np.array(frame_id)
        all_frame_path = np.array(frame_path)
        all_video_index = np.array(video_index, dtype=np.int64)
        all_global_index = np.array(global_index, dtype=np.int64)

        # === Insert into Milvus in batches ===
        print(f"Inserting {num_vectors} embeddings in batches of {batch_size}")

        for i in tqdm(range(0, num_vectors, batch_size), desc="Inserting batches"):
            end_idx = min(i + batch_size, num_vectors)

            batch_embeddings = embeddings[i:end_idx].tolist()
            entities = [
                all_ids[i:end_idx].tolist(),
                batch_embeddings,
                all_parent[i:end_idx].tolist(),
                all_video[i:end_idx].tolist(),
                all_frame_id[i:end_idx].tolist(),
                all_frame_path[i:end_idx].tolist(),
                all_video_index[i:end_idx].tolist(),
                all_global_index[i:end_idx].tolist(),
            ]
            collection.insert(entities)

        collection.flush()
        print("✅ Data flushed to disk")

        collection.load()
        print("✅ Collection loaded for search")

        return collection

    
    def get_collection_info(self):
        
        collection = Collection(self.collection_name, using=self.alias)
        num_entities = collection.num_entities
        print(f"Collection '{self.collection_name}' has {num_entities} entities")
        return num_entities
      
    
    def disconnect(self):
        if connections.has_connection(self.alias):
            connections.remove_connection(self.alias)
            print("Disconnected from Milvus")


def inject_embeddings_simple(
    embedding_file_path: str,
    setting: KeyFrameIndexMilvusSetting
):
    
    print(setting.HOST)
    
    injector = MilvusEmbeddingInjector(
        setting=setting,
        collection_name=setting.COLLECTION_NAME,
        host=setting.HOST,
        port=setting.PORT
    )
    

    injector.inject_embeddings(
        embedding_file_path=embedding_file_path,
        batch_size=setting.BATCH_SIZE
    )
    count = injector.get_collection_info()
    print(f"Successfully injected embeddings! Total entities: {count}")


def inject_embeddings_with_cutom_metadata(
    embedding_file_path: str,
    metadata_file_path: str,
    setting: KeyFrameIndexMilvusSetting
):
    
    print(setting.HOST)
    
    injector = MilvusEmbeddingInjector(
        setting=setting,
        collection_name=setting.COLLECTION_NAME,
        host=setting.HOST,
        port=setting.PORT
    )
    

    injector.inject_embeddings_with_metadata(
        embedding_file_path=embedding_file_path,
        metadata_file_path=metadata_file_path,
        batch_size=setting.BATCH_SIZE
    )

    count = injector.get_collection_info()
    print(f"Successfully injected embeddings! Total entities: {count}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Migrate embedding to Milvus.")
    parser.add_argument(
        "--emd-file-path", type=str, help="Path to embedding pt."
    )
    parser.add_argument(
        "--metadata-file-path", type=str, help="Path to embedding pt."
    )
    args = parser.parse_args()

    setting = KeyFrameIndexMilvusSetting()

    # inject_embeddings_simple(
    #     embedding_file_path=args.file_path,
    #     setting=setting
    # )

    inject_embeddings_with_cutom_metadata(
        embedding_file_path=args.emd_file_path,
        metadata_file_path=args.metadata_file_path,
        setting=setting
    )
