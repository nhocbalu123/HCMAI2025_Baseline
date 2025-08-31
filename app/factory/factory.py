import os
import sys
import torch
from pathlib import Path
import shutil
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from repository.mongo import KeyframeRepository
from repository.milvus import KeyframeVectorRepository
from service import KeyframeQueryService, ModelService
from models.keyframe import Keyframe
import open_clip
from pymilvus import connections, Collection as MilvusCollection
from app.core.settings import AppSettings

# import timm

from torchvision import transforms
from transformers import XLMRobertaTokenizerFast
from timm import create_model
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


class Processor():
    def __init__(self, tokenizer):
        self.image_processor = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        
        self.tokenizer = tokenizer
    
    def process(self, image=None, text=None):
        assert (image is not None) or (text is not None)
        language_tokens = None
        padding_mask = None
        if image is not None:
            image = self.image_processor(image)
            image = image.unsqueeze(0)
        if text is not None:
            language_tokens, padding_mask, _ = self.get_text_segment(text)
        return {'image': image, 'text_description': language_tokens, 'padding_mask': padding_mask}
            
        
    def get_text_segment(self, text, max_len=64):
        tokens = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.tokenizer.bos_token_id] + tokens[:] + [self.tokenizer.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - num_tokens)
        return torch.tensor([language_tokens]),  torch.tensor([padding_mask]), num_tokens


class ServiceFactory:
    def __init__(
        self,
        milvus_collection_name: str,
        milvus_host: str,
        milvus_port: str ,
        milvus_user: str ,
        milvus_password: str ,
        milvus_search_params: dict,
        model_name: str,
        milvus_db_name: str = "default",
        milvus_alias: str = "default",
        mongo_collection=Keyframe,
    ):
        self._mongo_keyframe_repo = KeyframeRepository(collection=mongo_collection)
        self._milvus_keyframe_repo = self._init_milvus_repo(
            search_params=milvus_search_params,
            collection_name=milvus_collection_name,
            host=milvus_host,
            port=milvus_port,
            user=milvus_user,
            password=milvus_password,
            db_name=milvus_db_name,
            alias=milvus_alias
        )

        # self._model_service = self._init_model_service(model_name)
        self._model_service = self._init_beit3_model_service(model_name)

        self._keyframe_query_service = KeyframeQueryService(
            keyframe_mongo_repo=self._mongo_keyframe_repo,
            keyframe_vector_repo=self._milvus_keyframe_repo
        )

    def _init_milvus_repo(
        self,
        search_params: dict,
        collection_name: str,
        host: str,
        port: str,
        user: str,
        password: str,
        db_name: str = "default",
        alias: str = "default"
    ):
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
        collection = MilvusCollection(collection_name, using=alias)

        return KeyframeVectorRepository(collection=collection, search_params=search_params)
    
    def _remove_hf_cache_locked(self):
        cache_path = Path(os.getenv("HF_HOME"))
        locks_dir = cache_path / ".locks"
        if locks_dir.exists():
            print("Removing .locks directory...")
            shutil.rmtree(locks_dir)
            print("✓ Removed lock files")

    def _init_model_service(self, model_name: str):
        print("---Start loading model")
        
        # self._remove_hf_cache_locked()

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            cache_dir=os.getenv("HF_HOME")
        )
        print("---Start loading tokenizer")
        tokenizer = open_clip.get_tokenizer(model_name)
        print("---End loading all")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ModelService(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)
    
    def _init_beit3_model_service(self, model_name: str = "beit3_large_patch16_224"):
        print("--- Start loading BEiT-3 model")

        try:
            # pick device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # tokenizer (BEiT-3 text pathway uses SentencePiece/BPE like XLM-R)
            print("--- Start loading tokenizer")
            tokenizer = XLMRobertaTokenizerFast.from_pretrained("/app/checkpoints/beit3.spm")

            # load checkpoint
            ckpt_path = os.path.join("app", "checkpoints", f"{model_name}.pth")
            if os.path.exists(ckpt_path):
                print(f"--- Loading checkpoint from {ckpt_path}")
                state = torch.load(ckpt_path, map_location=device)
                model = create_model(model_name)
                missing, unexpected = model.load_state_dict(state['model'], strict=False)

                if missing:
                    print(f"Missing keys: {len(missing)}")
                if unexpected:
                    print(f"Unexpected keys: {len(unexpected)}")
            else:
                print(f"!!! Warning: checkpoint not found at {ckpt_path}, using random init.")

            processor = Processor(tokenizer)

            print("--- End loading all")
            return ModelService(model=model, preprocess=processor, tokenizer=tokenizer, device=device)

        except Exception as e:
            print("Error initializing BEiT-3:", e)
            raise

    def get_mongo_keyframe_repo(self):
        return self._mongo_keyframe_repo

    def get_milvus_keyframe_repo(self):
        return self._milvus_keyframe_repo

    def get_model_service(self):
        return self._model_service

    def get_keyframe_query_service(self):
        return self._keyframe_query_service
