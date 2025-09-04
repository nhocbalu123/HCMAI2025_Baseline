import os
import sys
import types
import torch
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)

beit3_path = os.path.join(os.path.dirname(__file__), "..", "..", "unilm", "beit3")
sys.path.append(os.path.abspath(beit3_path))

from repository.mongo import KeyframeRepository
from repository.milvus import KeyframeVectorRepository
from service import KeyframeQueryService
from models.keyframe import Keyframe
from pymilvus import connections, Collection as MilvusCollection

from torchvision import transforms
from transformers import XLMRobertaTokenizer
import timm
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from app.service.model_service import ModelService

if "torch._six" not in sys.modules:
    six = types.ModuleType("torch._six")
    six.inf = float("inf")
    six.PY3 = True
    six.string_classes = (str,)
    six.int_classes = (int,)
    six.FileNotFoundError = FileNotFoundError
    six.filter = filter
    six.map = map
    six.zip = zip
    sys.modules["torch._six"] = six

import unilm.beit3.modeling_finetune


class Processor():
    def __init__(self, tokenizer):
        self.image_processor = transforms.Compose([
            transforms.Resize((384, 384), interpolation=3),
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
            language_tokens, padding_mask, _ = self.get_text_segment(text, max_len=512)
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
        milvus_port: str,
        milvus_user: str,
        milvus_password: str,
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
        """Remove HuggingFace cache locks if they exist"""
        cache_path = Path(os.getenv("HF_HOME", "~/.cache/huggingface"))
        locks_dir = cache_path / ".locks"
        if locks_dir.exists():
            print("Removing .locks directory...")
            shutil.rmtree(locks_dir)
            print("✓ Removed lock files")

    def _init_beit3_model_service(self, model_name: str = "beit3_large_patch16_384_retrieval"):
        """Refined version of your method"""
        print("--- Start loading BEiT-3 model")

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load tokenizer
            print("--- Loading tokenizer")
            tokenizer_path = "/app/checkpoints/beit3.spm"
            
            if os.path.exists(tokenizer_path):
                tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
                print(f"✓ Loaded tokenizer from {tokenizer_path}")
            else:
                raise("⚠️  /app/checkpoints/beit3.spm tokenizer")
                

            # Create model with timm - THIS IS THE KEY CHANGE
            print(f"--- Creating model with timm: {model_name}")
            
            # Check available models first
            available_models = timm.list_models("beit3*")
            print(f"Available BEiT3 models: {available_models}")
            
            if model_name in available_models:
                print("----in available models")
                model = timm.create_model(
                    model_name=model_name,
                    pretrained=False,
                    num_classes=0,  # For embedding extraction
                    cache_dir="/app/hf_cache"
                )
                print(f"✓ Created model: {model_name}")
            else:
                # Use closest match
                # if available_models:
                #     fallback_model = available_models[0]
                #     print(f"⚠️  {model_name} not found, using {fallback_model}")
                #     model = timm.create_model(
                #         model_name=fallback_model,
                #         pretrained=False,
                #         num_classes=0,
                #     )
                # else:
                raise ValueError("No BEiT3 models available in timm")

            # Load checkpoint
            ckpt_path = "/app/checkpoints/beit3_large_patch16_384_f30k_retrieval.pth"
            
            if os.path.exists(ckpt_path):
                print(f"--- Loading checkpoint from {ckpt_path}")
                state = torch.load(ckpt_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model' in state:
                    state_dict = state['model']
                elif 'state_dict' in state:
                    state_dict = state['state_dict']
                else:
                    state_dict = state
                
                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                if missing:
                    print(f"Missing keys: {len(missing)}")
                if unexpected:
                    print(f"Unexpected keys: {len(unexpected)}")
                    
                print("✓ Checkpoint loaded successfully")
            else:
                print(f"⚠️  Checkpoint not found at {ckpt_path}")
                # print("--- Loading pretrained weights from timm...")
                
                # Reload with pretrained weights
                # model = timm.create_model(
                #     model_name=model_name if model_name in available_models else available_models[0],
                #     pretrained=True,
                #     num_classes=0,
                # )

            # Initialize processor
            processor = Processor(tokenizer)
            
            # Move to device and set eval mode
            model = model.to(device)
            model.eval()

            print("--- BEiT3 model service initialization complete")
            
            return ModelService(
                model=model,
                preprocess=processor,
                tokenizer=tokenizer,
                device=device
            )

        except Exception as e:
            print(f"Error initializing BEiT-3: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_mongo_keyframe_repo(self):
        return self._mongo_keyframe_repo

    def get_milvus_keyframe_repo(self):
        return self._milvus_keyframe_repo

    def get_model_service(self):
        return self._model_service

    def get_keyframe_query_service(self):
        return self._keyframe_query_service
