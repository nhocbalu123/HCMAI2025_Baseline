import timm
import os
import sys
import torch
from app.factory.factory import Processor
from app.service.model_service import ModelService
from transformers import XLMRobertaTokenizer
import io
import time
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from minio import Minio
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import concurrent.futures
from dataclasses import dataclass
import logging
import re
from pathlib import Path


@contextmanager
def timeit_context(name: str = "Execution"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"[{name}] finished in {elapsed:.4f} seconds")


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class FrameEmbeddingResult:
    """Result container for frame embedding processing"""
    parent_namespace: str
    video_namespace: str
    embeddings: np.ndarray
    frame_ids: List[str]  # Frame IDs extracted from filenames (e.g., ['000000', '000053'])
    frame_paths: List[str]  # Full paths for reference
    frame_count: int
    success: bool
    error_message: Optional[str] = None


class CloudflareSettings:
    CLOUDFLARE_ACCESS_KEY_ID = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
    CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
    CLOUDFLARE_R2_ACC_ID = os.getenv("CLOUDFLARE_R2_ACC_ID")
    CLOUDFLARE_S3_URL = os.getenv("CLOUDFLARE_S3_URL")
    CLOUDFLARE_SECRET_KEY = os.getenv("CLOUDFLARE_SECRET_KEY")
    BUCKET_NAME = "demo-s3"


class OptimizedFrameEmbeddingProcessor:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.logger = self._setup_logging()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model_service = self._init_beit3_model_service()
        # Initialize S3 client
        self._s3_client = Minio(
            CloudflareSettings.CLOUDFLARE_S3_URL,
            access_key=CloudflareSettings.CLOUDFLARE_ACCESS_KEY_ID,
            secret_key=CloudflareSettings.CLOUDFLARE_SECRET_KEY
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def _init_beit3_model_service(self, model_name: str = "beit3_large_patch16_384_retrieval"):
        """Refined version of your method"""
        print("--- Start loading BEiT-3 model")

        try:
            # Load tokenizer
            print("--- Loading tokenizer")
            tokenizer_path = "./checkpoints/beit3.spm"
            
            if os.path.exists(tokenizer_path):
                tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
                print(f"✓ Loaded tokenizer from {tokenizer_path}")
            else:
                raise "⚠️  /app/checkpoints/beit3.spm tokenizer"
                
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
                    cache_dir=os.getenv("HF_HOME")
                )
                print(f"✓ Created model: {model_name}")
            else:
                raise ValueError("No BEiT3 models available in timm")

            # Load checkpoint
            ckpt_path = "./checkpoints/beit3_large_patch16_384_f30k_retrieval.pth"
            
            if os.path.exists(ckpt_path):
                print(f"--- Loading checkpoint from {ckpt_path}")
                state = torch.load(ckpt_path, map_location=self._device)
                
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
                raise f"⚠️  Checkpoint not found at {ckpt_path}"

            # Initialize processor
            processor = Processor(tokenizer)
            
            # Move to device and set eval mode
            model = model.to(self._device)
            model.eval()

            print("--- BEiT3 model service initialization complete")
            
            return ModelService(
                model=model,
                preprocess=processor,
                tokenizer=tokenizer,
                device=self._device
            )

        except Exception as e:
            print(f"Error initializing BEiT-3: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_frame_id(self, frame_path: str) -> str:
        """Extract frame ID from path like 'demo-s3/L21_a/L21_V001/000000.webp' -> '000000'"""
        filename = Path(frame_path).stem  # Gets '000000' from '000000.webp'
        return filename
    
    def get_ordered_frame_paths(self, bucket_name: str, parent_namespace: str, video_namespace: str) -> List[str]:
        """Get all frame paths for a video in numerical order"""
        try:
            prefix = f"{parent_namespace}/{video_namespace}/"
            objects = list(self._s3_client.list_objects(bucket_name, prefix=prefix, recursive=True))
            
            frame_paths = [obj.object_name for obj in objects]
            
            # Sort by numerical frame ID to maintain temporal order
            def get_sort_key(path):
                frame_id = self.extract_frame_id(path)
                # Extract numeric part for proper sorting (handles 000000, 000053, etc.)
                try:
                    return int(frame_id)
                except ValueError:
                    return 0
            
            frame_paths.sort(key=get_sort_key)
            return frame_paths
            
        except Exception as e:
            self.logger.error(f"Error listing frames for {parent_namespace}/{video_namespace}: {e}")
            return []
        
    def load_frame_batch_ordered(self, bucket_name: str, frame_paths: List[str]) -> Tuple[List[torch.Tensor], List[str], List[str]]:
        """Load and preprocess a batch of frames in order, returning tensors, frame_ids, and valid paths"""
        def load_single_frame(frame_path: str) -> Tuple[Optional[torch.Tensor], str, str]:
            try:
                self.logger.info(f"Processing frame from {frame_path}")
                response = self._s3_client.get_object(bucket_name, frame_path)
                image_data = response.read()
                
                # Fast image loading and preprocessing
                image = Image.open(io.BytesIO(image_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                inputs = self._model_service.preprocess.process(
                    image=image,
                    text=None
                )
            
                # Move to device
                image_tensor = inputs['image'].to(self._device)
                image_tensor = image_tensor.squeeze(0) # (C, H, W)
                print(f"image_tensor shape: {image_tensor.shape}")
                frame_id = self.extract_frame_id(frame_path)
                
                return image_tensor, frame_id, frame_path
                
            except Exception as e:
                self.logger.warning(f"Failed to load {frame_path}: {e}")
                return None, "", ""
        
        # Load frames in parallel but preserve order
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks in order
            futures = [executor.submit(load_single_frame, path) for path in frame_paths]
            results = [future.result() for future in futures]
        
        # Filter successful loads while maintaining order
        valid_tensors = []
        valid_frame_ids = []
        valid_paths = []
        
        for tensor, frame_id, path in results:
            if tensor is not None:
                valid_tensors.append(tensor)
                valid_frame_ids.append(frame_id)
                valid_paths.append(path)
        
        return valid_tensors, valid_frame_ids, valid_paths

    def generate_frame_embeddings_batch(self, frame_tensors: List[torch.Tensor]) -> np.ndarray:
        """Generate L2-normalized embeddings for a batch of frame tensors (following your pattern)"""
        if not frame_tensors:
            return np.array([])
        
        # Stack tensors and move to device
        batch_tensor = torch.stack(frame_tensors).to(self._device)
        print(f"generate_frame_embeddings_batch - {batch_tensor.shape}")
        with torch.no_grad():
            # TIMM models with num_classes=0 return features directly
            embeddings = self._model_service.encode_batch_images(batch_tensor)
            
            # L2 normalize (following your pattern)
            # features = features / features.norm(dim=-1, keepdim=True)
            # embeddings = features.cpu().numpy()
            
            return embeddings

    def process_video_frames(self, 
        bucket_name: str, 
        parent_namespace: str, 
        video_namespace: str,
        max_frames: Optional[int] = None,
        stride: int = 1) -> FrameEmbeddingResult:
        """
        Process all frames in a video with order preservation
        
        Args:
            max_frames: Maximum frames to process (None = all frames)
            stride: Take every Nth frame (1 = all frames, 2 = every other frame)
        """
        self.logger.info(f"Processing frames for {parent_namespace}/{video_namespace}")
        
        try:
            # Get ordered frame paths
            frame_paths = self.get_ordered_frame_paths(bucket_name, parent_namespace, video_namespace)
            
            if not frame_paths:
                return FrameEmbeddingResult(
                    parent_namespace=parent_namespace,
                    video_namespace=video_namespace,
                    embeddings=np.array([]),
                    frame_ids=[],
                    frame_paths=[],
                    frame_count=0,
                    success=False,
                    error_message="No frames found"
                )
            
            # Apply stride sampling while maintaining order
            if stride > 1:
                frame_paths = frame_paths[::stride]
            
            # Apply max_frames limit
            if max_frames:
                frame_paths = frame_paths[:max_frames]
            
            self.logger.info(f"Processing {len(frame_paths)} frames in batches of {self.batch_size}")
            
            all_embeddings = []
            all_frame_ids = []
            all_frame_paths = []
            
            # Process in batches while maintaining order
            for i in range(0, len(frame_paths), self.batch_size):
                batch_paths = frame_paths[i:i + self.batch_size]
                
                # Load batch with order preservation
                batch_tensors, batch_frame_ids, batch_valid_paths = self.load_frame_batch_ordered(
                    bucket_name, batch_paths
                )
                
                if not batch_tensors:
                    continue
                
                # Generate embeddings
                batch_embeddings = self.generate_frame_embeddings_batch(batch_tensors)
                
                # Store results maintaining order
                all_embeddings.append(batch_embeddings)
                all_frame_ids.extend(batch_frame_ids)
                all_frame_paths.extend(batch_valid_paths)
                
                # Memory cleanup
                del batch_tensors
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                self.logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(frame_paths)-1)//self.batch_size + 1}")
            
            if not all_embeddings:
                return FrameEmbeddingResult(
                    parent_namespace=parent_namespace,
                    video_namespace=video_namespace,
                    embeddings=np.array([]),
                    frame_ids=[],
                    frame_paths=[],
                    frame_count=0,
                    success=False,
                    error_message="No valid embeddings generated"
                )
            
            # Concatenate all embeddings (maintains order)
            final_embeddings = np.vstack(all_embeddings)
            
            self.logger.info(f"Generated {final_embeddings.shape[0]} embeddings of dimension {final_embeddings.shape[1]}")
            self.logger.info(f"Frame ID range: {all_frame_ids[0]} to {all_frame_ids[-1]}")
            
            return FrameEmbeddingResult(
                parent_namespace=parent_namespace,
                video_namespace=video_namespace,
                embeddings=final_embeddings,
                frame_ids=all_frame_ids,
                frame_paths=all_frame_paths,
                frame_count=final_embeddings.shape[0],
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {parent_namespace}/{video_namespace}: {e}")
            return FrameEmbeddingResult(
                parent_namespace=parent_namespace,
                video_namespace=video_namespace,
                embeddings=np.array([]),
                frame_ids=[],
                frame_paths=[],
                frame_count=0,
                success=False,
                error_message=str(e)
            )
        
    def get_all_video_namespaces(self, bucket_name: str, parent_namespace: str) -> List[str]:
        """Get all video namespaces under a parent"""
        try:
            objects = self._s3_client.list_objects(bucket_name, prefix=f"{parent_namespace}/", recursive=True)
            
            video_namespaces = set()
            for obj in objects:
                path_parts = obj.object_name.split('/')
                if len(path_parts) >= 2:
                    video_namespaces.add(path_parts[1])
            
            return sorted(list(video_namespaces))
            
        except Exception as e:
            self.logger.error(f"Error listing videos in {parent_namespace}: {e}")
            return []
    
    def process_multiple_videos(self, 
        bucket_name: str, 
        parent_namespace: str,
        video_namespaces: Optional[List[str]] = None,
        max_frames_per_video: Optional[int] = None,
        stride: int = 1) -> Dict[str, FrameEmbeddingResult]:
        """Process multiple videos or all videos under parent namespace"""
        
        if video_namespaces is None:
            video_namespaces = self.get_all_video_namespaces(bucket_name, parent_namespace)
        
        self.logger.info(f"Processing {len(video_namespaces)} videos under {parent_namespace}")
        self.logger.info(f"Videos: {video_namespaces}")
        
        results = {}
        for i, video_namespace in enumerate(video_namespaces, 1):
            self.logger.info(f"Processing video {i}/{len(video_namespaces)}: {video_namespace}")
            
            result = self.process_video_frames(
                bucket_name, 
                parent_namespace, 
                video_namespace,
                max_frames=max_frames_per_video,
                stride=stride
            )
            
            results[video_namespace] = result
            
            if result.success:
                self.logger.info(f"✅ {video_namespace}: {result.frame_count} embeddings, frames {result.frame_ids[0]}-{result.frame_ids[-1]}")
            else:
                self.logger.warning(f"❌ {video_namespace}: {result.error_message}")
        
        return results

    def save_embeddings_for_vector_db(self, results: Dict[str, FrameEmbeddingResult], output_dir: str):
        """Save embeddings optimized for vector databases (FAISS/Milvus)"""
        os.makedirs(output_dir, exist_ok=True)
        
        for video_namespace, result in results.items():
            if not result.success or result.embeddings.size == 0:
                continue
            
            # Create video-specific filename
            base_filename = f"{result.parent_namespace}_{video_namespace}"
            
            # Save embeddings as .npy (fastest loading for vector DBs)
            embeddings_file = os.path.join(output_dir, f"{base_filename}_embeddings.npy")
            np.save(embeddings_file, result.embeddings.astype(np.float32))  # float32 for efficiency
            
            # Save metadata as .npy for easy loading
            metadata_file = os.path.join(output_dir, f"{base_filename}_metadata.npy")
            metadata = {
                'parent_namespace': result.parent_namespace,
                'video_namespace': result.video_namespace,
                'frame_ids': result.frame_ids,
                'frame_paths': result.frame_paths,
                'frame_count': result.frame_count,
                'embedding_dim': result.embeddings.shape[1]
            }
            np.save(metadata_file, metadata)
            
            self.logger.info(f"Saved {video_namespace}: {embeddings_file}")
            self.logger.info(f"  Shape: {result.embeddings.shape}")
            self.logger.info(f"  Frame range: {result.frame_ids[0]} to {result.frame_ids[-1]}")


class CloudflareEmbeddingUploader:
    """Upload saved embedding files into Cloudflare R2 namespace"""
    
    def __init__(self, target_namespace: str = "beit3_retrieval_embeddings_for_vectordb"):
        """
        Args:
            target_namespace: Namespace/folder inside bucket to store embeddings
        """
        self._s3_client = Minio(
            CloudflareSettings.CLOUDFLARE_S3_URL,
            access_key=CloudflareSettings.CLOUDFLARE_ACCESS_KEY_ID,
            secret_key=CloudflareSettings.CLOUDFLARE_SECRET_KEY
        )
        self.bucket_name = CloudflareSettings.BUCKET_NAME
        self.target_namespace = target_namespace.rstrip("/")
        
        # Logger
        logging.basicConfig(level=logging.INFO, 
                          format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def upload_file(self, local_path: str, remote_filename: Optional[str] = None):
        """Upload a single file to Cloudflare R2"""
        if not os.path.exists(local_path):
            self.logger.error(f"File not found: {local_path}")
            return False
        
        if remote_filename is None:
            remote_filename = os.path.basename(local_path)
        
        object_name = f"{self.target_namespace}/{remote_filename}"
        
        try:
            self._s3_client.fput_object(
                self.bucket_name,
                object_name,
                local_path
            )
            self.logger.info(f"✅ Uploaded {local_path} -> {object_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Upload failed for {local_path}: {e}")
            return False

    def upload_directory(self, local_dir: str, pattern: str = "_embeddings.npy"):
        """
        Upload all embedding and metadata files in local_dir
        Args:
            local_dir: Directory containing saved npy files
            pattern: Substring to match embedding files (default: '_embeddings.npy')
        """
        uploaded = 0
        failed = 0
        for file in os.listdir(local_dir):
            if file.endswith(".npy") and pattern in file:
                local_path = os.path.join(local_dir, file)
                
                if self.upload_file(local_path):
                    uploaded += 1
                else:
                    failed += 1
                
                # Upload corresponding metadata file
                metadata_file = file.replace("_embeddings.npy", "_metadata.npy")
                metadata_path = os.path.join(local_dir, metadata_file)
                if os.path.exists(metadata_path):
                    self.upload_file(metadata_path)
        
        self.logger.info(f"=== Upload complete: {uploaded} succeeded, {failed} failed ===")

    def list_remote_files(self, file_pattern: str = None) -> List[str]:
        """
        List files in the remote namespace
        Args:
            file_pattern: Optional pattern to filter files
        Returns:
            List of remote file names
        """
        try:
            objects = self._s3_client.list_objects(
                self.bucket_name, 
                prefix=f"{self.target_namespace}/",
                recursive=True
            )
            
            files = []
            for obj in objects:
                # Remove the namespace prefix to get just the filename
                filename = obj.object_name.replace(f"{self.target_namespace}/", "")
                if file_pattern is None or file_pattern in filename:
                    files.append(filename)
            
            return sorted(files)
        except Exception as e:
            self.logger.error(f"❌ Failed to list remote files: {e}")
            return []
    
    def load_embeddings_from_remote(self, remote_filename: str) -> Optional[np.ndarray]:
        """
        Load embeddings array from remote storage
        Args:
            remote_filename: Name of the embeddings file (without namespace prefix)
        Returns:
            numpy array or None if failed
        """
        object_name = f"{self.target_namespace}/{remote_filename}"
        
        try:
            response = self._s3_client.get_object(self.bucket_name, object_name)
            
            # Read the data into a BytesIO buffer
            data = response.read()
            buffer = io.BytesIO(data)
            
            # Load numpy array from buffer
            embeddings = np.load(buffer)
            
            self.logger.info(f"✅ Loaded embeddings from {object_name}, shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load embeddings from {object_name}: {e}")
            return None
        finally:
            if 'response' in locals():
                response.close()
    
    def load_metadata_from_remote(self, remote_filename: str) -> Optional[np.ndarray]:
        """
        Load metadata array from remote storage
        Args:
            remote_filename: Name of the metadata file (without namespace prefix)
        Returns:
            numpy array or None if failed
        """
        object_name = f"{self.target_namespace}/{remote_filename}"
        
        try:
            response = self._s3_client.get_object(self.bucket_name, object_name)
            
            # Read the data into a BytesIO buffer
            data = response.read()
            buffer = io.BytesIO(data)
            
            # Load numpy array from buffer
            metadata = np.load(buffer, allow_pickle=True)
            
            self.logger.info(f"✅ Loaded metadata from {object_name}, shape: {metadata.shape}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load metadata from {object_name}: {e}")
            return None
        finally:
            if 'response' in locals():
                response.close()
    
    def verify_upload(self, local_dir: str, pattern: str = "_embeddings.npy") -> Dict[str, bool]:
        """
        Verify that uploaded files match local files
        Args:
            local_dir: Local directory with original files
            pattern: Pattern to match embedding files
        Returns:
            Dictionary mapping filenames to verification status
        """
        results = {}
        
        # Get list of local embedding files
        local_files = []
        for file in os.listdir(local_dir):
            if file.endswith(".npy") and pattern in file:
                local_files.append(file)
        
        for local_file in local_files:
            self.logger.info(f"🔍 Verifying {local_file}...")
            
            # Load local embeddings
            local_path = os.path.join(local_dir, local_file)
            local_embeddings = np.load(local_path)
            
            # Load remote embeddings
            remote_embeddings = self.load_embeddings_from_remote(local_file)
            
            if remote_embeddings is None:
                results[local_file] = False
                continue
            
            # Compare arrays
            if np.array_equal(local_embeddings, remote_embeddings):
                self.logger.info(f"✅ {local_file} verification PASSED")
                results[local_file] = True
            else:
                self.logger.error(f"❌ {local_file} verification FAILED - arrays don't match")
                results[local_file] = False
            
            # Also verify metadata file if it exists
            metadata_file = local_file.replace("_embeddings.npy", "_metadata.npy")
            metadata_path = os.path.join(local_dir, metadata_file)
            
            if os.path.exists(metadata_path):
                self.logger.info(f"🔍 Verifying {metadata_file}...")
                
                local_metadata = np.load(metadata_path, allow_pickle=True)
                remote_metadata = self.load_metadata_from_remote(metadata_file)
                
                if remote_metadata is None:
                    results[metadata_file] = False
                    continue
                
                if np.array_equal(local_metadata, remote_metadata):
                    self.logger.info(f"✅ {metadata_file} verification PASSED")
                    results[metadata_file] = True
                else:
                    self.logger.error(f"❌ {metadata_file} verification FAILED - arrays don't match")
                    results[metadata_file] = False
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        self.logger.info(f"=== Verification complete: {passed}/{total} files passed ===")
        
        return results
    
    def load_embedding_pair(self, base_filename: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load both embeddings and metadata for a given base filename
        Args:
            base_filename: Base name without suffix (e.g., "chunk_0" for "chunk_0_embeddings.npy")
        Returns:
            Tuple of (embeddings, metadata) or (None, None) if failed
        """
        embeddings_file = f"{base_filename}_embeddings.npy"
        metadata_file = f"{base_filename}_metadata.npy"
        
        embeddings = self.load_embeddings_from_remote(embeddings_file)
        metadata = self.load_metadata_from_remote(metadata_file)
        
        if embeddings is not None and metadata is not None:
            self.logger.info(f"✅ Loaded pair for {base_filename}: "
                           f"embeddings {embeddings.shape}, metadata {metadata.shape}")
        
        return embeddings, metadata
    

class EmbeddingPipeline:
    def __init__(
        self,
        parent_namespace,
        video_namespaces,
        bucket_name="demo-s3",
        embedding_parent_namespace="beit3_retrieval_embeddings_for_vectordb",
        model_name="beit3_large_patch16_384_retrieval",
        batch_size=16,
        num_workers=8
    ):
        self._bucket_name = bucket_name
        self._parent_namespace = parent_namespace
        self._video_namespaces = video_namespaces
        self._embedding_parent_namespace = embedding_parent_namespace
        self._model_name = model_name
        self._batch_size = batch_size
        self._num_workers = num_workers

    def perform(self):
        with timeit_context("Initialize processor"):
            processor = OptimizedFrameEmbeddingProcessor(
                model_name="beit3_large_patch16_384_retrieval",
            )

        with timeit_context("Generate embedding for some namespace under a parent namespace"):
            results = processor.process_multiple_videos(
                bucket_name=self._bucket_name,
                parent_namespace=self._parent_namespace,
                video_namespaces=self._video_namespaces,
                max_frames_per_video=None,
                stride=1  # All frames
            )
            
            processor.save_embeddings_for_vector_db(results, f"{self._embedding_parent_namespace}/")

        with timeit_context("Upload saved embeddings to Cloudflare R2"):
            uploader = CloudflareEmbeddingUploader(
                target_namespace=self._embedding_parent_namespace
            )
            
            uploader.upload_directory(f"{self._embedding_parent_namespace}/")

if __name__ == "__main__":
    embedding_pipeline = EmbeddingPipeline(
        model_name="beit3_large_patch16_384_retrieval",
        parent_namespace="K01",
        video_namespaces=None,
    )

    embedding_pipeline.perform()
