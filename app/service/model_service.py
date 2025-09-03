import torch
import numpy as np
from PIL import Image


class ModelService:
    def __init__(
        self,
        model,
        preprocess,
        tokenizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.model = model.to(device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        print(f"ModelService initialized on device: {device}")
        print(f"Model type: {type(self.model)}")
    
    def embedding(self, query_text: str) -> np.ndarray:
        """
        Generate text embedding for query text using BEiT3
        
        Args:
            query_text (str): Input text query
            
        Returns:
            np.ndarray: Normalized text embedding (shape: [embedding_dim])
        """
        try:
            # Process text input
            inputs = self.preprocess.process(image=None, text=query_text)
            
            # Move inputs to device
            text_description = inputs['text_description'].to(self.device)
            padding_mask = inputs['padding_mask'].to(self.device)
            
            print(f"Text input shape: {text_description.shape}")
            print(f"Padding mask shape: {padding_mask.shape}")
            
            with torch.no_grad():
                # Forward pass through BEiT3 model
                if hasattr(self.model, 'forward') and 'only_infer' in self.model.forward.__code__.co_varnames:
                    # Your current model interface
                    _, text_feature = self.model(
                        text_description=text_description,
                        padding_mask=padding_mask,
                        only_infer=True
                    )
                else:
                    print("Alternative interface for BEiT3ForRetrieval")
                    # outputs = self.model(
                    #     textual_tokens=text_description,
                    #     textual_attention_mask=(1 - padding_mask),  # Invert mask
                    #     image=None
                    # )
                    # text_feature = outputs.text_embeds

                    _, text_feature = self.model(
                        text_description=text_description,
                        padding_mask=padding_mask,
                        only_infer=True
                    )

                    # raise "Alternative interface for BEiT3ForRetrieval"
            
            # Normalize the embedding
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            # Convert to numpy and return
            embedding = text_feature[0].cpu().numpy().astype(np.float32)
            
            return embedding
            
        except Exception as e:
            print(f"Error in text embedding: {e}")
            print(f"Query text: '{query_text}'")
            raise e
    
    def encode_image(self, image) -> np.ndarray:
        """
        Generate image embedding using BEiT3
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            np.ndarray: Normalized image embedding (shape: [embedding_dim])
        """
        try:
            # Handle image input
            if isinstance(image, (str, Path)):
                raw_image = Image.open(image).convert("RGB")
            else:
                raw_image = image
            
            # Process image input
            inputs = self.preprocess.process(image=raw_image, text=None)
            
            # Move to device
            image_tensor = inputs['image'].to(self.device)
            
            with torch.no_grad():
                # Forward pass through BEiT3 model
                if hasattr(self.model, 'forward') and 'only_infer' in self.model.forward.__code__.co_varnames:
                    # Your current model interface
                    image_feature, _ = self.model(
                        image=image_tensor,
                        only_infer=True
                    )
                else:
                    # Alternative interface for BEiT3ForRetrieval
                    outputs = self.model(
                        textual_tokens=None,
                        textual_attention_mask=None,
                        image=image_tensor
                    )
                    image_feature = outputs.image_embeds
            
            # Normalize the embedding
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            
            # Convert to numpy and return
            embedding = image_feature[0].cpu().numpy().astype(np.float32)
            
            return embedding
            
        except Exception as e:
            print(f"Error in image embedding: {e}")
            raise e
    
    def encode_multimodal(self, query_text: str, image) -> np.ndarray:
        """
        Generate multimodal embedding using both text and image
        
        Args:
            query_text (str): Input text query
            image: PIL Image or path to image file
            
        Returns:
            np.ndarray: Normalized multimodal embedding
        """
        try:
            # Handle image input
            if isinstance(image, (str, Path)):
                raw_image = Image.open(image).convert("RGB")
            else:
                raw_image = image
            
            # Process both inputs
            inputs = self.preprocess.process(image=raw_image, text=query_text)
            
            # Move to device
            image_tensor = inputs['image'].to(self.device)
            text_description = inputs['text_description'].to(self.device)
            padding_mask = inputs['padding_mask'].to(self.device)
            
            with torch.no_grad():
                # Forward pass with both modalities
                if hasattr(self.model, 'forward') and 'only_infer' in self.model.forward.__code__.co_varnames:
                    # Your current model interface - this might need adjustment
                    # Check if your model supports multimodal input
                    outputs = self.model(
                        image=image_tensor,
                        text_description=text_description,
                        padding_mask=padding_mask,
                        only_infer=True
                    )
                    multimodal_feature = outputs  # Adjust based on actual output structure
                else:
                    # Alternative interface
                    outputs = self.model(
                        textual_tokens=text_description,
                        textual_attention_mask=(1 - padding_mask),
                        image=image_tensor
                    )
                    # Combine text and image features
                    multimodal_feature = (outputs.text_embeds + outputs.image_embeds) / 2
            
            # Normalize the embedding
            multimodal_feature = multimodal_feature / multimodal_feature.norm(dim=-1, keepdim=True)
            
            # Convert to numpy and return
            embedding = multimodal_feature[0].cpu().numpy().astype(np.float32)
            
            return embedding
            
        except Exception as e:
            print(f"Error in multimodal embedding: {e}")
            raise e
