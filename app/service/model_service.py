import torch
import numpy as np
from PIL import Image


class ModelService:
    def __init__(
        self,
        model ,
        preprocess ,
        tokenizer ,
        device: str='cuda'
        ):
        self.model = model
        self.model = model.to(device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    # def embedding(self, query_text: str) -> np.ndarray:
    #     """
    #     Return (1, ndim=1024) numpy embedding for BEiT-3 large
    #     """
    #     print("Not use at the moment")
    
    def encode_image(self, image):
        try:
            raw_image = Image.open(image).convert("RGB")
        except Exception as e:
            print(e)
            print("Use origin")
            raw_image = image
        inputs = self.preprocess.process(image=raw_image, text=None)

        with torch.no_grad():
            image_feature, _ = self.model(image=inputs['image'].to(self.device), only_infer=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        
        return image_feature[0].cpu().numpy().astype(np.float32)                       

    def embedding(self, query_text):
        inputs = self.preprocess.process(image=None, text=query_text)
        with torch.no_grad():
            _, text_feature = self.model(
                text_description=inputs['text_description'].to(self.device),
                padding_mask=inputs['padding_mask'].to(self.device),
                only_infer=True
            )
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        return text_feature[0].cpu().numpy().astype(np.float32)
            