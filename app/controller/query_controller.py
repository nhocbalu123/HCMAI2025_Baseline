from pathlib import Path
import json

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from service import ModelService, KeyframeQueryService
from schema.response import KeyframeServiceReponse


class QueryController:
    
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service

    
    def convert_model_to_path(
        self,
        model: KeyframeServiceReponse
    ) -> tuple[str, float]:
        # return os.path.join(self.data_folder, f"L{model.group_num:02d}/V{model.video_num:03d}/{model.keyframe_num:08d}.webp"), model.confidence_score
        return os.path.join(self.data_folder, f"{model.group_num}/{model.video_num}/{model.keyframe_num}.webp"), model.confidence_score
    
        
    async def search_text(
        self, 
        query: str,
        top_k: int,
        score_threshold: float
    ):
        embedding = self.model_service.embedding(query).tolist()

        result = await self.keyframe_service.search_by_text(embedding, top_k, score_threshold)
        return result


    async def search_text_with_exclude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[str]
    ):        
        embedding = self.model_service.embedding(query).tolist()
        result = await self.keyframe_service.search_by_text_exclude_ids(
            text_embedding=embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            exclude_groups=list_group_exlude
        )
        return result


    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[str],
        list_of_include_videos: list[str]
    ):
        embedding = self.model_service.embedding(query).tolist()
        result = await self.keyframe_service.search_by_text_include_ids(
            text_embedding=embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            include_groups=list_of_include_groups,
            include_videos=list_of_include_videos,
            exclude_ids=None
            
        )
        return result
    

        

