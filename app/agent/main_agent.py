import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)

from typing import Dict, List, cast
from llama_index.core.llms import LLM

from .agent import VisualEventExtractor, AnswerGenerator

from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.response import KeyframeServiceReponse
from functools import reduce




def apply_object_filter(
        keyframes: List[KeyframeServiceReponse], 
        objects_data: dict[str, list[str]], 
        target_objects: List[str]
    ) -> List[KeyframeServiceReponse]:
        
        if not target_objects:
            return keyframes
        
        target_objects_set = {obj.lower() for obj in target_objects}
        filtered_keyframes = []
        
        for kf in keyframes:
            keyy = f"L{kf.group_num:02d}/V{kf.video_num:03d}/{kf.keyframe_num:08d}.webb"
            keyframe_objects = objects_data.get(keyy, [])
            keyframe_objects_set = {obj.lower() for obj in keyframe_objects}
            
            if target_objects_set.intersection(keyframe_objects_set):
                filtered_keyframes.append(kf)
        
        return filtered_keyframes





class KeyframeSearchAgent:
    def __init__(
        self, 
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data: dict[str, list[str]],
        asr_data: dict[str, str | list[dict[str,str]]],
        top_k: int = 50
    ):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.data_folder = data_folder
        self.top_k = top_k

        self.objects_data = objects_data or {}
        self.asr_data = asr_data or {}

        self.query_extractor = VisualEventExtractor(llm)
        self.answer_generator = AnswerGenerator(llm, data_folder)

    
    async def process_query(self, user_query: str) -> str:
        """
        Main agent flow:
        1. Extract visual/event elements and rephrase query
        2. Search for top-K keyframes using rephrased query
        3. Score videos by averaging keyframe scores, select best video
        4. Optionally apply COCO object filtering
        5. Generate final answer with visual context
        """
        agent_response = await self.query_extractor.extract_visual_events(user_query)
        search_query = agent_response.refined_query
        suggested_objects = agent_response.list_of_objects

        embedding = self.model_service.embedding(search_query).tolist()[0]
        top_k_keyframes = await self.keyframe_service.search_by_text(
            text_embedding=embedding,
            top_k=self.top_k,
            score_threshold=0.3
        )

        video_scores = self.query_extractor.calculate_video_scores(top_k_keyframes)
        _, _, best_video_keyframes = video_scores[0]



        final_keyframes = best_video_keyframes
        if suggested_objects:
            filtered_keyframes = apply_object_filter(
                keyframes=best_video_keyframes,
                objects_data=self.objects_data,
                target_objects=suggested_objects
            )
            if filtered_keyframes:  
                final_keyframes = filtered_keyframes
        
        group = final_keyframes[0].group_num
        
        smallest_kf = min(final_keyframes, key=lambda x: int(x.keyframe_num))
        max_kf = max(final_keyframes, key=lambda x: int(x.keyframe_num))

        group_num = smallest_kf.group_num
        video_num = smallest_kf.video_num


        matching_asr = next(
            (entry for entry in self.asr_data if entry["file_path"] == f"L{group_num:02d}/V{video_num:03d}"),
            None
        )

        
        asr_entries = matching_asr["result"]
        asr_text_segments = [
            seg["text"]
            for seg in asr_entries
            if int(seg["start_frame"]) >= int(smallest_kf.keyframe_num)
            and int(seg["end_frame"]) <= int(max_kf.keyframe_num)
        ]
        asr_text = " ".join(asr_text_segments)



        answer = await self.answer_generator.generate_answer(
            original_query=user_query,
            final_keyframes=final_keyframes,
            objects_data=self.objects_data,
            asr_data=asr_text
        )

        return cast(str, answer)

        