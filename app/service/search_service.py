import os
import sys
import asyncio
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from repository.milvus import KeyframeVectorRepository
from repository.milvus import MilvusSearchRequest
from repository.mongo import KeyframeRepository

from schema.response import KeyframeServiceReponse

from service.ocr_service import OcrService

class KeyframeQueryService:
    def __init__(
            self, 
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,
            
        ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo= keyframe_mongo_repo


    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        print(keyframes[:5])
  
        keyframe_map = {k.key: k for k in keyframes}
        return_keyframe = [
            keyframe_map[k] for k in ids
        ]   
        return return_keyframe

    async def _search_keyframes(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        include_groups: list[str] | None = None,
        include_videos: list[str] | None = None,
        exclude_indices: list[str] | None = None
    ) -> list[KeyframeServiceReponse]:
        
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k,
            include_groups=include_groups,
            include_videos=include_videos,
            exclude_ids=exclude_indices
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)
        
        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        # sorted_ids = [result.id_ for result in sorted_results]

        # keyframes = await self._retrieve_keyframes(sorted_ids)



        # keyframe_map = {k.key: k for k in keyframes}
        BASE_URL = "https://pub-6dc786c2b53e460d9ef9948fd14a8a9a.r2.dev/"
        frame_paths = [f"{BASE_URL}{result.frame_path}" for result in sorted_results]
        result_ids = [f"{BASE_URL}{result.id_}" for result in sorted_results]

        if frame_paths:
            print(f"Processing OCR for {len(frame_paths)} images using Tesseract...")
            start_time = asyncio.get_event_loop().time()
            ocr_service = OcrService()
            ocr_results = ocr_service.process_urls_batch(urls=frame_paths)
            end_time = asyncio.get_event_loop().time()
            print(f"OCR processing completed in {end_time - start_time:.2f} seconds")
        else:
            ocr_results = [""] * len(sorted_results)
        
        dict_ocr_results = {r_id: ocr_text for r_id, ocr_text in ocr_results]}
        response = []

        for result in sorted_results:
            # keyframe = keyframe_map.get(result.id_)
            if result.frame_id is not None:
                response.append(
                    KeyframeServiceReponse(
                        key=result.id_,
                        video_num=result.video_namespace,
                        group_num=result.parent_namespace,
                        keyframe_num=result.frame_id,
                        global_index=result.global_index,
                        confidence_score=result.distance,
                        frame_path=result.frame_path,
                        ocr_text=dict_ocr_results.get(result.id_, "")
                    )
                )
        return response
    

    async def search_by_text(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        return await self._search_keyframes(text_embedding, top_k, score_threshold, None)   
    

    async def search_by_text_range(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        range_queries: list[tuple[int,int]]
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """

        all_ids = self.keyframe_vector_repo.get_all_id()
        allowed_ids = set()
        for start, end in range_queries:
            allowed_ids.update(range(start, end + 1))
        
        
        exclude_ids = [id_ for id_ in all_ids if id_ not in allowed_ids]

        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)   

    async def search_by_text_exclude_ids(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        exclude_groups: list[str] | None
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """
        return await self._search_keyframes(
            text_embedding=text_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            exclude_indices=exclude_groups
        )

    async def search_by_text_include_ids(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        include_groups: list[str] | None,
        include_videos: list[str] | None,
        exclude_ids: list[str] | None
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """
        return await self._search_keyframes(
            text_embedding=text_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            include_groups=include_groups,
            include_videos=include_videos,
            exclude_indices=exclude_ids
        ) 
