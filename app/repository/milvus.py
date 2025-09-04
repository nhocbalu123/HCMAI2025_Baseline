"""
The implementation of Vector Repository. The following class is responsible for getting the vector by many ways
Including Faiss and Usearch
"""


import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from typing import cast
from common.repository import MilvusBaseRepository
from pymilvus import Collection as MilvusCollection
from pymilvus.client.search_result import SearchResult
from schema.interface import  MilvusSearchRequest, MilvusSearchResult, MilvusSearchResponse


class KeyframeVectorRepository(MilvusBaseRepository):
    
    OUTPUT_FIELDS = [
        "id",
        "embedding",
        "global_index",
        "frame_id",
        "frame_path",
        "parent_namespace",
        "video_namespace",
    ]

    def __init__(
        self, 
        collection: MilvusCollection,
        search_params: dict
    ):
        
        super().__init__(collection)
        self.search_params = search_params
    
    async def search_by_embedding(
        self,
        request: MilvusSearchRequest
    ):
        expr = None
        if request.exclude_ids:
            expr = f"id not in {request.exclude_ids}"

        search_results= cast(SearchResult, self.collection.search(
            data=[request.embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=request.top_k,
            expr=expr ,
            output_fields=KeyframeVectorRepository.OUTPUT_FIELDS,
            _async=False
        ))

        results = []
        for hits in search_results:
            for hit in hits:
                result = MilvusSearchResult(
                    id_=hit.id,
                    distance=hit.distance,
                    embedding=hit.entity.get("embedding", None),
                    global_index=hit.entity.get("global_index", None),
                    frame_id=hit.entity.get("frame_id", None),
                    frame_path=hit.entity.get("frame_path", None),
                    parent_namespace=hit.entity.get("parent_namespace", None),
                    video_namespace=hit.entity.get("video_namespace", None)
                )
                results.append(result)

        
        return MilvusSearchResponse(
            results=results,
            total_found=len(results),
        )
    
    def get_all_id(self) -> list[int]:
        return list(range(self.collection.num_entities))
