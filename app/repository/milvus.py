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
    
    def _build_expression(self, request: MilvusSearchRequest) -> str | None:
        conditions = []
        
        # Exclude IDs
        if request.exclude_ids:
            conditions.append(f"parent_namespace not in {request.exclude_ids}")
        
        # Include specific parent namespaces
        if hasattr(request, 'include_groups') and request.include_groups:
            conditions.append(f"parent_namespace in {request.include_groups}")
        
        # Include specific video namespaces
        if hasattr(request, 'include_videos') and request.include_videos:
            conditions.append(f"video_namespace in {request.include_videos}")
        
        # Exclude specific parent namespaces
        # if hasattr(request, 'exclude_parent_namespaces') and request.exclude_parent_namespaces:
        #     conditions.append(f"parent_namespace not in {request.exclude_parent_namespaces}")
        
        # Join conditions with AND
        return " and ".join(conditions) if conditions else None

    async def search_by_embedding(
        self,
        request: MilvusSearchRequest
    ):
        expr = self._build_expression(request=request)

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
