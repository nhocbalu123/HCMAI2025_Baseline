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
from schema.interface import  MilvusSearchRequest, MilvusSearchResult, MilvusSearchResponse,\
    NeighborSearchRequest, TemporalMilvusSearchRequest


class KeyframeVectorRepository(MilvusBaseRepository):
    
    OUTPUT_FIELDS = [
        "id",
        "embedding",
        "global_index",
        "fps",
        "frame_id",
        "pts_time",
        "frame_path",
        "parent_namespace",
        "video_namespace",
    ]

    STANDARD_FRAME_ID_LEN = 7

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
        request: MilvusSearchRequest | TemporalMilvusSearchRequest
    ):
        expr = None

        if isinstance(request, MilvusSearchRequest):
            expr = self._build_expression(request=request)
        elif isinstance(request, TemporalMilvusSearchRequest):
            expr = request.expr

        print("search_by_embedding", self.search_params)

        search_results = cast(SearchResult, self.collection.search(
            data=[request.embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=request.top_k,
            expr=expr if expr else None,
            output_fields=KeyframeVectorRepository.OUTPUT_FIELDS,
            _async=False
        ))

        print("Done search. Preparing results")

        results = []
        for hits in search_results:
            for hit in hits:
                result = MilvusSearchResult(
                    id_=hit.id,
                    distance=hit.distance,
                    embedding=hit.entity.get("embedding", None),
                    global_index=hit.entity.get("global_index", None),
                    fps=hit.entity.get("fps", None),
                    frame_id=hit.entity.get("frame_id", None),
                    pts_time=hit.entity.get("pts_time", None),
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

    @staticmethod
    def standardize_frame_id_format(value_like_frame_id: str | int) -> str:
        """
        Format value like frame ID format into same format
        for lexicographic comparison
        """
        return str(value_like_frame_id).zfill(
            KeyframeVectorRepository.STANDARD_FRAME_ID_LEN
        )

    async def neighboring_frames_search(
        self,
        neighbor_request: NeighborSearchRequest
    ):
        """
        Neighboring frames search for temporal search
        """
        # Calculate frame index range
        start_idx = max(
            0,
            int(neighbor_request.frame_id) - neighbor_request.window_size
        )
        end_idx = int(neighbor_request.frame_id) + neighbor_request.window_size + 1
        
        start_idx = KeyframeVectorRepository.standardize_frame_id_format(start_idx)
        end_idx = KeyframeVectorRepository.standardize_frame_id_format(end_idx)
        frame_idx = KeyframeVectorRepository.standardize_frame_id_format(
            neighbor_request.frame_id
        )
        
        # Query for neighboring frames
        expr = (
            f'video_namespace == "{neighbor_request.video_namespace}" && '
            f'frame_id >= {start_idx} && frame_id < {end_idx} && '
            f'frame_id != "{frame_idx}"'
        )
        
        results = cast(SearchResult, self.collection.query(
            data=[neighbor_request.query_embedding],
            anns_field="embedding",
            expr=expr,
            output_fields=KeyframeVectorRepository.OUTPUT_FIELDS
        ))

        return results
