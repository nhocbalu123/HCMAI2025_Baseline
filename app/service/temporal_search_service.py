import numpy as np
from typing import List, Dict, Tuple, Optional, Any, cast
from pymilvus.client.search_result import SearchResult
from schema.interface import  TemporalMilvusSearchRequest, NeighborSearchRequest,\
    MilvusSearchResult
from schema.response import KeyframeServiceReponse
from repository.milvus import KeyframeVectorRepository
from core.settings import TemporalSettings


class TemporalSearchService:
    def __init__(
        self,
        keyframe_vector_repo: KeyframeVectorRepository
    ):
        self._keyframe_vector_repo = keyframe_vector_repo
        self._temporal_settings = TemporalSettings()

    async def _get_neighboring_frames(
        self,
        video_namespace: str,
        frame_id: str,
        query_embedding: np.ndarray,
        window_size: int = 100
    ) -> List[Dict]:
        """Get neighboring frames within the same video"""

        neighbor_request = NeighborSearchRequest(
            video_namespace=video_namespace,
            frame_id=frame_id,
            query_embedding=query_embedding,
            window_size=window_size
        )

        results = await self._keyframe_vector_repo.neighboring_frames_search(
            neighbor_request=neighbor_request
        )

        sorted_results = sorted(
            results, key=lambda r: r.distance, reverse=True
        )

        response = []

        for result in sorted_results:
            if result.frame_id is not None:
                response.append(
                    KeyframeServiceReponse(
                        key=result.id_,
                        video_num=result.video_namespace,
                        group_num=result.parent_namespace,
                        fps=result.fps,
                        keyframe_num=result.frame_id,
                        pts_time=result.pts_time,
                        global_index=result.global_index,
                        confidence_score=result.distance,
                        frame_path=result.frame_path,
                    )
                )

        return response

    async def _calculate_temporal_context_score(
        self,
        video_namespace: str,
        frame_id: str,
        query_embedding: np.ndarray,
        window_size: int = 100
    ) -> float:
        """Calculate temporal context score based on neighboring frames"""
        neighbors = await self._get_neighboring_frames(
            video_namespace=video_namespace,
            frame_id=frame_id,
            query_embedding=query_embedding,
            window_size=window_size
        )

        if not neighbors:
            return 0.0

        # Calculate similarities with neighbors
        similarities = []

        for keyframe_response in neighbors:
            similarities.append(
                self._convert_distance_to_sim_score(
                    keyframe_response.confidence_score
                )
            )

        return np.mean(similarities) if similarities else 0.0

    def _build_filter_expression(
        self,
        temporal_window: Optional[Tuple[float, float]] = None,
        video_namespaces: Optional[List[str]] = None
    ) -> str:
        """Build Milvus filter expression"""
        conditions = []

        if temporal_window:
            start_time, end_time = temporal_window
            conditions.append(f"pts_time >= {start_time} && pts_time <= {end_time}")

        if video_namespaces:
            video_filter = " || ".join([
                f'video_namespace == "{vid}"'
                for vid in video_namespaces
            ])
            conditions.append(f"({video_filter})")

        return " && ".join(conditions) if conditions else ""

    def _calculate_combined_score(
        self,
        semantic_score,
        temporal_score
    ) -> float:
        score = (
            ((1 - self._temporal_settings.TEMPORAL_WEIGHT) * semantic_score)
            +
            (self._temporal_settings.TEMPORAL_WEIGHT * temporal_score)
        )

        return score

    def _convert_distance_to_sim_score(self, cosine_distance):
        """Maps [-1,1] to [0,1]"""
        return (cosine_distance + 1) / 2.0

    async def search(
        self,
        query_embedding: np.ndarray, 
        top_k: int = 10,
        temporal_window: Optional[Tuple[float, float]] = None,
        video_namespaces: Optional[List[str]] = None,
        top_k_weight: int = 2,
        temporal_window_size: Optional[int] = 1000,
    ) -> List[MilvusSearchResult]:
        """
        Perform multi-stage temporal search
        
        Args:
            query_embedding: Text query embedding
            top_k: Number of results to return
            temporal_window: Optional (start_time, end_time) in seconds
            video_ids: Optional list of video IDs to search within
            search_params: Milvus search parameters
            
        Returns:
            List of TemporalSearchResult objects ranked by combined score
        """
        
        # Build filter expression
        filter_expr = self._build_filter_expression(
            temporal_window=temporal_window,
            video_namespaces=video_namespaces
        )

        temporal_request = TemporalMilvusSearchRequest(
            embedding=query_embedding,
            top_k=top_k * top_k_weight,
            expr=filter_expr
        )
        
        # Stage 1: Semantic similarity search in Milvus
        print("TemporalSearchService - Stage 1")
        search_results = await self._keyframe_vector_repo.search_by_embedding(
           request=temporal_request
        )
        
        if not search_results or not search_results.results:
            return []
        
        # Stage 2: Temporal context enhancement
        print("TemporalSearchService - Stage 2")
        results = []

        print(len(search_results.results))
        
        for milvus_result in search_results.results:
            # Extract hit information
            distance = milvus_result.distance

            # Convert distance to similarity score (for cosine distance)
            semantic_score = self._convert_distance_to_sim_score(distance)

            # Calculate temporal context score
            temporal_score = await self._calculate_temporal_context_score(
                video_namespace=milvus_result.video_namespace,
                frame_id=milvus_result.frame_id,
                query_embedding=query_embedding,
                window_size=temporal_window_size
            )

            # Combine scores
            combined_score = self._calculate_combined_score(
                semantic_score=semantic_score,
                temporal_score=temporal_score
            )

            milvus_result.combined_score = combined_score
            milvus_result.temporal_score = temporal_score

            result = KeyframeServiceReponse(
                key=milvus_result.id_,
                video_num=milvus_result.video_namespace,
                group_num=milvus_result.parent_namespace,
                fps=milvus_result.fps,
                keyframe_num=milvus_result.frame_id,
                pts_time=milvus_result.pts_time,
                global_index=milvus_result.global_index,
                confidence_score=milvus_result.distance,
                frame_path=milvus_result.frame_path,
                temporal_score=milvus_result.temporal_score,
                combined_score=milvus_result.combined_score
            )

            results.append(result)

        # Stage 3: Final ranking by combined score
        print("TemporalSearchService - Stage 3")

        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]
