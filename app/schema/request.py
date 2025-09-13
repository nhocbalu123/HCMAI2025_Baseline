from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class BaseSearchRequest(BaseModel):
    """Base search request with common parameters"""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=500, description="Number of top results to return")
    score_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")
    using_translator: bool = Field(default=False, description="Using translator flag for triggering translation on query or not")


class TextSearchRequest(BaseSearchRequest):
    """Simple text search request"""
    pass


class TextSearchWithExcludeGroupsRequest(BaseSearchRequest):
    """Text search request with group exclusion"""
    exclude_groups: str = Field(
        default_factory=str,
        description="String of List of group IDs to exclude from search results",
    )


class TextSearchWithSelectedGroupsAndVideosRequest(BaseSearchRequest):
    """Text search request with specific group and video selection"""
    include_groups: str = Field(
        default_factory=str,
        description="String of List of group IDs to include in search results",
    )
    include_videos: str = Field(
        default_factory=str,
        description="String of List of video IDs to include in search results",
    )


class TemporalSearchRequest(BaseSearchRequest):
    include_videos: str = Field(
        default_factory=str,
        description="String of List of video IDs to include in search results",
    )
    temporal_window: Optional[str | None] = Field(
        default=None,
        description="String of list Temporal Window (start, end) for temporal search"
    )
    top_k_weight: Optional[int] = Field(
        default=2,
        description="Extending top_k by top_k_weight times"
    )
    temporal_window_size: int = Field(
        default=1000,
        description="Window size for searching about a specific keyframe index"
    )
