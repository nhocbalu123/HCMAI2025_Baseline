from pydantic import BaseModel, Field
from typing import Optional


class KeyframeServiceReponse(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: str = Field(..., description="Video ID")
    group_num: str = Field(..., description="Group ID")
    fps: str = Field(..., description="Video FPS")
    keyframe_num: str = Field(..., description="Keyframe ID")
    pts_time: float = Field(..., description="Keyframe timestamp")
    confidence_score: float = Field(..., description="Keyframe number")
    global_index: int = Field(..., description="Global index for matching")
    frame_path: str = Field(..., description="Frame Path")
    temporal_score: Optional[float] = Field(default=None, description="Temporal score in temporal search")
    combined_score: Optional[float] = Field(default=None, description="Combined score in temporal search")


class SingleKeyframeDisplay(BaseModel):
    path: str
    score: float
    fps: str
    pts_time: float


class TemporalKeyframeDisplay(SingleKeyframeDisplay):
    temporal_score: float
    combined_score: float


class KeyframeDisplay(BaseModel):
    results: list[SingleKeyframeDisplay | TemporalKeyframeDisplay]
