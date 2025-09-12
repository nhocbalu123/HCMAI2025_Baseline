from pydantic import BaseModel, Field


class KeyframeServiceReponse(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: str = Field(..., description="Video ID")
    group_num: str = Field(..., description="Group ID")
    fps: str = Field(..., description="Video FPS")
    keyframe_num: str = Field(..., description="Keyframe ID")
    pts_time: float = Field(..., description="Keyframe timestamp")
    confidence_score: float = Field(..., description="Keyframe number"),
    global_index: int = Field(..., description="Global index for matching"),
    frame_path: str = Field(..., description="Frame Path"),
    


class SingleKeyframeDisplay(BaseModel):
    path: str
    score: float
    fps: str
    pts_time: float

class KeyframeDisplay(BaseModel):
    results: list[SingleKeyframeDisplay]
