from pydantic import BaseModel, Field


class KeyframeServiceReponse(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: str = Field(..., description="Video ID")
    group_num: str = Field(..., description="Group ID")
    keyframe_num: str = Field(..., description="Keyframe number")
    confidence_score: float = Field(..., description="Keyframe number"),
    global_index: int = Field(..., description="Global index for matching"),
    frame_path: str = Field(..., description="Frame Path"),
    ocr_text: str = Field(default="", description="OCR Extraction")
    


class SingleKeyframeDisplay(BaseModel):
    path: str
    score: float

class KeyframeDisplay(BaseModel):
    results: list[SingleKeyframeDisplay]
