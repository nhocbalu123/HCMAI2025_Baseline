from pydantic import BaseModel, Field

class AgentResponse(BaseModel):
    refined_query: str = Field(..., description="The rephrased response")
    list_of_objects: list[str] | None = Field(None, description="The list of objects for filtering(Object from coco class), optionall")

    