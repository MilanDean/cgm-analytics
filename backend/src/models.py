import pydantic

class ProcessedDataResponse(pydantic.BaseModel):
    message: str
    file: str
