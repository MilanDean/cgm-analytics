import pydantic


class GlucoseData(pydantic.BaseModel):
    Time: str
    BG: float
    CGM: float
    CHO: float
    insulin: float
    LBGI: float
    HBGI: float
    Risk: float


class ProcessedDataResponse(pydantic.BaseModel):
    message: str
    file: str
