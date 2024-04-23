from typing import List

from pydantic import BaseModel


class Plate(BaseModel):
    plate: str


class RecognitionOutput(BaseModel):
    status: str = "success"
    results: List[Plate]
    predict_time: float
    confidence: float
