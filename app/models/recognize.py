from pydantic import BaseModel
from typing import List



class Plate(BaseModel):
    plate: str



class RecognitionOutput(BaseModel):
    status: str = 'success'
    results: List[Plate]
    predict_time: float
    confidence: float
