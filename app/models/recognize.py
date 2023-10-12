from pydantic import BaseModel



class RecognitionTask(BaseModel):
    task_id: str
    status: str = 'pending'


class RecognitionOutput(BaseModel):
    status: str = 'success'
    plate: str
    predict_time: float
    confidence: float