from fastapi import(
    APIRouter, 
    status,
    File
)
import numpy as np
import os
import cv2
from app import anpr
from app.models.recognize import RecognitionOutput, Plate



router = APIRouter(
    prefix="/recognition",
    tags=["recognition"],
)
RUNTYPE = os.getenv('RUNTYPE') if os.getenv('RUNTYPE') else 'cpu'
model = anpr.Anpr(RUNTYPE)



@router.post(
    '/recognize',
    responses={
        status.HTTP_200_OK: {
            "description": "Returns recognition result or task id if time out"
        },
    }
)
def recognize(images: bytes = File()) -> RecognitionOutput:
    nparr = np.fromstring(images, np.uint8)
    try:
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except cv2.error as ex_cv2:
        print(f"Error in decoding image: {ex_cv2}")
        return RecognitionOutput(
            results=[
                Plate(
                    plate="no detection, error in decoding image (maybe empty image sended)"
                )
            ]
            confidence=-1,
            predict_time=0.0
        )

    try:
        result = model(img_np)
    except AttributeError as ex_attrerror:
        print(ex_attrerror)
        return RecognitionOutput(
            results=[
                Plate(
                    plate=f"no detection, error in model inference (maybe image shape is 0) recieved image len: {len(nparr)}"
                )
            ]
            confidence=-1,
            predict_time=0.0
        )

    if not result['recognition']:
        return RecognitionOutput(
            results=[
                Plate(
                    plate=f"no detection"
                )
            ]
            confidence=-1,
            predict_time=result['detection']['speed']['total']
        )
    plate_num = result['recognition']['text']
    total_prob = result['detection']['prob'] * result['recognition']['prob']
    total_time = float(result['detection']['speed']['total']) + float(result['recognition']['speed']['total'])


    return RecognitionOutput(
        results=[
            Plate(
                plate=plate_num
            )
        ]
        confidence=total_prob,
        predict_time=total_time
    )
