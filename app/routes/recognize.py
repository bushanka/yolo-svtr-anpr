import time
from fastapi import(
    APIRouter, 
    status,
    File
)
import numpy as np
import os
import cv2
import logging
import sys
from app import anpr
from app.models.recognize import RecognitionOutput, Plate


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
            "description": "Returns recognition result."
        },
    }
)
def recognize(images: bytes = File()) -> RecognitionOutput:
    nparr = np.fromstring(images, np.uint8)
    try:
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except cv2.error as ex_cv2:
        logger.info(f"Error in decoding image: {ex_cv2}")
        return RecognitionOutput(
            status='error',
            results=[
                Plate(
                    plate="no detection, error in decoding image (maybe empty image sended)"
                )
            ],
            confidence=-1,
            predict_time=0.0
        )

    try:
        result = model(img_np)
    except AttributeError as ex_attrerror:
        logger.info(ex_attrerror)
        return RecognitionOutput(
            status='error',
            results=[
                Plate(
                    plate=f"no detection, error in model inference (maybe image shape is 0) recieved image len: {len(nparr)}"
                )
            ],
            confidence=-1,
            predict_time=0.0
        )

    if not result['recognition']:
        logger.info("no detection")
        ts = round(time.time())
        cv2.imwrite(os.path.join("app", "images", f"{ts}_no_detecion.jpg"), img_np)
        return RecognitionOutput(
            results=[
                Plate(
                    plate=f"no detection"
                )
            ],
            confidence=-1,
            predict_time=result['detection']['speed']['total']
        )
    plate_num = result['recognition']['text']
    total_prob = result['detection']['prob'] * result['recognition']['prob']
    total_time = float(result['detection']['speed']['total']) + float(result['recognition']['speed']['total'])

    logger.info(f"detected: {plate_num} - prob: {total_prob} - took: {total_time}")
    ts = round(time.time())
    cv2.imwrite(os.path.join("app", "images", f"{ts}_{plate_num}.jpg"), img_np)

    return RecognitionOutput(
        results=[
            Plate(
                plate=plate_num
            )
        ],
        confidence=total_prob,
        predict_time=total_time
    )
