from fastapi import(
    APIRouter, 
    status,
    File
)

from celery.result import AsyncResult
from celery import Celery

from app.models.recognize import RecognitionOutput, RecognitionTask

from pydantic import BaseModel
from typing import Union, Tuple
from dotenv import load_dotenv
import os
import asyncio


load_dotenv()
MAX_TRIES = int(os.getenv('MAX_TRIES'))
DELAY = float(os.getenv('DELAY'))

router = APIRouter(
    prefix="/recognition",
    tags=["recognition"],
)
celery_app = Celery('app.tasks.recognition_task', backend='rpc://', broker=os.getenv('APP_BROKER_URI'))



async def celery_async_wrapper(
        app: Celery, 
        task_name: str, 
        task_args: Tuple, 
        queue: str
) -> Union[RecognitionOutput, int]:
    delay = DELAY
    max_tries = MAX_TRIES

    task_id = app.send_task(task_name, [*task_args], queue=queue)
    task = AsyncResult(task_id)

    while not task.ready() and max_tries > 0:
        await asyncio.sleep(delay)
        # Через 5 итераций выходит на 2 секунды
        # Total wait: 3.1 sec после 5 итераций, далее по 2 сек делей
        # Максимум будет 33 секунды - потом Time out 
        delay = min(delay * 2, 2)  # exponential backoff, max 2 seconds
        max_tries -= 1

    if max_tries <= 0:
        return task_id

    result = task.get()

    return RecognitionOutput(
        plate=result['plate'],
        predict_time=result['predict_time'],
        confidence=result['confidence']
    )


@router.post(
    '/recognize',
    responses={
        status.HTTP_200_OK: {
            "description": "Recognition successful"
        },
        status.HTTP_201_CREATED: {
            "description": "Recognition is still pending"
        }
    }
)
async def recognize(image_bytes: bytes = File()) -> Union[RecognitionOutput, RecognitionTask]:
    result = await celery_async_wrapper(
        celery_app, 
        'recognize', 
        task_args=(image_bytes,), 
        queue='recognitiontask_queue'
    )
    if isinstance(result, RecognitionOutput):
        return result
    else:
        return RecognitionTask(task_id=str(result))


@router.get(
        '/result/{task_id}',
        responses={
            status.HTTP_200_OK: {
                "description": "Returns status of recognition task"
            }
        }
)
async def fetch_result(task_id: int) -> Union[RecognitionTask, RecognitionOutput]:
    task = AsyncResult(task_id)

    if not task.ready():
        return RecognitionTask(task_id=str(task_id))
    
    result = task.get()

    return RecognitionOutput(
        plate=result['plate'],
        predict_time=result['predict_time'],
        confidence=result['confidence']
    )
