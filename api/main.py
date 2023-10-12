import uuid
from typing import List, Union
 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from api.recognition_task import recognize_plate
from celery.result import AsyncResult
from celery import Celery
import numpy as np
import asyncio



# class InputImage(BaseModel):
#     image_bytes: bytes = File()


class RecognitionTask(BaseModel):
    task_id: str
    status: str = 'pending'


class Output(BaseModel):
    status: str = 'success'
    plate: str
    # plate_type: str
    # plate_region: str
    predict_time: float
    confidence: float


app = FastAPI()
celery_app = Celery('recognition_task', backend='rpc://', broker='pyamqp://guest@localhost//')

async def celery_async_wrapper(app, task_name, task_args, queue):
    delay = 0.1
    max_tries = 20

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
        return

    result = task.get()

    return Output(
        plate=result['plate'],
        # plate_type='car',
        # plate_region='ru',
        predict_time=result['predict_time'],
        confidence=result['confidence']
    )


@app.get('/ping')
async def check():
    return 'pong'


@app.post('/recognize')
async def recognize(image_bytes: bytes = File()):
    task_id = recognize_plate.delay(image_bytes)

    result = await celery_async_wrapper(
        celery_app, 
        'recognize', 
        task_args=(image_bytes,), 
        queue='recognitiontask_queue'
    )
    if result:
        return result
    else:
        new_task = RecognitionTask(task_id=str(task_id))
        return new_task


@app.get('/result/{task_id}')
async def fetch_result(task_id):
    task = AsyncResult(task_id)
    if not task.ready():
        return RecognitionTask(task_id=str(task_id))
    
    result = task.get()

    return Output(
        plate=result['plate'],
        # plate_type='car',
        # plate_region='ru',
        predict_time=result['predict_time'],
        confidence=result['confidence']
    )


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")