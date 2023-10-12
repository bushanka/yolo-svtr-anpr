from celery import Celery

from app import anpr

from dotenv import load_dotenv
import cv2
import numpy as np
import os



load_dotenv()
BROKER = os.getenv('APP_BROKER_URI')
RUNTYPE = os.getenv('RUNTYPE') 

app = Celery('app.tasks.recognition_task', backend='rpc://', broker=BROKER)
app.conf.task_routes = {'recognize': {'queue': 'recognitiontask_queue'}}

model = anpr.Anpr(RUNTYPE)


@app.task(name='recognize')
def recognize_plate(image_bytes):
    nparr = np.fromstring(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = model(img_np)

    if not result['recognition']:
        return {
            'plate': "no detection",
            'confidence': -1,
            'predict_time': result['detection']['speed']['total']
    
        }

    plate_num = result['recognition']['text']
    total_prob = result['detection']['prob'] * result['recognition']['prob']
    total_time = float(result['detection']['speed']['total']) + float(result['recognition']['speed']['total'])

    return {
        'plate': plate_num,
        'confidence': total_prob,
        'predict_time': total_time
    }