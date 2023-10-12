import requests
import pprint
import cv2
import json

# resp = requests.get('158.0.0.1:8000/ping')
# image = cv2.imread('/home/bush/project/anpr/test/T008AT197.jpg')
# image_bytes = cv2.imencode('.jpg', image)[1].tostring()

file = {
    "image": open('/home/bush/project/anpr/test/T008AT197.jpg', 'rb')
}
resp = requests.post('http://158.160.100.102:8080/recognize', files=file)
print(resp)




# import cv2

# cam = cv2.VideoCapture(0)

# # get image from web camera
# ret, frame = cam.read()

# # convert to jpeg and save in variable
# image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()