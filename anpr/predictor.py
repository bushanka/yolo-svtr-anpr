from .models import Yolo, Ocr

class Anpr():
    def __init__(self) -> None:
        self.yolo = Yolo()
        self.ocr = Ocr()
    
    def __call__(self, image):
        result_yolo = self.yolo.detect(image)

        box = result_yolo['box']

        result_ocr = self.ocr.recognize(image[box[1]:box[3],box[0]:box[2]])

        return {
            'detection': result_yolo,
            'recognition': result_ocr,
        }
        