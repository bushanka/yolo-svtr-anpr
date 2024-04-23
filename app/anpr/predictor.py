from .models import Ocr, Yolo


class Anpr:
    def __init__(self, device="cpu") -> None:
        self.yolo = Yolo(device)
        self.ocr = Ocr(device)

    def __call__(self, image):
        result_yolo = self.yolo.detect(image)

        box = result_yolo["box"]

        if box is not None:
            result_ocr = self.ocr.recognize(image[box[1] : box[3], box[0] : box[2]])
        else:
            result_ocr = None

        return {
            "detection": result_yolo,
            "recognition": result_ocr,
        }
