import onnxruntime
import numpy as np
import cv2
import time
import os


class Yolo():
    def __init__(self) -> None:
        # Тут можно параметры сессии добавить
        # opt_session = onnxruntime.SessionOptions()
        # opt_session.enable_mem_pattern = False
        # opt_session.enable_cpu_mem_arena = False
        # opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.conf_thresold = 0.6

        module_dir = os.path.dirname(__file__)
        model_path = os.path.join(module_dir, 'weights', 'yolo_sim.onnx')

        EP_list = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

        self.ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

        model_inputs = self.ort_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.input_height, self.input_width = input_shape[2:]
        self.resized_height, self.resized_width = (640, 640)

        model_output = self.ort_session.get_outputs()
        self.output_names = [model_output[i].name for i in range(len(model_output))]

    
    def rescale_prediction(self, xyxy, dwdh, ratio):
        """Rescale prediction back to original image size.

        Args:
            image: image whose characters need to be rescaled

        Returns:
            Rescaled image predictions
        """
        xyxy = np.array(xyxy, 'float64')
        xyxy -= np.tile(dwdh, 2)
        xyxy /= ratio
        xyxy = np.array(xyxy).round().astype(np.int32)

        return xyxy
    

    def _letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=False, stride=32) -> np.ndarray:
        """Resize image by applying letterbox.

        Args:
            new_shape: New dimension of images
            color: Letterbox background color
            auto: Minimum rectangle
            scaleup: Only scale down, do not scale up (for better val mAP)
            stride: Stride for scaleup
        """

        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # print(new_unpad)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        im = image
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # print(im.shape)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        resized_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                         value=color)  # add border

        return resized_img, np.array((dw, dh), dtype='float64'), r
    

    def _prepocess(self, image):
        img, dwdh, ratio = self._letterbox(image)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img / 255)

        return img, dwdh, ratio


    def _postprocess(self, outputs):
        predictions = np.squeeze(outputs).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)

        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold] 

        # Get max probability element
        max_index_score = np.where(scores == max(scores))

        # Get bounding boxes for each object
        boxes = predictions[:, :4]

        # rescale box
        input_shape = np.array([
            self.input_width, self.input_height, 
            self.input_width, self.input_height
        ])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([
            self.resized_height, self.resized_width, 
            self.resized_height, self.resized_width
        ])

        return boxes[max_index_score], scores[max_index_score]
    

    def _xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y


    def detect(self, image):
        start_preprocess = time.time()
        input_tensor, dwdh, ratio  = self._prepocess(image)
        took_preprocess = (time.time() - start_preprocess) * 1000

        start_detecting = time.time()
        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        took_detecting = (time.time() - start_detecting) * 1000

        start_postprocess = time.time()
        result_box, result_score = self._postprocess(outputs)
        took_postprocess  = (time.time() - start_postprocess) * 1000

        return {
            'box': self.rescale_prediction(self._xywh2xyxy(result_box[0]), dwdh, ratio),
            'prob': result_score[0],
            'speed': {
                'preprocess': f'{took_preprocess:.3f}',
                'detecting': f'{took_detecting:.3f}',
                'postprocess': f'{took_postprocess:.3f}',
                'total': f'{(took_preprocess + took_detecting + took_postprocess):.3f}'
            }
        }


if __name__ == '__main__':
    import pprint


    y = Yolo()
    image = cv2.imread('/home/bush/project/anpr/test/T008AT197.jpg')
    result = y.detect(image)
    pprint.pprint(result)
    box = result['box']
    cv2.imwrite('tmp.jpg', image[box[1]:box[3],box[0]:box[2]])