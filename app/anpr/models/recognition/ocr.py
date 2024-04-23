import os
import time

import cv2
import numpy as np
import onnxruntime


class Ocr:
    def __init__(self, device) -> None:
        self.chars = [
            "blank",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "E",
            "H",
            "K",
            "M",
            "O",
            "P",
            "T",
            "X",
            "Y",
        ]
        self.conf_thresold = 0.8

        module_dir = os.path.dirname(__file__)
        model_path = os.path.join(module_dir, "weights", "svtr.onnx")

        # EP_list = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        if device == "gpu":
            EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            EP_list = ["CPUExecutionProvider"]
        elif device == "trt":
            EP_list = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        self.ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)

        model_inputs = self.ort_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.input_height, self.input_width = input_shape[2:]

        model_output = self.ort_session.get_outputs()
        self.output_names = [model_output[i].name for i in range(len(model_output))]

    def _prepocess(self, image):
        imgC, imgH, imgW = [3, self.input_height, self.input_width]  # 48 320

        resized_image = cv2.resize(image, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255

        resized_image -= 0.5
        resized_image /= 0.5

        norm_img = resized_image[np.newaxis, :]

        return norm_img, imgH, imgW

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.chars[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def _postprocess(self, outputs, image_height, image_width):
        preds_idx = outputs.argmax(axis=2)
        preds_prob = outputs.max(axis=2)

        res = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        return {"text": res[0][0], "prob": res[0][1]}

    def recognize(self, image):
        start_preprocess = time.time()
        input_tensor, image_height, image_width = self._prepocess(image)
        took_preprocess = time.time() - start_preprocess

        start_detecting = time.time()
        outputs = self.ort_session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )[0]
        took_recognizing = time.time() - start_detecting

        start_postprocess = time.time()
        result = self._postprocess(outputs, image_height, image_width)
        took_postprocess = time.time() - start_postprocess

        result["speed"] = {
            "preprocess": f"{took_preprocess:.3f}",
            "recognizing": f"{took_recognizing:.3f}",
            "postprocess": f"{took_postprocess:.3f}",
            "total": f"{(took_preprocess + took_recognizing + took_postprocess):.3f}",
        }

        return result


if __name__ == "__main__":
    import pprint

    ocr = Ocr()
    image = cv2.imread("/home/ubuntu/proj/tmp.jpg")
    result = ocr.recognize(image)
    pprint.pprint(result)
    count = 0
    total = 0
    for i in range(1000):
        result = ocr.recognize(image)
        count += 1
        total += float(result["speed"]["total"])

    print(f"average speed, 1000 runs: {total / count} ms")
