import os

import anpr

if __name__ == "__main__":
    import pprint

    import cv2

    it = anpr.Anpr("cpu")  # also 'cpu' and 'trt' is available (trt not tested)

    # warmup
    image = cv2.imread("images/1702667219_E018EE77.jpg")
    pprint.pprint(it(image))

    count = 0
    total = 0
    num_runs = 2
    for i in range(num_runs):
        result = it(image)
        count += 1
        total += float(result["detection"]["speed"]["total"]) + float(
            result["recognition"]["speed"]["total"]
        )

    print(f"average speed, {num_runs} runs: {total / count} ms")
