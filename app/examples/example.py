import anpr

if __name__ == '__main__':
    import pprint
    import cv2

    it = anpr.Anpr('cpu')  # also 'cpu' and 'trt' is available (trt not tested)

    # warmup
    image = cv2.imread('/home/bush/project/anpr/photo_2023-10-16_13-37-16.jpg')
    pprint.pprint(it(image))

    # count = 0
    # total = 0
    # for i in range(10):
    #     result = it(image)
    #     count += 1
    #     total += float(result['detection']['speed']['total']) + float(result['recognition']['speed']['total'])

    # print(f'average speed, 1000 runs: {total / count} ms')
