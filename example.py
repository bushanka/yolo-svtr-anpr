import anpr


if __name__ == '__main__':
    import pprint
    import cv2


    it = anpr.Anpr('gpu')

    # warmup
    image = cv2.imread('/home/ubuntu/proj/H560PA78.jpg')
    pprint.pprint(it(image))

    count = 0
    total = 0
    for i in range(10):
        result = it(image)
        count += 1
        total += float(result['detection']['speed']['total']) + float(result['recognition']['speed']['total'])

    print(f'average speed, 1000 runs: {total / count} ms')