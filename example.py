import anpr


if __name__ == '__main__':
    import pprint
    import cv2


    it = anpr.Anpr()
    image = cv2.imread('/home/bush/project/anpr/test/T008AT197.jpg')
    pprint.pprint(it(image))