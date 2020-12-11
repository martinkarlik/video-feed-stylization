import cv2
import numpy as np
import requests
from urllib.request import urlopen


def transfer_style(target):

    cv2.imwrite('cache.png', target)

    result = requests.post(
        "https://api.deepai.org/api/fast-style-transfer",
        files={
            'content': open('cache.png', 'rb'),
            'style': open('styles/ugly_duckling.gif', 'rb'),
        },
        headers={'api-key': 'abc5a0b2-9d44-4e93-9e85-baaeb177f809'}
    )

    url = urlopen(result.json()["output_url"])
    image = np.asarray(bytearray(url.read()), dtype=np.uint8)

    return cv2.imdecode(image, -1)


def apply_sketch(target):

    horizontal = cv2.Sobel(target, 0, 1, 0, cv2.CV_64F)

    # the thresholds are like
    # (variable,0,<x axis>,<y axis>,cv2.CV_64F)
    vertical = cv2.Sobel(target, 0, 0, 1, cv2.CV_64F)

    # DO the Bitwise operation
    return cv2.bitwise_or(horizontal, vertical)


def apply_watercolor(target):

    return cv2.stylization(target, sigma_s=60, sigma_r=0.6)



def main():
    input_feed = cv2.VideoCapture("input_feed/havnegade_cut1.mp4")
    output_feed = []
    fps = input_feed.get(cv2.CAP_PROP_FPS)
    frame_count = input_feed.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = None

    index = 0.0
    while input_feed.isOpened():
        _, frame = input_feed.read()
        if frame is None or cv2.waitKey(int(1 / fps * 1000)) & 0xFF == ord('q'):
            break

        stylized_frame = apply_watercolor(frame)
        output_feed.append(stylized_frame)

        if frame_size is None:
            frame_size = (stylized_frame.shape[1], stylized_frame.shape[0])

        index += 1
        print("{} / {} {:10.2f}%".format(index, frame_count, index / frame_count * 100))



    input_feed.release()
    cv2.destroyAllWindows()

    print(frame_size)
    out = cv2.VideoWriter('output_feed/watercolor_havnegade_cut1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, frame_size)

    for i in range(len(output_feed)):
        out.write(output_feed[i])
    out.release()


main()
