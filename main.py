import cv2
import numpy as np
import requests
from urllib.request import urlopen

INPUT_VIDEO_PATH = "input_feed/havnegade_cut1.mp4"
OUTPUT_VIDEO_PATH = "output_feed/cartoon_havnegade.avi"


def transfer_style(source):

    cv2.imwrite('cache.png', source)

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


def apply_sketch(source):

    horizontal = cv2.Sobel(source, 0, 1, 0, cv2.CV_64F)
    vertical = cv2.Sobel(source, 0, 0, 1, cv2.CV_64F)

    return cv2.bitwise_or(horizontal, vertical)


def apply_cartoon(source):

    output = np.uint8(source)
    output[:] = ((np.round((output / 255) * 10) / 10) * 255)

    horizontal = cv2.Sobel(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), 0, 1, 0, cv2.CV_64F)
    vertical = cv2.Sobel(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), 0, 0, 1, cv2.CV_64F)
    edge_image = cv2.bitwise_or(horizontal, vertical)

    output[edge_image > 50] = (0, 0, 0)

    return output


def apply_watercolor(target):

    return cv2.stylization(target, sigma_s=60, sigma_r=0.6)


def main():
    input_feed = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = input_feed.get(cv2.CAP_PROP_FPS)
    frame_count = input_feed.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = None

    out = None

    index = 0.0
    while input_feed.isOpened():
        _, frame = input_feed.read()
        if frame is None or cv2.waitKey(int(1 / fps * 1000)) & 0xFF == ord('q'):
            break

        stylized_frame = apply_cartoon(frame)

        if frame_size is None:
            frame_size = (stylized_frame.shape[1], stylized_frame.shape[0])
            out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'DIVX'), 24, frame_size)

        out.write(stylized_frame)

        index += 1
        print("{} / {} {:10.2f}%".format(index, frame_count, index / frame_count * 100))

    out.release()
    input_feed.release()

    cv2.destroyAllWindows()

    print("Frame size: {}".format(frame_size))


main()
