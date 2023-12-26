from typing import Union
import numpy as np
from PIL import Image
import cv2


class Facesegmenter():
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def __call__(self, image: Union[str, np.ndarray, Image.Image]):
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)

        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        face = self.face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 10))
        plt.imshow(img)
        plt.axis('off')
        return


if __name__ == '__main__':
    segmenter = Facesegmenter()
    segmenter('/Users/istvanfe/Downloads/art-7944154.jpg')