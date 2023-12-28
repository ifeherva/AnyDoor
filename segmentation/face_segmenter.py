from typing import Union
import numpy as np
from PIL import Image
import cv2


class FaceSegmenter:
    def __init__(self, min_size=50):
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.min_size = min_size

    def __call__(self, image: Union[str, np.ndarray, Image.Image]):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        face = self.face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_size, self.min_size)
        )

        result = np.zeros_like(image)[:, :, 0].astype(np.uint8)

        if len(face) > 0:
            # pick the largest area
            areas = [x[2]*x[3] for x in face]
            max_idx = np.argmax(areas)

            x, y, w, h = face[max_idx]
            result[y:y+h, x:x+h] = 1

        return result


if __name__ == '__main__':
    segmenter = FaceSegmenter()
    segmenter('/Users/istvanfe/Downloads/art-7944154.jpg')
