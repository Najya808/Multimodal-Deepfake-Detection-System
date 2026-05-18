import numpy as np
from facenet_pytorch import MTCNN

detector = MTCNN(keep_all=True)

def detect_faces(img):

    img = np.array(img)

    # RGBA → RGB fix
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    boxes, _ = detector.detect(img)

    faces = []

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = [int(v) for v in box]

            face = img[y1:y2, x1:x2]

            faces.append(face)

    return faces