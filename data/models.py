from face_recognition import compare_faces
import numpy as np


class Face:

    def __init__(self, top, right, bottom, left, model, image):
        self.location = {'top': top, 'right': right,
                         'bottom': bottom, 'left': left}
        self.model = model
        self.image = image
        self.id = None

    def compare(self, known_faces_array):
        for known_face in known_faces_array:
            results = compare_faces(
                [np.array(known_face["model"])], np.array(self.model))
            if results[0]:
                self.id = known_face["id"]
                break

    def to_dict(self):
        return {"location": self.location,
                "model": self.model,
                "image": self.image,
                "id": self.id}
