from data.models import Face
from face_recognition import face_locations, face_encodings
import base64
from cv2 import imencode, cvtColor, COLOR_BGR2RGB


def imagearray_to_b64(imagearray):
    imageRGB = cvtColor(imagearray, COLOR_BGR2RGB)
    jpg = imencode('.jpg', imageRGB)[1]
    b64_string = base64.b64encode(jpg).decode('utf-8')
    return b64_string


def get_image_faces(image_narray, known_faces_array):
    face_locations_list = face_locations(image_narray)
    if not face_locations_list:
        return []
    else:
        face_encodings_list = face_encodings(
            image_narray, face_locations_list, model="large")
        faces_list = []
        # Iterating through each face found in image
        for i in range(len(face_locations_list)):
            face_location = face_locations_list[i]
            face_encoding = face_encodings_list[i]
            top, right, bottom, left = face_location
            face_image = image_narray[top:bottom, left:right]
            face = Face(top, right, bottom, left, face_encoding.tolist(),
                        imagearray_to_b64(face_image))
            # Comparing the face models to the known ones and adding id if
            # it corresponds
            face.compare(known_faces_array)
            faces_list.append(face.to_dict())
        return faces_list
