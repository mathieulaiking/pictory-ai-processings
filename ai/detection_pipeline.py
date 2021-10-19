from ai.face import get_image_faces
from ai.object import get_image_objects
from ai.scene import get_image_places
from image_processing.blur_detector import detect
import timeit


def start_detections(image_narray, image_dict, known_faces_array):
    start = timeit.default_timer()
    # Getting faces locations , encodings and id if face already known
    image_dict["faces"] = get_image_faces(image_narray, known_faces_array)
    facet = timeit.default_timer()
    # Getting objects detected in image
    image_dict["objects"] = get_image_objects(image_narray)
    objectt = timeit.default_timer()
    # Detecting blur in picture and adding it to dict
    image_dict["blur_score"] = detect(image_narray)
    blurt = timeit.default_timer()
    # Detecting picture scene
    image_dict["scenes"] = get_image_places(image_narray)
    scenet = timeit.default_timer()
    # Printing message sent and execution time
    print("\nFACES ID DETECTED", [face["id"]
                                  for face in image_dict["faces"]])
    print("\nOBJECTS DETECTED: ", image_dict["objects"])
    print("\nSCENES DETECTED: ", image_dict["scenes"])
    print("\nBLUR SCORE (below 10 is blurred) : ", image_dict["blur_score"])
    times = [start, facet, objectt, blurt, scenet]
    tlabels = ["FACES RECOGNITION", "OBJECT RECOGNITION",
               "BLUR DETECTION", "SCENE DETECTION"]
    for i in range(1, len(times)):
        print("\n", tlabels[i-1], " TIME : ", str(
            times[i] - times[i-1])[:4], " s")
    print("\nTOTAL IMAGE PROCESSING TIME : ",
          str(scenet - start)[:4], " s\n\n")

    return image_dict
