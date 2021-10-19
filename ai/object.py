from imageai.Classification import ImageClassification
import os


def load_object_detection_model():
    execution_path = os.getcwd()
    prediction = ImageClassification()
    prediction.setModelTypeAsResNet50()
    prediction.setModelPath(os.path.join(
        execution_path, "ai"+os.sep+"models" +
        os.sep+"resnet50_imagenet_tf.2.0.h5"))
    prediction.loadModel()
    return prediction


prediction = load_object_detection_model()


def get_image_objects(image_narray, threshold=30):
    predictions, probabilities = prediction.classifyImage(
        image_narray, input_type="array")
    objects_list = []
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        if eachProbability > threshold:
            objects_list.append(eachPrediction)
    return objects_list
