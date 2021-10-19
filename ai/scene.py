import ai.wideresnet as wideresnet
import os
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
from PIL import Image


execution_path = os.getcwd()


def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = os.path.join(
        execution_path, "ai"+os.sep+"models" +
        os.sep+"categories_places365.txt")
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = os.path.join(
        execution_path, "ai"+os.sep+"models" +
        os.sep+"IO_places365.txt")
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = os.path.join(
        execution_path, "ai"+os.sep+"models" +
        os.sep+"labels_sunattribute.txt")

    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = os.path.join(
        execution_path, "ai"+os.sep+"models" +
        os.sep+"W_sceneattribute_wideresnet18.npy")
    W_attribute = np.load(file_name_W)
    return classes, labels_IO, labels_attribute, W_attribute


def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = os.path.join(
        execution_path, "ai"+os.sep+"models" +
        os.sep+"wideresnet18_places365.pth.tar")

    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(
        model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k,
                  v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()

    features_names = ['layer4', 'avgpool']
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


# load the labels
classes, labels_IO, labels_attribute, W_attribute = load_labels()

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF()  # image transformer


def get_image_places(image_narray, category_array_length=5,
                     scene_probability_threshold=0):
    img = Image.fromarray(image_narray)
    input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]])  # vote for the indoor or outdoor
    io_value = None
    if io_image < 0.5:
        io_value = "indoor"
    else:
        io_value = "outdoor"

    # output the prediction of scene category
    category_array = []
    for i in range(0, category_array_length):
        if probs[i] >= scene_probability_threshold:
            category_array.append(classes[idx[i]])

    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    attribute_array = [labels_attribute[idx_a[i]] for i in range(-1, -10, -1)]

    return {"io": io_value, "categories": category_array,
            "attributes": attribute_array}
