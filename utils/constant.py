import numpy as np

PASCAL_VOC_CLASSES = {
    0: "background",
    1: "airplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "table",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted_plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv",
    21: "void",
}

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])