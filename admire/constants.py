import os

# ADNI group constants
COGNITIVELY_NORMAL="CN"
ALZHEIMERS_DISEASE="AD"
MILD_COGNITIVE_IMPAIRMENT="MCI"

ADNI_IMAGE_DIMENSIONS=[182,218]

# FSL constants
FSLDIR = os.getenv('FSLDIR')

# Supported architectures
VGG_16 = "VGG16"
VGG_19 = "VGG19"

RES_NET_50 = "ResNet50"
RES_NET_152 = "ResNet152"

DENSE_NET_121 = "DenseNet121"
DENSE_NET_201 = "DenseNet201"

# Keras app package name
KERAS_APP_PKG = {
    VGG_16: "vgg16",
    VGG_19: "vgg19",
    RES_NET_50: "resnet",
    RES_NET_152:"resnet",
    DENSE_NET_121:"densenet",
    DENSE_NET_201:"densenet"
}