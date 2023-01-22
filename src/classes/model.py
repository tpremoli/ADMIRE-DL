from tensorflow.keras.preprocessing import image
import tensorflow.keras.applications as apps
import numpy as np


class Model:
    def __init__(self, kaggle=False):
        self.model = None

# this is how its gonna work roughly
class VGG16Model(Model):
    def __init__(self, kaggle=False):
        super().__init__(kaggle)

class VGG19Model(Model):
    def __init__(self, kaggle=False):
        super().__init__(kaggle)