import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout

from tensorflow.keras.layers import Input, concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import layers
#from models.FPN

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

class Sem_Seg_Head(Model):
    def __init__(self):
        super(Sem_Seg_Head, self).__init__()
        self.dropout = Dropout(rate=0.01)
        self.conv = Conv2D(filters=20, kernel_size=3, strides=1, padding='same', data_format='channels_last')
        self.sem_softmax = Softmax(axis=3) 

    def call(self, x):
        y = self.dropout(x)
        y = self.conv(y)
        y = self.softmax(y)
        return y

if __name__ == '__main__':
    sem_seg_head = Sem_Seg_Head()
    sem_seg_head.build(input_shape=(None, 32, 64, 1024))
    sem_seg_head.call(Input(shape=(32, 64, 1024)))
    sem_seg_head.summary()
