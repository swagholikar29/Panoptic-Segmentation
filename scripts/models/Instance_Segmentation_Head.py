import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout

from tensorflow.keras.layers import Input, concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import layers
#from models.FPN
import models.model.MaskRCNN as MaskRCNN 

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

class Inst_Seg_Head(Model):
    def __init__(self): #Input Shape et. al
        super(Inst_Seg_Head, self).__init__()
        self.mask_rcnn_meta = MaskRCNN() #Invoking the model here because it calls on the build function in its constructor

    def call(self, y_list_rpn, y_list_mrcnn, y):
        y = mask_rcnn_meta.build(y_list_rpn, y_list_mrcnn, y)
        y = 
        return y

if __name__ == '__main__':
    # tf.enable_eager_execution()

    inst_seg_head = Inst_Seg_Head()
    inst_seg_head.call(Input(shape=(32, 64, 1024)))
    inst_seg_head.summary()
