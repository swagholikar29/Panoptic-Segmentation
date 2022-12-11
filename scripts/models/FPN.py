import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Softmax, Input
from tensorflow.keras.models import Model
from models.encoder import Encoder
from models.decoder import Decoder

INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

#Segmentation
class FPN(Model): 
    def __init__(self, inp_shape=(None, 64, 1024, 5), rnn_flag=False, pixel_shuffle=False): #Remove rnn flag
        super(FPN, self).__init__()
        # self.inp = Input(shape=inp_shape)
        self.encoder = Encoder(pixel_shuffle=pixel_shuffle) 
        self.decoder = Decoder(pixel_shuffle=pixel_shuffle) 

    def call(self, x):        
        y = self.encoder(x)
        
        skips, os = self.encoder.get_skips()

        if self.rnn_flag:
            flag = False
            try:
                cur_y = y.numpy()
                flag = True
            except:
                pass
            if flag:
                if hasattr(self, 'prev_y'):
                    y = y + tf.constant(self.prev_y)
                self.prev_y = cur_y

        self.decoder.set_skips(skips, os)
        F, y = self.decoder(y)
        return F, y
    
    def summary(self):
        super(FPN, self).summary()
        count = 24
        print("\n\n"+"="*count+" Encoder Summary "+"="*count+"\n\n")
        self.encoder.summary()
        print("\n\n"+"="*count+" Decoder Summary "+"="*count+"\n\n")
        self.decoder.summary()

if __name__ == '__main__':
    FPN_model = FPN(pixel_shuffle=False)
    FPN.build(input_shape=(None, 64, 1024, 5))
    FPN.call(Input(shape=(64, 1024, 5)))
    FPN.summary()

