import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Softmax, Input
from tensorflow.keras.models import Model
#from models.encoder import Encoder
#from models.decoder import Decoder
from models.FPN import FPN
from models.Semantic_Segmentation_Head import Sem_Seg_Head
#Import models.Instance_Segmentation_Head import Inst_Seg_Head
#Import models.Panoptic_Segmentation_Head import Pano_Seg_Head


INPUT_DEPTH = 5
INPUT_HEIGHT = 64
INPUT_WIDTH = 1024

#Panoptic Segmentation Pipeline
class PanopticModel(Model): 
    def __init__(self, inp_shape=(None, 64, 1024, 5), rnn_flag=False, pixel_shuffle=False): 
        super(PanopticModel, self).__init__()
        self.feature_extractor = FPN() #Extracted the feature maps C1, C2, C3, C4, C5 #Added FPN call
        self.semantic_head = Sem_Seg_Head() #self.FPN
        self.instance_head = Inst_Seg_Head()
        #self.sem_rnn_flag = rnn_flag
        # write and add self.panoptic_head = Pano_Seg_Head()
        
        #Panoptic Head call 
        return y
    
    def summary(self):
        super(PanopticModel, self).summary()
        count = 24
        print("\n\n"+"="*count+" Encoder Summary "+"="*count+"\n\n")
        self.encoder.summary()
        print("\n\n"+"="*count+" Decoder Summary "+"="*count+"\n\n")
        self.decoder.summary()
        print("\n\n"+"="*count+" Semantic Head Summary "+"="*count+"\n\n")
        self.semantic_head.summary()
        #print("\n\n"+"="*count+" Instance Head _mmary "+"="*count+"\n\n")
        #self.instance_head.summary()
        #Add instance head.summary
        #self.panoptic_head.summary()
        #Add panoptic head.summary
        
    def call(self, x):  
    	self.FPN_rpn_feature_maps, self.FPN_mrcnn_feature_maps, self.final_feature = self.feature_extractor(x)      
        y_list_rpn  = self.FPN_rpn_feature_maps
        y_list_mrcnn = self.FPN_mrcnn_feature_maps
        y = self.final_feature        
        
        y_sem = self.semantic_head(y)
        y_inst = self.instance_head(rpn_feature_maps = y_list_rpn, mrcnn_feature_maps = y_list_mrcnn, final = y)
        
        return y_sem, y_inst

if __name__ == '__main__':
    panoptic_model = panoptic_model(rnn_flag=False, pixel_shuffle=False)
    panoptic_model.build(input_shape=(None, 64, 1024, 5))
    panoptic_model.call(Input(shape=(64, 1024, 5)))
    panoptic_model_model.summary()

