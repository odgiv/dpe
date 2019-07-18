import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Layer, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, Concatenate, ZeroPadding2D, BatchNormalization, Dense, Activation, Lambda, Add
import numpy as np


class SymmetricPadding2D(Layer):
    def __init__(self, output_dim, kernel, padding=[1,1], data_format="channels_last", **kwargs):
        self.output_dim = output_dim
        self.data_format = data_format
        pad_size = (kernel - 1) // 2
        self.padding = (pad_size, pad_size)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SymmetricPadding2D, self).build(input_shape)

    def call(self, inputs):        
        if self.data_format == "channels_last":
            #(batch, depth, rows, cols, channels)
            pad = [[0,0]] + [[i,i] for i in self.padding] + [[0,0]]
        elif  self.data_format == "channels_first":
            #(batch, channels, depth, rows, cols)
            pad = [[0, 0], [0, 0]] + [[i,i] for i in self.padding]

        paddings = tf.constant(pad)
        return tf.pad(inputs, paddings, "SYMMETRIC")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class GlobalConcat(Layer):

    def __init__(self, concat_to_tensor_shape):
        self.concat_to_tensor_shape = concat_to_tensor_shape
        super(GlobalConcat, self).build()
    
    def build(self, input_shape):
        super(GlobalConcat, self).build(input_shape)

    def call(self, global_tensor, concat_to_tensor=None):
        """
        concat_to_tensor is in shape [n,32,32,128]
        global_tensor is in shape [n,1,1,128]
        """
        if concat_to_tensor is None:
            raise Exception("concat_to_tensor must not be None")

        h = tf.shape(concat_to_tensor)[1]
        w = tf.shape(concat_to_tensor)[2]

        global_tensor = tf.squeeze(global_tensor, [1,2])
        dims = global_tensor.get_shape()[-1]
        batch_unpacked = tf.unpack(global_tensor, axis=0)
        batch_repacked = []
        for batch in batch_unpacked:
            batch = tf.tile(batch, [h*w])
            batch = tf.reshape(batch, [h, w, -1])
            batch_repacked.append(batch)
        global_vector = tf.stack(batch_repacked)
        global_vector.set_shape(global_vector.get_shape().as_list()[:3] + [dims])
        tensor = tf.concat(3, [concat_to_tensor, global_tensor])

        return tensor


    def compute_output_shape(self, input_shape):
        return (*self.concat_to_tensor_shape[:-1], self.concat_to_tensor_shape[-1] * 2)



class DPE(Model):
    
    def init(self, img_h, img_w): #1365x2048
        concat_axis = 3
        self.img_h = img_h
        self.img_w = img_w
        
        # contracting path        
        #self.pad1 = SymmetricPadding2D(output_dim=(512, 512, 3), kernel=5, padding=[pad_size, pad_size], input_shape=(512, 512, 3))
        self.conv1 = Conv2D(16, (5,5), (1,1), padding="same", activation="selu") #1365x2048x16                
        self.batch1 = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size=(2, 2)) 

        self.conv2 = Conv2D(32, (5,5), (2,2), padding="same", activation="selu") #256x256x32                
        self.batch2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(2, 2)) 

        self.conv3 = Conv2D(64, (5,5), (2,2), padding="same", activation="selu") #128x128x64               
        self.batch3 = BatchNormalization()
        self.pool3 = MaxPooling2D(pool_size=(2, 2))

        self.conv4 = Conv2D(128, (5,5), (2,2), padding="same", activation="selu") #64x64x128               
        self.batch4 = BatchNormalization()
        self.pool4 = MaxPooling2D(pool_size=(2, 2))
        
        self.conv5 = Conv2D(128, (5,5), (2,2), padding="same", activation="selu") #32x32x128                               
        self.batch5 = BatchNormalization()
        
        # global features
        self.pool5 = MaxPooling2D(pool_size=(2, 2))

        self.conv6 = Conv2D(128, (5,5), (2,2), padding="same", activation="selu") #16x16x128                               
        self.batch6 = BatchNormalization()
        self.pool6 = MaxPooling2D(pool_size=(2, 2))

        self.conv7 = Conv2D(128, (5,5), (2,2), padding="same", activation="selu") #8x8x128                               
        self.batch7 = BatchNormalization()
        
        # fc -> selu -> fc
        self.conv8 = Conv2D(128, (8,8), (1,1), padding="valid", activation="selu") #1x1x128
        self.conv9 = Conv2D(128, (1,1), (1,1), padding="valid") #1x1x128
        
        # expanding path
        self.conv10 = Conv2D(128, (3,3), (1,1), padding="same")        
        self.global_concat = GlobalConcat((1, 512, 512, 128))
        self.conv11 = Conv2D(128, (1,1), (1,1), padding="same", activation="selu")        
        self.batch11 = BatchNormalization()

        self.conv12 = Conv2D(128, (3,3), (1,1), padding="same")
        self.resize12 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x.get_shape()*2), align_corners=True))
        self.concat12 = Concatenate(axis=concat_axis)
        self.act12 = Activation('selu')
        self.batch12 = BatchNormalization()

        self.conv13 = Conv2D(128, (3,3), (1,1), padding="same")
        self.resize13 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x.get_shape()*2), align_corners=True))
        self.concat13 = Concatenate(axis=concat_axis)
        self.act13 = Activation('selu')
        self.batch13 = BatchNormalization()

        self.conv14 = Conv2D(64, (3,3), (1,1), padding="same")
        self.resize14 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x.get_shape()*2), align_corners=True))
        self.concat14 = Concatenate(axis=concat_axis)
        self.act14 = Activation('selu')
        self.batch14 = BatchNormalization()

        self.conv15 = Conv2D(32, (3,3), (1,1), padding="same")
        self.resize15 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(x.get_shape()*2), align_corners=True))
        self.concat15 = Concatenate(axis=concat_axis)
        self.act15 = Activation('selu')
        self.batch15 = BatchNormalization()

        self.conv16 = Conv2D(16, (3,3), (1,1), padding="same")
        self.act16 = Activation('selu')
        self.batch16 = BatchNormalization()
                
        self.conv17 = Conv2D(3, (3,3), (1,1), padding="same")

        self.add = Add()
                

    def call(self, inputs):      

        # contracting path
        dpe_conv1 = self.conv1(inputs)
        dpe_batch1 = self.batch1(dpe_conv1)
        dpe_pool1 = self.pool1(dpe_batch1)

        dpe_conv2 = self.conv2(dpe_pool1)
        dpe_batch2 = self.batch2(dpe_conv2)
        dpe_pool2 = self.pool2(dpe_batch2)

        dpe_conv3 = self.conv3(dpe_pool2)
        dpe_batch3 = self.batch3(dpe_conv3)
        dpe_pool3 = self.pool3(dpe_batch3)

        dpe_conv4 = self.conv4(dpe_pool3)
        dpe_batch4 = self.batch4(dpe_conv4)
        dpe_pool4 = self.pool4(dpe_batch4)

        dpe_conv5 = self.conv5(dpe_pool4)
        dpe_batch5 = self.batch5(dpe_conv5)
        dpe_pool5 = self.pool5(dpe_batch5)

        # global features
        dpe_conv6 = self.conv6(dpe_pool5)
        dpe_batch6 = self.conv6(dpe_conv6)
        dpe_pool6 = self.pool6(dpe_batch6)

        dpe_conv7 = self.conv7(dpe_pool6)
        dpe_batch7 = self.conv7(dpe_conv7)
        

        dpe_conv8 = self.conv8(dpe_batch7)
        dpe_conv9 = self.conv9(dpe_conv8)        

        # global concat
        dpe_conv10 = self.conv10(dpe_batch5)
        dpe_global_concat = self.global_concat(dpe_conv10, dpe_conv9)
        dpe_conv11 = self.conv11(dpe_global_concat)
        dpe_batch11 = self.batch11(dpe_conv11)

        dpe_conv12 = self.conv12(dpe_batch11)
        # resize 
        dpe_resized12 = self.resize12(dpe_conv12)
        # concat 
        dpe_concat12 = self.concat12([dpe_resized12, dpe_batch5])
        # non-linear
        dpe_act12 = self.act12(dpe_concat12)
        

        dpe_conv13 = self.conv13(dpe_act12)
        # resize 
        dpe_resized13 = self.resize13(dpe_conv13)
        # concat 
        dpe_concat13 = self.concat13([dpe_resized13, dpe_batch4])
        # non-linear
        dpe_act13 = self.act13(dpe_concat13)


        dpe_conv14 = self.conv14(dpe_act13)
        # resize 
        dpe_resized14 = self.resize14(dpe_conv14)
        # concat 
        dpe_concat14 = self.concat14([dpe_resized14, dpe_batch3])
        # non-linear
        dpe_act14 = self.act14(dpe_concat14)


        dpe_conv15 = self.conv15(dpe_act14)
        # resize 
        dpe_resized15 = self.resize15(dpe_conv15)
        # concat 
        dpe_concat15 = self.concat15([dpe_resized15, dpe_batch2])
        # non-linear
        dpe_act15 = self.act15(dpe_concat15)
        dpe_batch15 = self.batch15(dpe_act15)

        dpe_conv16 = self.conv16(dpe_batch15)
        dpe_act16 = self.act16(dpe_conv16)
        dpe_batch16 = self.batch16(dpe_act16)
                
        dpe_conv17 = self.conv17(dpe_batch16)
        
        res_connect_axis = [0, 1, 2]
        l = [dpe_conv17[:, :, :, i] for i in res_connect_axis]
        res_tensor = tf.pack(l, -1)

        tensor = self.add([input, res_tensor])

        return tensor   