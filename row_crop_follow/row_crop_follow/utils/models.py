from tensorflow.keras.models import Model,load_model

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Activation, Input, Add, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape,Dropout, Multiply, Flatten,UpSampling2D
from tensorflow.keras.losses import BinaryCrossentropy


from tensorflow.keras.metrics import MeanIoU
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

# some global constants
activation_number=41


def build_model_multi(base_model, dropout_rate, n_class): 
    global activation_number
    #1/8 resolution output
    
    out_1_8= base_model.get_layer('activation_15').output
    
    #1/16 resolution output
    
    out_1_16= base_model.get_layer('activation_29').output
    
    
    # branch1
    x1 = Conv2D(128, (1, 1))(out_1_16)
    x1 = BatchNormalization()(x1)
    
    layer_name_act="activation_head"+str(activation_number)
    x1 = Activation('relu',name=layer_name_act)(x1)
    activation_number+=1
    
    # branch2
    s = x1.shape

    #custom average pooling2D
    x2 = AveragePooling2D(pool_size=(12, 12), strides=(4, 5),data_format='channels_last')(out_1_16)
    x2 = Conv2D(128, (1, 1))(x2)
    
    
    layer_name_act="activation_head"+str(activation_number)
    
    x2 = Activation('sigmoid',name=layer_name_act)(x2)
    activation_number+=1
    

    x2 = UpSampling2D(size=(int(s[1]), int(s[2])),data_format='channels_last',interpolation="bilinear")(x2)

    
    
    # branch3
    x3 = Conv2D(n_class, (1, 1))(out_1_8)
    
    # multiply
    m1 = Multiply()([x1, x2])

    m1 = UpSampling2D(size=(2, 2),data_format='channels_last',interpolation="bilinear")(m1)
    m1 = Conv2D(n_class, (1, 1), name='last_conv')(m1)

    # add
    m2 = Add()([m1, x3])

    
    #adding this UPsampling of factor 8
    predictions = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")(m2)
    
    
    activation_number+=1 


    # final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


def build_model_binary(base_model, dropout_rate, n_class): 
    global activation_number
    #1/8 resolution output
    
    out_1_8= base_model.get_layer('activation_15').output
    
    #1/16 resolution output
    
    out_1_16= base_model.get_layer('activation_29').output
    
    
    # branch1
    x1 = Conv2D(128, (1, 1))(out_1_16)
    x1 = BatchNormalization()(x1)
    
    layer_name_act="activation_head"+str(activation_number)
    x1 = Activation('relu',name=layer_name_act)(x1)
    activation_number+=1
    
    # branch2
    s = x1.shape

    #custom average pooling2D # Diego pool_size=(12,12) strides=(4,5)
    x2 = AveragePooling2D(pool_size=(12, 12), strides=(4, 5),data_format='channels_last')(out_1_16)
    x2 = Conv2D(128, (1, 1))(x2)
    
    layer_name_act="activation_head"+str(activation_number)
    
    x2 = Activation('sigmoid',name=layer_name_act)(x2)
    activation_number+=1
    

    x2 = UpSampling2D(size=(int(s[1]), int(s[2])),data_format='channels_last',interpolation="bilinear")(x2)

    
    
    # branch3
    x3 = Conv2D(n_class, (1, 1))(out_1_8)
    
    # multiply
    m1 = Multiply()([x1, x2])

    m1 = UpSampling2D(size=(2, 2),data_format='channels_last',interpolation="bilinear")(m1)
    m1 = Conv2D(n_class, (1, 1), name='last_conv')(m1)

    # add
    m2 = Add()([m1, x3])

    
    #adding this UPsampling of factor 8
    predictions = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")(m2)
    m2 = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")(m2)

    
    # predictions 
    layer_name_act="activation_head"+str(activation_number)
    predictions = Activation('sigmoid',name=layer_name_act)(m2)
    
    
    activation_number+=1 


    # final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model
