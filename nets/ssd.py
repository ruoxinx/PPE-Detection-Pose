from keras.layers import (Activation, Conv2D, Flatten, Input, Reshape, concatenate)
from keras.models import Model
from nets.ssd_layers import Normalize, PriorBox
from nets.VGG16 import VGG16


def SSD300(input_shape, num_classes=21, anchors_size=[30,60,111,162,213,264,315]):
    input_tensor = Input(shape=input_shape)
    net = VGG16(input_tensor)

    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 4

    net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same', name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])

    net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[0], max_size=anchors_size[1], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])

    num_priors = 6

    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3,3),padding='same',name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])

    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3,3),padding='same',name='fc7_mbox_conf')(net['fc7'])
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[1], max_size=anchors_size[2], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])

    num_priors = 6

    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = x
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])

    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv6_2_mbox_conf')(net['conv6_2'])
    net['conv6_2_mbox_conf'] = x
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[2], max_size=anchors_size[3], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])

    num_priors = 6

    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = x
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])

    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf'] = x
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[3], max_size=anchors_size[4], aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])

    num_priors = 4

    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc'] = x
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv8_2_mbox_conf')(net['conv8_2'])
    net['conv8_2_mbox_conf'] = x
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])

    priorbox = PriorBox(input_shape, anchors_size[4], max_size=anchors_size[5], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])

    num_priors = 4

    x = Conv2D(num_priors * 4, kernel_size=(3,3), padding='same',name='conv9_2_mbox_loc')(net['conv9_2'])
    net['conv9_2_mbox_loc'] = x
    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    x = Conv2D(num_priors * num_classes, kernel_size=(3,3), padding='same',name='conv9_2_mbox_conf')(net['conv9_2'])
    net['conv9_2_mbox_conf'] = x
    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    
    priorbox = PriorBox(input_shape, anchors_size[5], max_size=anchors_size[6], aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')

    net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])

    net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                             net['fc7_mbox_loc_flat'],
                             net['conv6_2_mbox_loc_flat'],
                             net['conv7_2_mbox_loc_flat'],
                             net['conv8_2_mbox_loc_flat'],
                             net['conv9_2_mbox_loc_flat']],
                            axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                              net['fc7_mbox_conf_flat'],
                              net['conv6_2_mbox_conf_flat'],
                              net['conv7_2_mbox_conf_flat'],
                              net['conv8_2_mbox_conf_flat'],
                              net['conv9_2_mbox_conf_flat']],
                             axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                  net['fc7_mbox_priorbox'],
                                  net['conv6_2_mbox_priorbox'],
                                  net['conv7_2_mbox_priorbox'],
                                  net['conv8_2_mbox_priorbox'],
                                  net['conv9_2_mbox_priorbox']],
                                  axis=1, name='mbox_priorbox')

    net['mbox_loc'] = Reshape((-1, 4),name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((-1, num_classes),name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = concatenate([net['mbox_loc'],
                                    net['mbox_conf'],
                                    net['mbox_priorbox']],
                                    axis=2, name='predictions')

    model = Model(net['input'], net['predictions'])
    return model