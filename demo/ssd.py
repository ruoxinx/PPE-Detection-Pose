# A Keras implementation (Tensorflow backend) for PPE detection inspired by [bubbliiiing]
import colorsys
import os
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageDraw, ImageFont

from nets import ssd
from utils.ssd_utils import BBoxUtility, letterbox_image, ssd_correct_boxes

class SSD(object):
    _defaults = {
        "model_path"        : '../model/trained_ssd_weights.h5',
        "classes_path"      : '../model/ppe_classes.txt',
        "input_shape"       : (300, 300, 3),
        "confidence"        : 0.5,
        "nms_iou"           : 0.5,
        'anchors_size'      : [30,60,111,162,213,264,315]
    }

    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session()
        self.generate()
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.num_classes = len(self.class_names) + 1
        self.ssd_model = ssd.SSD300(self.input_shape, self.num_classes, anchors_size=self.anchors_size)
        self.ssd_model.load_weights(self.model_path, by_name=True)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        crop_img = letterbox_image(image, (self.input_shape[1],self.input_shape[0]))
        photo = np.array(crop_img,dtype = np.float64)

        photo = preprocess_input(np.reshape(photo,[1, self.input_shape[0], self.input_shape[1], 3]))
        preds = self.ssd_model.predict(photo)

        results = self.bbox_util.detection_out(preds, confidence_threshold=self.confidence)
        
        if len(results[0])<=0:
            return image

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        

        boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)

        font = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)-1]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)-1])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)-1])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()