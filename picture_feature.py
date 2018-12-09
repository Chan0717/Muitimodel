# -*- coding: utf-8 -*-
from process_picture import *
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import load_model, Model

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

pictures = get_four_pictures()
new_pictures = []
for picture in pictures:
    picture = np.array(picture) / 255.0
    new_pictures.append(picture)
new_pictures = np.array(new_pictures)
print new_pictures.shape

# model = InceptionResNetV2()
# weight_path = './pretrained_model/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
# model = Xception(include_top=False, weights=weight_path,
#                                     input_tensor=None, input_shape=(224, 224, 3),
#                                     pooling='max', classes=1000)
# model = VGG19(weights='imagenet')
# model = load_model('model_test_Keras_02(CNN-3-mnist).h5')
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])
#
# layer = Model(inputs=model.input, outputs=model.get_layer(name='feature').output)
# model = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
# features = model.predict(new_pictures)
# print features.shape
# np.savetxt('img_feature', new_pictures)
