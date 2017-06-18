import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np

X = np.load('/home/arunvaish9/PycharmProjects/TensorFlow/fer2013/X_train.npy')
test_x = np.load('/home/arunvaish9/PycharmProjects/TensorFlow/fer2013/X_privatetest.npy')
Y = np.load('/home/arunvaish9/PycharmProjects/TensorFlow/fer2013/y_train.npy')
test_y = np.load('/home/arunvaish9/PycharmProjects/TensorFlow/fer2013/y_privatetest.npy')

#X = X.reshape([-1, 28, 28, 1])
#test_x = test_x.reshape([-1, 28, 28, 1])

convnet = input_data(shape=[None, 48, 48, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True)



#model.save('quicktest.model')