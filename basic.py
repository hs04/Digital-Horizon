import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
import os
import tensorflow
import matplotlib.pyplot as plt
from random import shuffle
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model
from keras.models import Model
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

pepsi = "/home/harsh/Desktop/base_model/pepsi/p/"
nutella = "/home/harsh/Desktop/base_model/nutella/n/"
test = "/home/harsh/Desktop/base_model/DH/test/"
model_path = "/home/harsh/Desktop/base_model/"

soaps = "/home/harsh/Desktop/base_model/DH/soaps/"
hairprods = "/home/harsh/Desktop/base_model/DH/hairproducts/"
softdrinks = "/home/harsh/Desktop/base_model/DH/softdrinks/"
icecream = "/home/harsh/Desktop/base_model/DH/icecream/"
beverages = "/home/harsh/Desktop/base_model/DH/beverages/"

biscuits = "/home/harsh/Desktop/base_model/DH/biscuits/"
dentalcare = "/home/harsh/Desktop/base_model/DH/dentalcare/"

def train_data():
	train_img = []
	for p in os.listdir(soaps):
		img = cv2.imread(soaps+p)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		b,g,r = cv2.split(img)
		img1 = cv2.imread(soaps+p)
		img1 = cv2.resize(img1,(64,64))
		train_img.append([img,[1,0,0,0,0]])
		
	for p in os.listdir(hairprods):
		img = cv2.imread(hairprods+p)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		img1 = cv2.imread(hairprods+p)
		img1 = cv2.resize(img1,(64,64))
		train_img.append([img,[0,1,0,0,0]])

	for p in os.listdir(softdrinks):
		img = cv2.imread(softdrinks+p)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		img1 = cv2.imread(softdrinks+p)
		img1 = cv2.resize(img1,(64,64))
		train_img.append([img,[0,0,1,0,0]])

	for p in os.listdir(icecream):
		img = cv2.imread(icecream+p)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		
		train_img.append([img,[0,0,0,1,0]])

	for p in os.listdir(beverages):
		img = cv2.imread(beverages+p)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		
		train_img.append([img,[0,0,0,0,1]])

	'''
	for p in os.listdir(biscuits):
		img = cv2.imread(biscuits+p)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		
		train_img.append([img,[0,0,0,0,0,1,0]])

	for p in os.listdir(dentalcare):
		img = cv2.imread(dentalcare+p)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		
		train_img.append([img,[0,0,0,0,0,0,1]])
	'''
	shuffle(train_img)
	return train_img

def test_img():
	test_img = []
	ohl = [0,0,0,1]
	for i in os.listdir(test):
		img = cv2.imread(test+i)
		img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
		img1 = cv2.imread(test+i)
		img1 = cv2.resize(img1,(64,64))
		if i[0]=="2":
			ohl = [1,0,0,0]
		elif i[0]=="3":
			ohl = [0,1,0,0]
		elif i[0]=="5":
			ohl = [0,0,1,0]
		test_img.append([img,ohl,i,img1])
	return test_img


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

training_img = train_data()
#testing_img = test_img()
testing_img = training_img[-100:]
training_img = training_img[:-100]



tr_data = np.array([i[0] for i in training_img]).reshape(-1,64,64,3)
tr_lbl = np.array([i[1] for i in training_img])
		
te_data = np.array([i[0] for i in testing_img]).reshape(-1,64,64,3)
te_lbl = np.array([i[1] for i in testing_img])


model1 = Sequential()

model1.add((InputLayer(input_shape = [64,64,3])))
model1.add(Conv2D(filters = 16,kernel_size = 5,padding='same',activation = 'relu'))
model1.add(MaxPool2D(pool_size=2,padding='same'))

model1.add(Conv2D(filters = 32,kernel_size = 3,padding='same',activation = 'relu'))
model1.add(MaxPool2D(pool_size=2,padding='same'))

#model1.add(Conv2D(filters = 32,kernel_size = 3,padding='same',activation = 'relu'))
#model1.add(MaxPool2D(pool_size=2,padding='same'))

model1.add(Conv2D(filters = 64,kernel_size = 3,padding='same',activation = 'relu'))
model1.add(MaxPool2D(pool_size=2,padding='same'))

#model1.add(Conv2D(filters = 64,kernel_size = 3,padding='same',activation = 'relu'))
#model1.add(MaxPool2D(pool_size=3,padding='same'))

#model1.add(Conv2D(filters = 80,kernel_size = 3,padding='same',activation = 'relu'))
#model1.add(MaxPool2D(pool_size=3,padding='same'))

#model1.add(Conv2D(filters = 100,kernel_size = 3,padding='same',activation = 'relu'))
#model1.add(MaxPool2D(pool_size=3,padding='same'))

model1.add(Conv2D(filters = 128,kernel_size = 3,padding='same',activation = 'relu'))
model1.add(MaxPool2D(pool_size=2,padding='same'))

model1.add(Conv2D(filters = 256,kernel_size = 2,activation = 'relu'))
#model1.add(MaxPool2D(pool_size=2,padding='same'))

#model1.add(Conv2D(filters = 256,kernel_size = 3,padding='same',activation = 'relu'))
#model1.add(MaxPool2D(pool_size=3,padding='same'))

model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(256,activation='relu'))
model1.add(Dropout(rate = 0.25))
model1.add(Dense(128,activation='relu'))
model1.add(Dropout(rate = 0.25))
model1.add(Dense(5,activation='softmax'))
optimizer = Adam(lr=8e-4)

callbacks = [
    EarlyStoppingByLossVal(monitor='loss', value=0.02, verbose=1)
]

model1.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
#model1.load_weights("/home/harsh/Desktop/base_model/90percent/model1x.h5")
model1.fit(x = tr_data,y=tr_lbl,epochs=80,batch_size=32,callbacks=callbacks)
model1.summary()
model1.save(model_path+'model1x8.h5')
plot_model(model1, to_file='model_x8.png',show_shapes=True,show_layer_names=True)


#shuffle(testing_img)

n = 0
x = 0
fig = plt.figure(figsize=(15,15))
store = {}

for cnt,data in enumerate(testing_img):
	clr = 1	
	x = x + 1
	y = fig.add_subplot(10,10,cnt+1)
	img = data[0]
	ans = data[1]
	
	datam = img.reshape(1,64,64,3)
	model_out= model1.predict([datam])
	
	confidence = 100*(model_out[0][np.argmax(model_out)]/np.sum(model_out))
	#layer = 'dense_2'
	#int_model = Model(inputs = model1.input,outputs=model1.get_layer(layer).output)
	#store[data[2]]=int_model.predict([datam])
	if np.argmax(model_out)==1:
		str_label = 'shampoo'
		if ans[1]==1:
			n = n+1
			clr = 0
	elif np.argmax(model_out)==2:
		str_label = 'softdrink'
		if ans[2]==1:
			n = n+1
			clr = 0
	elif np.argmax(model_out)==3:
		str_label = 'ice cream'
		if ans[3]==1:
			n = n+1
			clr = 0
	elif np.argmax(model_out)==4:
		str_label = 'beverages'
		if ans[4]==1:
			n = n+1
			clr = 0
	else:
		str_label = 'soap'
		if ans[0]==1:
			n = n+1
			clr = 0
	str_label = str_label + "\n("+str(round(confidence,1))+" %)"
	y.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
	if clr==0:
		plt.title(str_label)
	else:
		obj = plt.title(str_label)
		plt.setp(obj,color='r')
		

'''	
	elif np.argmax(model_out)==5:
		str_label = 'biscuits'
		if ans[5]==1:
			n = n+1
	
	elif np.argmax(model_out)==6:
		str_label = 'dental care'
		if ans[6]==1:
			n = n+1
'''
	

plt.show()
print(n,x)
np.save("features.npy", store)










