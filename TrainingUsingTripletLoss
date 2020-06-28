## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam
import keras.backend as K

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

## for visualizing 
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model
from keras.models import Model
import keras.backend as K


##
import os
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
from random import shuffle
import numpy as np

K.get_session().run(tf.global_variables_initializer())


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def get_triplet_mask(labels):
	# size of labels is nx1
	label = tf.keras.backend.get_value(labels)
	n = len(label)
	mask = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if label[i]==label[j]:
				mask[i][j]=1
	mask = np.reshape(mask,(1,n,n))
	return mask
	
def triplet_loss_semihard(y_true,y_pred):
	del y_true
	
	margin = 1.0
	labels = y_pred[:,:1]
	labels = tf.cast(labels,dtype='int32')
	
	embeddings = y_pred[:,1:]
	
	
	pw_dist = pairwise_distance(embeddings,squared=True)
	
	anchor_p = tf.expand_dims(pw_dist,2)	
	anchor_n = tf.expand_dims(pw_dist,1)
	
	loss = anchor_p + margin - anchor_n	
	
	mask = math_ops.equal(labels,array_ops.transpose(labels))

	print()	
	print("stat")
	print()

	mask = tf.to_float(mask)
	loss = tf.multiply(mask,loss)
	
	loss = tf.maximum(loss,0.0)
	
	valid_triplets = tf.to_float(tf.greater(loss,1e-16))
	num_positive_triplets = tf.reduce_sum(valid_triplets)
	num_valid_triplets = tf.reduce_sum(mask)
	frac = num_positive_triplets/(num_valid_triplets + 1e-16)
	
	loss = tf.reduce_sum(loss)/(num_positive_triplets+1e-16)
	
	return loss


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensror` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums



def triplet_loss_adapted_from_tf(y_true, y_pred):
    print(1)
    del y_true
    margin = 2.0
    labels = y_pred[:, :1]

 
    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]


    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size  
    batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    #generating semi-hard pairs
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    
    ### Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance


def hard_triplet_loss(y_true,y_pred):
	labels = y_pred[:,:1]
	labels = tf.cast(labels,dtype='int32')
	embeddings = y_pred[:,1:]

	pw_dist = pairwise_distance(embeddings,squared=True)
	
	mask_p = _get_anchor_positive_triplet_mask(labels)
	mask_p = tf.to_float(mask_p)
	mask_n = _get_anchor_negative_triplet_mask(labels)
	mask_n = tf.to_float(mask_n)
	
	anchor_positive_dist = tf.multiply(mask_p,pw_dist)
	hardest_pdist = tf.reduce_max(anchor_positive_dist,axis=1)
	
	max_anchor_negative_dist = tf.reduce_max(pw_dist,axis=1,keepdims=True)
	
	anchor_negative_dist =pw_dist + max_anchor_negative_dist*(1.0-mask_n)
	
	hardest_ndist = tf.reduce_min(anchor_negative_dist,axis=1)
	
	triplet_loss = tf.maximum(hardest_pdist-hardest_ndist + 0.5,0.0)
	
	return triplet_loss


	
def create_base_network(image_input_shape,embedding_size):
	'''
	input_image = Input(shape = image_input_shape)#28x28x1
	x = Flatten()(input_image)
	#x1 = Dense(512,activation='relu')(x)
	#x1 = Dense(512,activation='relu')(x1)
	#x1 = Dense(512,activation='relu')(x1)
	#x1 = Dense(512,activation='relu')(x1)
	#x1 = Dense(512,activation='relu')(x1)
	#x1 = Dense(512,activation='relu')(x1)
	#x1 = Dense(512,activation='relu')(x1)
	x3 = Dense(256,activation='relu')(x)
	x3 = Dropout(0.4)(x3)
	x3 = Dense(256,activation='relu')(x3)
	x3 = Dropout(0.4)(x3)
	x4 = Dense(256,activation='relu')(x3)
	#x4 = Dense(256,activation='relu')(x4)
	#x4 = Dense(256,activation='relu')(x4)
	#x4 = Dense(256,activation='relu')(x4)
	#x4 = Dense(256,activation='relu')(x4)
	#x4 = Dense(256,activation='relu')(x4)
	#x4 = Dense(256,activation='relu')(x4)
	#x4 = Dense(256,activation='relu')(x4)
	x4 = Dropout(0.4)(x4)	
	#x4 = Dense(128,activation='relu')(x4)
	x4 = Dense(128,activation='relu')(x4)
	x4 = Dropout(0.3)(x4)	
	x4 = Dense(128,activation='relu')(x4)	
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	#x4 = Dense(128,activation='relu')(x4)
	x4 = Dropout(0.3)(x4)
	#x4 = Dense(64,activation='relu')(x4)
	#x4 = Dense(64,activation='relu')(x4)
	#x4 = Dense(64,activation='relu')(x4)
	#x4 = Dense(64,activation='relu')(x4)
	x4 = Dense(64,activation='relu')(x4)
	x4 = Dense(64,activation='relu')(x4)	
	x4 = Dropout(0.2)(x4)
	x5 = Dense(embedding_size)(x4)
	'''
	model1 = Sequential()
	model1.add((InputLayer(input_shape = image_input_shape)))
	#model1.add(Conv2D(filters = 16,kernel_size = 3, activation = 'relu',padding='same'))
	model1.add(ZeroPadding2D(padding=1))
	model1.add(Conv2D(filters = 16,kernel_size = 3,strides = (3,3),activation = 'relu'))
	#model1.add(MaxPool2D(pool_size=2,padding='same'))	
	model1.add(Conv2D(filters = 32,kernel_size = 3,activation = 'relu'))
	model1.add(Conv2D(filters = 32,kernel_size = 3, activation = 'relu',padding='same'))
	#model1.add(MaxPool2D(pool_size=2,padding='same'))
	model1.add(Conv2D(filters = 64,kernel_size = 3,activation = 'relu'))
	#model1.add(MaxPool2D(pool_size=2,padding='same'))
	model1.add(Conv2D(filters = 80,kernel_size = 3,activation = 'relu'))
	#model1.add(MaxPool2D(pool_size=2,padding='same'))
	model1.add(Conv2D(filters = 128,kernel_size = 3,activation = 'relu'))
	#model1.add(MaxPool2D(pool_size=2,padding='same'))
	model1.add(Conv2D(filters = 256,kernel_size = 2,activation = 'relu'))
	#model1.add(MaxPool2D(pool_size=2,padding='same'))
	model1.add(Dropout(0.25))
	model1.add(Flatten())
	#model1.add(Dense(128,activation='relu'))
	#model1.add(Dropout(rate = 0.10))
	#model1.add(Dense(256,activation='relu'))	
	#model1.add(Dropout(rate = 0.25))
	#model1.add(Dense(64,activation='relu'))
	#model1.add(Dropout(rate = 0.25))
	model1.add(Dense(embedding_size))
	print("SUMMARY")
	print(model1.summary())
	plot_model(model1, to_file='conv_trip.png',show_shapes=True,show_layer_names=True)
	'''
	base_net = Model(inputs=input_image,outputs=x5)
	base_net.summary()
	plot_model(base_net, to_file='base_network.png',show_shapes=True,show_layer_names=True)
	'''	
	return model1


pepsi = "/home/harsh/Desktop/base_model/pepsi/p/"#1
nutella = "/home/harsh/Desktop/base_model/nutella/n/"#2
biscuits = "/home/harsh/Desktop/base_model/biscuits/"#3
test = "/home/harsh/Desktop/base_model/test_img/"

soaps = "/home/harsh/Desktop/base_model/DH/soaps/"
hairprods = "/home/harsh/Desktop/base_model/DH/hairproducts/"
softdrinks = "/home/harsh/Desktop/base_model/DH/softdrinks/"
icecream = "/home/harsh/Desktop/base_model/DH/icecream/"


train_img = []
test_img = []

'''
for p in os.listdir(soaps):
	img = cv2.imread(soaps+p)
	img = cv2.resize(img,(28,28))
	train_img.append([[1.0],img])
	
for p in os.listdir(hairprods):
	img = cv2.imread(hairprods+p)
	img = cv2.resize(img,(28,28))
	train_img.append([[2.0],img])

for p in os.listdir(softdrinks):
	img = cv2.imread(softdrinks+p)
	img = cv2.resize(img,(28,28))
	train_img.append([[3.0],img])
'''
'''
for p in os.listdir(icecream):
	img = cv2.imread(icecream+p,cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(28,28))
	train_img.append([[4.0],img])
'''
'''
for p in os.listdir(test):
	img = cv2.imread(test+p,cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img,(28,28))
	tag = 30.0
	if p.split("_")[0]=="pepsi":
		tag = 10.0
	if p.split("_")[0]=="nutella":
		tag = 20.0
	test_img.append([[tag],img])

shuffle(train_img)

test_img = train_img[-100:]
train_img = train_img[:-100]

print(len(train_img))
x_train = ([i[1] for i in train_img])
y_train = ([i[0] for i in train_img])
print(y_train)

x_test = ([i[1] for i in test_img])
y_test = ([i[0] for i in test_img])


x_train =([i/255.0 for i in x_train])
x_test =([i/255.0 for i in x_test])

'''
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_train = x_train/255.0

x_test = x_test.astype('float32')
x_test = x_test/255.0


input_image_shape = (28,28,1)
embedding_size = 64



base_network = create_base_network(input_image_shape,embedding_size)
input_images = Input(shape = input_image_shape,name ="input_image")
input_labels = Input(shape = (1,),name = 'input_label') 
embeddings = base_network([input_images])
label_embeddings = concatenate([input_labels,embeddings])

model = Model(inputs = (input_labels,input_images),outputs = label_embeddings)

model.summary()

opt = Adam(lr=1e-3)
print(10)
model.compile(loss = triplet_loss_adapted_from_tf,optimizer=opt,metrics=['accuracy'])
print(20)
print(len(x_train))


dummy = np.zeros((len(x_train),embedding_size+1))

x_train = np.reshape(x_train,(len(x_train),28,28,1))
y_train = np.array(y_train)
y_test = np.array(y_test)


history = model.fit([y_train,x_train],dummy,batch_size=20,epochs = 5)

plt.plot(history.history['loss'])
plt.show()



test_emb = create_base_network(input_image_shape,embedding_size)
x_emb_before_train = test_emb.predict(np.reshape(x_test,(len(x_test),28,28,1)))
print("````````````````````````````````````````")
print(x_emb_before_train[0])
for layer_tar,layer_src in zip(test_emb.layers,model.layers[2].layers):
	weights = layer_src.get_weights()
	layer_tar.set_weights(weights)
	del weights

x_emb = test_emb.predict(np.reshape(x_test,(len(x_test),28,28,1)))
print("````````````````````````````````````````")
print(x_emb[0])
dict_emb = {}
dict_gray = {}
step = 1
test_class_labels = np.unique(np.array(y_test))

pca = PCA(n_components = 2)
decomposed_embeddings = pca.fit_transform(x_emb)
decomposed_gray = pca.fit_transform(x_emb_before_train)
y_test = np.reshape(y_test,(-1))
fig = plt.figure(figsize = (16,8))
for label in test_class_labels:
	decomposed_embeddings_class = decomposed_embeddings[y_test==label]
	decomposed_gray_class = decomposed_gray[y_test==label]
	plt.subplot(1,2,1)
	plt.scatter(decomposed_gray_class[::,1],decomposed_gray_class[::,0],label=str(label))
	plt.title('Before training')
	plt.legend()
	plt.subplot(1,2,2)
	plt.scatter(decomposed_embeddings_class[::,1],decomposed_embeddings_class[::,0],label = str(label))
	plt.xlim([-25,25])
	plt.ylim([-25,25])
	plt.title("After training")
	plt.legend()

plt.show()

print((len(y_train)))











































