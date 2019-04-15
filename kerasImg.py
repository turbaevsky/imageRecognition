#Подключаем библиотеки
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from tqdm import tqdm
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from skimage import io


r = 4 # classes range
class_names = ['Nobody','Below','Upstairs','Something'] if r==4 else ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x, y = 128*1, 96*1



def preprocess(path):
    X = []
    for p in path:
        img = io.imread(p, as_grey = True)
        img = cv2.resize(img, (x, y))
        X.append(img)
    X = np.array(X, dtype=np.float64)
    X = X/255.0
    return X


def load():
  #fashion_mnist = keras.datasets.fashion_mnist
  #(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  train_images, train_labels = [],[]
  with open ('/home/tur/imageCat.csv') as f:
      lis = [line.split(',') for line in f]
      for f, a in lis:
          #f = '/home/tur/tmp/pics/'+f[-19:]
          im = io.imread(f, as_grey = True)
          im = cv2.resize(im, (x, y))
          train_images.append(im)
          train_labels.append(int(a))

  #print(len(X),len(Y))

  train_images = np.array(train_images, dtype=np.float64)

  test_images = train_images[-20:]     
  test_labels = train_labels[-20:]   
  train_images = train_images[:-20]
  train_labels = train_labels[:-20]

  train_images = train_images / 255.0
  test_images = test_images / 255.0

  #print(train_labels)

  #plt.figure()
  #plt.imshow(train_images[0])
  #plt.colorbar()
  #plt.grid(False)
  #plt.show()

  #plt.figure(figsize=(10,10))

  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(train_images[i], cmap=plt.cm.binary)
      plt.xlabel(class_names[train_labels[i]])
  plt.show()

  return [train_images, train_labels, test_images, test_labels]


def getModel(train_images, train_labels, test_images, test_labels):
  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(y, x)),
      keras.layers.Dense(512, activation=tf.nn.relu),
      keras.layers.Dense(r, activation=tf.nn.softmax)
  ])

  model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_images, train_labels, epochs=3, steps_per_epoch=200)

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('Test accuracy:', test_acc)

  #predictions = model.predict(test_images)

  #print(np.argmax(predictions[0]))

  model.save('imageRec.h5')

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(r), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#i = 0
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions, test_labels, test_images)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions,  test_labels)
#plt.show()

def plotPredictions(predictions, test_images, test_labels):
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

# Возьмём изображение из тестового набора данных
#img = test_images[0]

#Добавим изображение в пакет, где он является единственным членом
#img = (np.expand_dims (img, 0))

#predictions_single = model.predict(img)
#print(predictions_single)

#np.argmax(predictions_single[0])

path = '/home/tur/rasp/camera/20190219/images/'
names = os.listdir(path)
#print(p)

p = []
for fn in names:
    p.append(path+fn)

getModel(load()[0],load()[1],load()[2],load()[3])

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('imageRec.h5')
plotPredictions(model.predict(load()[2]),load()[2],load()[3])

#for a, fn in zip(model.predict(preprocess(p)),p):
#    print(fn, class_names[np.argmax(a)])