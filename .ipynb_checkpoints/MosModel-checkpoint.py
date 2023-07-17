import os

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import model_from_json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from jupyterthemes import jtplot
jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False) 

Directory = 'C:\\Users\\Sanjana Rattan\\Mosquito Detection\\Mosquito-Detection-App\\train'
print(os.listdir(Directory))
image_generator=ImageDataGenerator(rescale = 1./255, validation_split=0.2)
train_generator=image_generator.flow_from_directory(batch_size=40,directory=Directory,shuffle=True,
                                                    target_size=(256,256),class_mode='categorical',subset='training')

validation_generator=image_generator.flow_from_directory(batch_size=40,directory=Directory,shuffle=True 
                                                         ,target_size=(256,256),class_mode='categorical',subset='training')

train_images,train_labels=next(train_generator)
print(train_labels)
label_names={0:'Aedes' ,1:'Anopheles',2:'Culex',3:"Butterfly"}
print(train_images.shape)
print(train_labels.shape)

L=4
W=4

fig, axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()
for i in np.arange(0, L*W):
    axes[i].imshow(train_images[i])
    axes[i].set_title(label_names[np.argmax(train_labels[i])])
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)

basemodel=ResNet50(weights='imagenet',include_top=False,input_tensor=Input(shape=(256,256,3)))
basemodel.summary()

for layer in basemodel.layers[:-10]:
    layers.trainable=False

headmodel=basemodel.output
headmodel=AveragePooling2D(pool_size=(4,4))(headmodel)
headmodel=Flatten(name='flatten')(headmodel)
headmodel=Dense(256,activation='relu')(headmodel)
headmodel=Dropout(0.3)(headmodel)
headmodel=Dense(128,activation='relu')(headmodel)
headmodel=Dropout(0.2)(headmodel)

headmodel=Dense(4, activation='softmax')(headmodel)

model=Model(inputs=basemodel.input,outputs=headmodel)
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=1e-4),
    metrics=["accuracy"]
)

earlystopping = EarlyStopping(monitor='val_loss', 
                              mode='min', verbose=1, patience=20)

checkpointer = ModelCheckpoint(filepath="Mosmodel.hdf5", 
                               verbose=1, save_best_only=True)

train_generator = image_generator.flow_from_directory(batch_size = 4, directory= Directory,
                                                     shuffle= True, target_size=(256,256), 
                                                      class_mode= 'categorical', subset="training")
val_generator = image_generator.flow_from_directory(batch_size = 4, directory= Directory, 
                                                    shuffle= True, target_size=(256,256),
                                                   class_mode= 'categorical', subset="validation")

history = model.fit(train_generator, steps_per_epoch= train_generator.n // 4, 
                    epochs = 1, validation_data= val_generator, validation_steps= 
                    val_generator.n // 4, callbacks=[checkpointer, earlystopping])




# load json and create model

json_file = open('trained_mos_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('mos_trained_model.h5')
print("Loaded model from disk")

model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6),
              loss='categorical_crossentropy', metrics=['accuracy'])


test_directory = 'C:\\Users\\Sanjana Rattan\\Mosquito Detection\\Mosquito-Detection-App\\test'

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_directory(batch_size=40, directory=test_directory, shuffle=True,
                                              target_size=(256, 256), class_mode='categorical')

evaluation = model.evaluate_generator(test_generator, steps=test_generator.n // 4, verbose=1)

print('Accuracy Test: {}'.format(evaluation[1]))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2
prediction = []
original = []
image = []

for i in range(len(os.listdir(test_directory))):
    for item in os.listdir(os.path.join(test_directory, str(i))):
        img = cv2.imread(os.path.join(test_directory, str(i), item))
        if img is not None:
            img = cv2.resize(img, (256, 256))
            image.append(img)
            img = img / 255
            img = img.reshape(-1, 256, 256, 3)
            predict = model.predict(img)
            predict = np.argmax(predict)
            prediction.append(predict)
            original.append(i)


len(original)

score = accuracy_score(original,prediction)
print("Test Accuracy : {}".format(score))

L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(image[i])
    axes[i].set_title('Guess={}\nTrue={}'.format(str(label_names[prediction[i]]), str(label_names[original[i]])))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1.2) 

print(classification_report(np.asarray(original), np.asarray(prediction)))

cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')






