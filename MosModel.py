import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths
mosquito_directory = 'C:\\Users\\Sanjana Rattan\\Mosquito Database\\train'
print(os.listdir(mosquito_directory))

# Image preprocessing and data generators
image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = image_generator.flow_from_directory(batch_size=40, directory=mosquito_directory,
                                                      shuffle=True, target_size=(256, 256),
                                                      class_mode='categorical', subset='training')
validation_generator = image_generator.flow_from_directory(batch_size=40, directory=mosquito_directory,
                                                           shuffle=True, target_size=(256, 256),
                                                           class_mode='categorical', subset='validation')

train_images, train_labels = next(train_generator)
label_names = {0: 'Aedes Aegypti', 1: 'Anopheles Stephens'}

# Model architecture
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

for layer in base_model.layers[:-10]:
    layer.trainable = False

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(256, activation='relu')(head_model)
head_model = Dropout(0.3)(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.2)(head_model)
head_model = Dense(2, activation='softmax')(head_model)  # 2 classes: Aedes Aegypti, Anopheles Stephens

model = Model(inputs=base_model.input, outputs=head_model)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

# Checkpoint callback
checkpoint_path = 'weights.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
                                                save_best_only=True, mode='min')

# Model training
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // 40, epochs=20,
                    validation_data=validation_generator, validation_steps=validation_generator.samples // 40,
                    callbacks=[checkpoint])

# Save the entire model
model.save('mosquito_model.h5')







