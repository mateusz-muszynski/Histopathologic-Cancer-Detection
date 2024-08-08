import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('mixed_float16')

data_dir = '.'

train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))

train_labels['id'] = train_labels['id'].apply(lambda x: f"{x}.tif")

train_labels['label'] = train_labels['label'].astype(str)

sample_size = 5000
train_labels_subset = train_labels.sample(n=sample_size, random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_labels_subset,
    directory=os.path.join(data_dir, 'train'),
    x_col='id',
    y_col='label',
    subset='training',
    batch_size=64,  
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(96, 96)
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_labels_subset,
    directory=os.path.join(data_dir, 'train'),
    x_col='id',
    y_col='label',
    subset='validation',
    batch_size=64,  
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(96, 96)
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,  # Reduced number of epochs
    callbacks=[early_stopping]
)

model.save('trained_model.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
