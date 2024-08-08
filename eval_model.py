import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = '.'

train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))

train_labels['id'] = train_labels['id'].apply(lambda x: f"{x}.tif")

train_labels['label'] = train_labels['label'].astype(str)

sample_size = 5000
train_labels_subset = train_labels.sample(n=sample_size, random_state=42)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=train_labels_subset,
    directory=os.path.join(data_dir, 'train'),
    x_col='id',
    y_col='label',
    subset='validation',
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode='binary',
    target_size=(96, 96)
)

model = load_model('trained_model.h5')

results = model.evaluate(validation_generator)
print(f"Validation Loss: {results[0]}")
print(f"Validation Accuracy: {results[1]}")

predictions = model.predict(validation_generator)
predicted_classes = (predictions > 0.5).astype("int32")

true_classes = validation_generator.classes
correctly_classified = (predicted_classes.flatten() == true_classes).sum()
incorrectly_classified = (predicted_classes.flatten() != true_classes).sum()

fig, ax = plt.subplots()
bar_labels = ['Correctly Classified', 'Incorrectly Classified']
bar_values = [correctly_classified, incorrectly_classified]
ax.bar(bar_labels, bar_values, color=['green', 'red'])
ax.set_ylabel('Number of Images')
ax.set_title('Classification Results')
plt.show()

from sklearn.metrics import classification_report
class_labels = list(validation_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
