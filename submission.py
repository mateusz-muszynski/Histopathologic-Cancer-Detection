import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = '.'

test_labels = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))  # Assuming the file is named 'sample_submission.csv'

test_labels['id'] = test_labels['id'].apply(lambda x: f"{x}.tif")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_labels,
    directory=os.path.join(data_dir, 'test'),
    x_col='id',
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(96, 96)
)

model = load_model('trained_model.h5')

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)

filenames = test_generator.filenames

filenames = [os.path.splitext(f)[0] for f in filenames]


submission = pd.DataFrame({'id': filenames, 'label': predicted_classes.flatten()})


assert submission.shape[0] == 57458, f"Submission file has {submission.shape[0]} rows, expected 57458 rows."


submission.to_csv('submission.csv', index=False)
