import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

data_dir = '.'

train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))

train_labels['id'] = train_labels['id'].apply(lambda x: f"{x}.tif")

print("First few rows of the dataset:")
print(train_labels.head())

print("\nSize and Dimensions of the dataset:")
print(f"Number of rows: {train_labels.shape[0]}")
print(f"Number of columns: {train_labels.shape[1]}")

print("\nStructure of the dataset:")
print(train_labels.info())

print("\nSummary statistics of the dataset:")
print(train_labels.describe())

print("\nMissing values in the dataset:")
print(train_labels.isnull().sum())

print("\nDistribution of the labels:")
label_counts = train_labels['label'].value_counts()
print(label_counts)

plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_labels)
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
train_labels['label'].hist()
plt.title('Histogram of Labels')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()

def display_samples(data_dir, df, num_samples=10):
    fig, axs = plt.subplots(1, num_samples, figsize=(20, 20))
    for i, ax in enumerate(axs):
        sample = df.sample(1).iloc[0]
        img_path = os.path.join(data_dir, 'train', sample['id'])
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(f"Label: {sample['label']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print("Displaying 10 random images from the dataset:")
display_samples(data_dir, train_labels, num_samples=10)
