import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

class SequentialImageDataGenerator:
    def __init__(self):
        (train_images, train_labels), _ = mnist.load_data()
        train_images = train_images.astype('float32') / 255.0
        self.data = pd.DataFrame({"labels": train_labels, 'imgs': train_images.tolist()})
        self.new_df = pd.DataFrame(columns=['input1', 'input2', 'input3', 'input4', 'output'])

    def filter_by_label(self, label):
        return self.data[self.data['labels'] == label]['imgs']

    def generate_formula_data(self, n):
        return n, n+1, n+2, n+3, n+4

    def generate_data(self):
        for i in range(6):  # dari 0-4 ke 4-8
            labels = list(self.generate_formula_data(i))
            min_count = min(self.data[self.data['labels'].isin(labels)]['labels'].value_counts())
            for j in range(min_count):
                row = [self.filter_by_label(label).iloc[j] for label in labels]
                self.new_df.loc[len(self.new_df)] = row
        return self.new_df

    def re_scale_data(self):
        df = self.generate_data()
        X = np.stack([np.array(df[col].tolist()) for col in ['input1', 'input2', 'input3', 'input4']], axis=-1)
        y = np.stack([np.array(df['output'].tolist())], axis=-1)
        X = X.reshape(-1, 4, 28*28)
        y = y.reshape(-1, 28, 28)
        return train_test_split(X, y, test_size=0.2, random_state=42)