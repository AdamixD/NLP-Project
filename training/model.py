from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import pyarrow as pa
from keras import backend as K
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

class Model:
    def __init__(self, model_path, useGPU=False) -> None:
        if useGPU:
            # TODO: create training with CUDA
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path).to("cuda")
        else:
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    @staticmethod
    def create_dataset(data):
        schema = pa.schema([
            ("text", pa.string()),
            ("label", pa.int64()),
            ("hashtags", pa.list_(pa.string())),
            ("emojis", pa.list_(pa.string())),
            ("polarity", pa.float64()),
            ("subjectivity", pa.float64()),
            ("sentiment", pa.string()),
        ])

        arrow_table = pa.Table.from_batches([], schema=schema)
        dataset = Dataset(arrow_table=arrow_table)

        return dataset.from_pandas(data)

    def tokenize_dataset(self, dataset):
        return self.tokenizer(dataset["text"][:511])

    def prepare_dataset(self, dataset):
        dataset = dataset.map(self.tokenize_dataset)
        tf_dataset = self.model.prepare_tf_dataset(
            dataset, batch_size=16, shuffle=True, tokenizer=self.tokenizer
        )

        return tf_dataset

    def prepare_train_test_data(self, dataset):
        train_data, test_data = train_test_split(dataset, train_size=0.8, random_state=42)
        train_dataset = self.create_dataset(train_data)
        test_dataset = self.create_dataset(test_data)

        tf_train = self.prepare_dataset(train_dataset)
        tf_test = self.prepare_dataset(test_dataset)

        return tf_train, tf_test

    # def compile(self):
    #     loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #     metrics = keras.metrics.SparseCategoricalAccuracy('accuracy')
    #     self.model.compile(optimizer=Adam(3e-5), loss=loss, metrics=[metrics, f1_score])

    # def fit(self, train_data, validation_data, epochs=3):
    #     return self.model.fit(train_data, epochs=epochs, validation_data=validation_data)

    @staticmethod
    def get_true_positive(y_true, y_pred):
        return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    def recall_m(self, y_true, y_pred):
        true_positives = self.get_true_positive(y_true, y_pred)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = self.get_true_positive(y_true, y_pred)
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def compile(self):
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=Adam(3e-5), loss=loss, metrics=[metrics, self.f1_m])

    def fit(self, train_data, validation_data, epochs=3):
        return self.model.fit(train_data, epochs=epochs, validation_data=validation_data, batch_size=16)

    def evaluate(self, dataset):
        return self.model.evaluate(dataset)

    @staticmethod
    def load_saved_model(model_path, useGPU=False):
        return Model(model_path=model_path, useGPU=useGPU)

    def classify_text(self, text, printing=True):
        encoded_text = self.tokenizer.encode(text, truncation=True, padding=True, return_tensors='tf')
        prediction = self.model(encoded_text).logits.numpy()[0]
        probs = np.exp(prediction) / np.sum(np.exp(prediction))
        predicted_class = np.argmax(probs)

        if printing:
            print("Predicted class:", predicted_class)
            print("Probability distribution:", probs)

        return [predicted_class, probs]

    def save_model(self, save_path):
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)

