import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics


class ModelEvaluator:

    @staticmethod
    def evaluate(model, dataset, set_type):
        y_true = dataset["label"].tolist()
        y_pred = ModelEvaluator.predict(model, dataset)

        scores = ModelEvaluator.calculate_metrics(y_true, y_pred)
        ModelEvaluator.plot_confusion_matrix(y_true, y_pred, set_type)

        return scores, y_true, y_pred

    @staticmethod
    def predict(model, dataset):
        y_pred = []

        for index, row in dataset.iterrows():
            text = row['text']
            y_pred.append(model.classify_text(text, printing=False)[0])

        return y_pred

    @staticmethod
    def get_false_predicted_elements(dataset, y_true, y_pred):
        different_indexes = []

        for index, (elem1, elem2) in enumerate(zip(y_true, y_pred)):
            if elem1 != elem2:
                different_indexes.append(index)

        return dataset.iloc[different_indexes]

    @staticmethod
    def get_correctly_predicted_elements(dataset, y_true, y_pred):
        same_indexes = []

        for index, (elem1, elem2) in enumerate(zip(y_true, y_pred)):
            if elem1 == elem2:
                same_indexes.append(index)

        return dataset.iloc[~dataset.index.isin(same_indexes)]

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        accuracy = round(metrics.accuracy_score(y_true, y_pred), 5)
        precision = round(metrics.precision_score(y_true, y_pred), 5)
        recall = round(metrics.recall_score(y_true, y_pred), 5)
        f1_score = round(metrics.f1_score(y_true, y_pred), 5)

        print("Accuracy:  ", accuracy)
        print("Precision: ", precision)
        print("Recall:    ", recall)
        print("F1 score:  ", f1_score)
        print("\n")

        return [accuracy, precision, recall, f1_score]

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, set_type="test"):
        cf_matrix = metrics.confusion_matrix(y_true, y_pred)
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(cf_matrix, center=True, annot=labels, fmt="", cmap='viridis')
        plt.title(f"Confusion matrix for {set_type} set")
        plt.show()
