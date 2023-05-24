import logging
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd

from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
logging.basicConfig(level=logging.INFO)


class DataAnalyzer:

    @staticmethod
    def plot_label_distribution(dataset):
        dataset['label'].value_counts().plot(kind='bar')
        plt.title('Data distribution by label value')

    @staticmethod
    def plot_polarity_distribution(dataset, label_type=None):
        selected_data = DataAnalyzer.select_data(dataset, label_type)

        if label_type is not None:
            title_text = f'Data polarity distribution (label {label_type})'
        else:
            selected_data = selected_data.groupby('label')
            title_text = 'Data polarity distribution by label value'

        selected_data['polarity'].hist(alpha=0.8, bins=15, legend=True)
        plt.title(title_text)

    @staticmethod
    def plot_subjectivity_distribution(dataset, label_type=None):
        selected_data = DataAnalyzer.select_data(dataset, label_type)

        if label_type is not None:
            title_text = f'Data subjectivity distribution (label {label_type})'
        else:
            selected_data = selected_data.groupby('label')
            title_text = 'Data subjectivity distribution by label value'

        selected_data['subjectivity'].hist(alpha=0.8, bins=15, legend=True)
        plt.title(title_text)

    @staticmethod
    def plot_most_popular_words_map(
            most_popular_words,
            most_popular_words_num=100,
            label_type=None
    ):
        words = most_popular_words[0:most_popular_words_num]

        if label_type is not None:
            title_text = f'{most_popular_words_num} Most Popular Words (label {label_type})'
        else:
            title_text = f'{most_popular_words_num} Most Popular Words'

        wordcloud = WordCloud(
            width=1400,
            height=700,
            background_color='white'
        ).generate(str(words))

        plt.figure(figsize=(30, 10), facecolor='white')
        plt.imshow(wordcloud, interpolation="bilinear", cmap='viridis')
        plt.axis('off')
        plt.title(title_text, fontsize=38)

    @staticmethod
    def plot_most_popular_words_histogram(
            most_popular_words,
            most_popular_words_num=25,
            label_type=None
    ):

        words = most_popular_words[0:most_popular_words_num]

        x, y = zip(*words)

        if label_type is not None:
            title_text = f'Frequency of {most_popular_words_num} Most Popular Words (label {label_type})'
        else:
            title_text = f'Frequency of {most_popular_words_num} Most Popular Words'

        plt.figure(figsize=(30, 15))
        plt.margins(0.05)
        plt.bar(x, y)
        plt.xlabel('Words', fontsize=50)
        plt.ylabel('Frequency of Words', fontsize=50)
        plt.yticks(fontsize=40)
        plt.xticks(rotation=60, fontsize=40)
        plt.title(title_text, fontsize=64)

    @staticmethod
    def select_data(dataset, label_type=None):
        if label_type == 0:
            return dataset.groupby(['label']).get_group(0)
        elif label_type == 1:
            return dataset.groupby(['label']).get_group(1)
        return dataset

    @staticmethod
    def get_most_popular_words(
            dataset,
            language,
            most_popular_words_num=100,
            label_type=None
    ):
        selected_data = DataAnalyzer.select_data(dataset, label_type)

        if language == "japanese":
            selected_data = DataAnalyzer.remove_unicode_chars(selected_data)
            selected_data = DataAnalyzer.remove_stopwords(selected_data)

            dictionary = selected_data['text'].str.split("[^\w+]").explode().tolist()

        else:
            text = selected_data['text'].str.lower().tolist()
            cleaned_text = []
            stopwords_set = set(stopwords.words(language))

            for sentence in text:
                words = re.findall(r'\b(?!(?:A|the|e|it)\b)\w+\b', sentence)
                cleaned_text.append([word for word in words if word.lower() not in stopwords_set and not word.isdigit()])

            dictionary = [word for sublist in cleaned_text for word in sublist]

        most_popular_words = FreqDist(dictionary).most_common(most_popular_words_num)

        return most_popular_words

    @staticmethod
    def __remove_unicode_chars_row(row: pd.DataFrame) -> str:
        return row['text'].encode("ascii", "ignore").decode()

    @staticmethod
    def remove_unicode_chars(dataset) -> pd.DataFrame:
        dataset['text'] = dataset.apply(DataAnalyzer.__remove_unicode_chars_row, axis=1)

        return dataset

    @staticmethod
    def __remove_stopwords_row(row: pd.DataFrame) -> str:
        text_tokens = word_tokenize(row['text'])
        tokens_without_sw = [
            word for word in text_tokens if word not in stopwords.words()
        ]

        return ' '.join(tokens_without_sw)

    @staticmethod
    def remove_stopwords(dataset) -> pd.DataFrame:
        dataset['text'] = dataset.apply(DataAnalyzer.__remove_stopwords_row, axis=1)

        return dataset
