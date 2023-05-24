import emoji
import logging
import nltk
import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spylls.hunspell import Dictionary
from textblob import TextBlob
from googletrans import Translator

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
logging.basicConfig(level=logging.INFO)


class Preprocess:
    """
    The class responsible for preprocessing data
    """

    def __init__(self, df: pd.DataFrame, path: str, language="en_US") -> None:
        """
        Constructs all the necessary attributes for the **Preprocess** class object.
        :param df: dataframe containing data to preprocessing
        :param path: path to dataset
        """

        self.df = df
        self.path = path
        self.language = language

    def preprocess(self, save_path: str = None) -> None:
        """
        Method preprocesses data and saves it.
        """

        logging.info("Start of preprocessing ...")

        # logging.info("Links preprocessing ...")
        # self.links_preprocess()

        # logging.info("References preprocessing ...")
        # self.references_preprocess()

        logging.info("Hashtags preprocessing ...")
        self.preprocess_hashtags()

        logging.info("Emojis preprocessing ...")
        self.preprocess_emojis()

        # if "en" not in self.language:
        #     logging.info("Translation  ...")
        #     self.translate()

        # logging.info("Removing unicode chars ...")
        # self.remove_unicode_chars()
        #
        # logging.info("Removing punctuation ...")
        # self.remove_punctuation()

        logging.info("Checking spelling ...")
        self.check_spelling()

        logging.info("Analyzing sentiment ...")
        self.analyze_sentiment()

        # logging.info("Converting to lowercase ...")
        # self.convert_to_lowercase()
        #
        # logging.info("Removing stopwords ...")
        # self.remove_stopwords()

        # logging.info("Lemmatization ...")
        # self.lemmatize()

        logging.info("Saving preprocessed data ...")
        self.save_file(save_path)

        logging.info("The data was saved!")

    @staticmethod
    def __find_hashtags(row: pd.DataFrame) -> list:
        """
        Method to find hashtags in text.
        :param row: row from dataframe
        :return: list of words classified as hashtags
        """

        hashtags = re.findall(r"#\w+", row['text'])
        hashtags = [re.sub(r"^#", "", hashtag) for hashtag in hashtags]

        return hashtags

    @staticmethod
    def __remove_hashtags(row: pd.DataFrame) -> str:
        """
        Method to remove hashtags from text.
        :param row: row from dataframe
        :return: text without words classified as hashtags
        """

        return re.sub(r"#", "", row['text'])

    def preprocess_hashtags(self) -> None:
        """
        Method preprocesses hashtags in whole dataset, creates new column
        with list of words classified as hashtags and removes hashtags from text.
        """

        self.df['hashtags'] = self.df.apply(self.__find_hashtags, axis=1)
        self.df['text'] = self.df.apply(self.__remove_hashtags, axis=1)

    @staticmethod
    def __find_emojis(row: pd.DataFrame) -> list:
        """
        Method to find emojis in text.
        :param row: row from dataframe
        :return: list of emojis found in text
        """

        return emoji.distinct_emoji_list(row['text'])

    @staticmethod
    def __interpret_emojis(row: pd.DataFrame) -> list:
        """
        Method interprets emojis.
        :param row: row from dataframe
        :return: list of interpreted emojis
        """

        return [emoji.demojize(emoji_item, delimiters=("", "")) for emoji_item in row['emojis']]

    @staticmethod
    def __remove_emojis(row: pd.DataFrame) -> str:
        """
        Method to remove emojis from text.
        :param row: row from dataframe
        :return: text without emojis
        """

        return emoji.replace_emoji(row['text'])

    def preprocess_emojis(self) -> None:
        """
        Method preprocesses emojis in whole dataset, creates new column
        with interpreted emojis and removes emojis from text.
        """

        self.df['emojis'] = self.df.apply(self.__find_emojis, axis=1)
        self.df['emojis'] = self.df.apply(self.__interpret_emojis, axis=1)
        self.df['text'] = self.df.apply(self.__remove_emojis, axis=1)

    def translate_row(self, row: pd.DataFrame):
        translator = Translator()

        try:
            translated = translator.translate(row["text"], dest=self.language)
            return translated.text
        except:
            return None

    def translate(self) -> None:
        self.df['text'] = self.df.apply(self.translate_row, axis=1)
        self.df.dropna(subset=["text"], inplace=True)

    @staticmethod
    def __remove_unicode_chars_row(row: pd.DataFrame) -> str:
        """
        Method to remove unicode chars from text.
        :param row: row from dataframe
        :return: text without unicode chars
        """

        return row['text'].encode("ascii", "ignore").decode()

    def remove_unicode_chars(self) -> None:
        """
        Method removes unicode chars from text in whole dataset.
        """

        self.df['text'] = self.df.apply(self.__remove_unicode_chars_row, axis=1)

    @staticmethod
    def __remove_punctuation_row(row: pd.DataFrame) -> str:
        """
        Method to remove punctuation from text.
        :param row: row from dataframe
        :return: text without punctuation
        """

        return "".join([char for char in row['text'] if char not in string.punctuation])

    def remove_punctuation(self) -> None:
        """
        Method removes punctuation from text in whole dataset.
        """

        self.df['text'] = self.df.apply(self.__remove_punctuation_row, axis=1)

    def __check_spelling_row(self, row: pd.DataFrame) -> str:
        """
        Method to check spelling in text.
        :param row: row from dataframe
        :return: spell-checked text
        """

        text_tokens = word_tokenize(row['text'])
        dictionary = Dictionary.from_files(self.language)
        new_text = []

        for word in text_tokens:
            if dictionary.lookup(word):
                new_text.append(word)
            else:
                try:
                    new_text.append(next(dictionary.suggest(word)))
                except StopIteration:
                    print(f"Very misspelled word occurred: {word}")
                    new_text.append(word)

        return ' '.join(new_text)

    def check_spelling(self) -> None:
        """
        Method checks spelling in text in whole dataset.
        """

        self.df['text'] = self.df.apply(self.__check_spelling_row, axis=1)

    @staticmethod
    def __define_sentiment(row: pd.DataFrame) -> str:
        """
        Method to define sentiment from numeric to textual form
        :param row: row from dataframe
        :return: spell-checked text
        """

        return 'negative' if row['polarity'] < 0 else 'positive' if row['polarity'] > 0 else 'neutral'

    def analyze_sentiment(self) -> None:
        """
        Method analyzes sentiments and creates new columns:
            - polarity - numerical value of text polarity \n
            - subjectivity - numerical value of text subjectivity \n
            - sentiment - textual form of text polarity
        """

        sentiment_items = [TextBlob(text) for text in self.df['text'].tolist()]
        self.df['polarity'] = [text.sentiment.polarity for text in sentiment_items]
        self.df['subjectivity'] = [text.sentiment.subjectivity for text in sentiment_items]
        self.df['sentiment'] = self.df.apply(self.__define_sentiment, axis=1)

    def convert_to_lowercase(self) -> None:
        """
        Method converts text to lowercase in whole dataset.
        """

        self.df['text'] = self.df['text'].apply(lambda text: text.lower())

    @staticmethod
    def __remove_stopwords_row(row: pd.DataFrame) -> str:
        """
        Method to remove stopwords from text.
        :param row: row from dataframe
        :return: text without stopwords
        """

        text_tokens = word_tokenize(row['text'])
        tokens_without_sw = [
            word for word in text_tokens if word not in stopwords.words()
        ]

        return ' '.join(tokens_without_sw)

    def remove_stopwords(self) -> None:
        """
        Method removes stopwords from text in whole dataset.
        """

        self.df['text'] = self.df.apply(self.__remove_stopwords_row, axis=1)

    @staticmethod
    def __lemmatize_row(row: pd.DataFrame) -> str:
        """
        Method to lemmatize text.
        :param row: row from dataframe
        :return: lemmatized text
        """

        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(row['text'])
        new_text = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(new_text)

    def lemmatize(self) -> None:
        """
        Method lemmatizes text in whole dataset.
        """

        self.df['text'] = self.df.apply(self.__lemmatize_row, axis=1)

    @staticmethod
    def __generate_path_to_save(path: str) -> str:
        """
        Method to generate path to text.
        :param path: path of dataset
        :return: generated save path
        """

        split_path = path.split('/')
        file_name = split_path[-1]
        save_path = f"data/preprocessed/{file_name}"

        return save_path

    def save_file(self, save_path: str = None) -> None:
        """
        Method saves preprocessed data to .json file. If path is not determined,
        data is saved under automatically generated path.
        """

        if not save_path:
            save_path = self.__generate_path_to_save(self.path)

        self.df.to_json(save_path, orient="records", lines=True)
