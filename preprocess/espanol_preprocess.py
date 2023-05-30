import pandas as pd

from preprocess.preprocess import Preprocess

ESPANOL_DATA_PATH = "data/basic/spanish/espanol.csv"


class EspanolPreprocess(Preprocess):
    """
    The class responsible for preprocessing world news data.
    """

    def __init__(self, path: str = ESPANOL_DATA_PATH, language: str = "es") -> None:
        """
        Constructs all the necessary attributes for the **WorldNewsPreprocess** class object.
        :param path: path to dataset (*world_news* default)
        """

        prepared_data = self.prepare_world_news_data(path)
        super().__init__(prepared_data, path, language)

    @staticmethod
    def prepare_world_news_data(path: str = ESPANOL_DATA_PATH) -> pd.DataFrame:
        """
        Prepares world news data for later preprocessing.
        :param path: path to dataset (*world_news* default)
        :return: prepared world news data
        """

        df = pd.read_csv(path)
        df.text = df.title + df.text
        df.drop(columns=['title'], inplace=True)
        df.dropna(inplace=True)

        return df
