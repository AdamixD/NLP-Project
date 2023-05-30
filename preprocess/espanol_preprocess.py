import pandas as pd

from preprocess.preprocess import Preprocess


class EspanolPreprocess(Preprocess):
    """
    The class responsible for preprocessing espanol news data.
    """

    def __init__(self, path: str, language: str = "es") -> None:
        """
        Constructs all the necessary attributes for the **EspanolPreprocess** class object.
        :param path: path to dataset
        """

        prepared_data = self.prepare_japanese_data(path)
        super().__init__(prepared_data, path, language)

    @staticmethod
    def prepare_japanese_data(path: str) -> pd.DataFrame:
        """
        Prepares espanol data for later preprocessing.
        :param path: path to dataset
        :return: prepared espanol news data
        """

        df = pd.read_csv(path)
        df.text = df.title + df.text
        df.drop(columns=['title'], inplace=True)
        df.dropna(inplace=True)

        return df
