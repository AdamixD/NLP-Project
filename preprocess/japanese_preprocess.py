import pandas as pd

from preprocess.preprocess import Preprocess
JAPANESE_DATA_PATH = "data/basic/japanese"


class JapanesePreprocess(Preprocess):
    """
    The class responsible for preprocessing japanese data.
    """

    def __init__(self, path: str = JAPANESE_DATA_PATH, language="ja-JP") -> None:
        """
        Constructs all the necessary attributes for the **Japanese** class object.
        :param path: path to dataset (*japanese* default)
        :param language: selected language
        """

        prepared_data = self.prepare_japanese_data(path)
        super().__init__(prepared_data, path, language)

    @staticmethod
    def prepare_japanese_data(path: str = JAPANESE_DATA_PATH) -> pd.DataFrame:
        """
        Prepares japanese data for later preprocessing.
        :param path: path to dataset (*japanese* default)
        :return: prepared japanese data
        """

        df = pd.read_csv(path + "/japanese.csv")
        df = df[['text', 'label']]
        return df
