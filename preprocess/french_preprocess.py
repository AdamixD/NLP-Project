import pandas as pd

from preprocess.preprocess import Preprocess
FRENCH_DATA_PATH = "data/basic/french"


class FrenchPreprocess(Preprocess):
    """
    The class responsible for preprocessing french data.
    """

    def __init__(self, path: str = FRENCH_DATA_PATH, language="fr") -> None:
        """
        Constructs all the necessary attributes for the **FrenchPreprocess** class object.
        :param path: path to dataset (*french* default)
        :param language: selected language
        """

        prepared_data = self.prepare_french_data(path)
        super().__init__(prepared_data, path, language)

    @staticmethod
    def prepare_french_data(path: str = FRENCH_DATA_PATH) -> pd.DataFrame:
        """
        Prepares french data for later preprocessing.
        :param path: path to dataset (*french* default)
        :return: prepared french data
        """

        df = pd.read_csv(path + "/french.csv", lines=True)
        df = df[['text', 'label']]
        return df
