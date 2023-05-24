import pandas as pd

from preprocess.preprocess import Preprocess

COVID_DATA_PATH = "data/basic/covid"


class CovidPreprocess(Preprocess):
    """
    The class responsible for preprocessing covid data.
    """

    def __init__(self, path: str = COVID_DATA_PATH, language="en_US") -> None:
        """
        Constructs all the necessary attributes for the **CovidPreprocess** class object.
        :param path: path to dataset (*covid* default)
        :param language: selected language
        """

        prepared_data = self.prepare_covid_data(path)
        super().__init__(prepared_data, path, language)

    @staticmethod
    def prepare_covid_data(path: str = COVID_DATA_PATH) -> pd.DataFrame:
        """
        Prepares covid data for later preprocessing.
        :param path: path to dataset (*covid* default)
        :return: prepared covid data
        """

        df = pd.read_csv(path + "/covid.csv")
        df.rename(columns={"headlines": "text", "outcome": "label"}, inplace=True)

        return df
