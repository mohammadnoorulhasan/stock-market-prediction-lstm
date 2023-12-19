import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("src")

import pandas as pd
import numpy as np
from config import logger

class CreateDataset:
    """
    A class for loading financial data, creating datasets with features and labels, and saving the dataset to a CSV file.
    
    Parameters:
        - dataPath (str): The file path of the financial data.
        - csvPath (str): The file path to save the generated dataset.
        - columns (list): List of columns to extract from the financial data.
        - timestep (int, optional): The time step for creating sequences in the dataset. Default is 100.
    """

    def __init__(self, dataPath, csvPath, columns, timestep=100):
        self.dataPath = dataPath
        self.csvPath = csvPath
        self.columns = columns
        self.timestep = timestep
        self.loadData()
        self.createDataset(timestep)
        self.saveCsv()
        

    def loadData(self):
        """
        Load financial data from the specified file path and extract selected columns.
        """
        featuresLen = len(self.columns)
        self.stockMarketData = pd.read_csv(self.dataPath)
        self.stockMarketData = self.stockMarketData.reset_index()[self.columns]
        

    def createDataset(self, timeStep=10):
        """
        Create a dataset with features and labels from the loaded financial data.

        Parameters:
            - timeStep (int, optional): The time step for creating sequences in the dataset. Default is 10.
        """
        self.features, self.labels = [],[]

        for i in range(len(self.stockMarketData) - timeStep - 1):
            feature = self.stockMarketData.iloc[i:(i + timeStep), :].values
            self.features.append(feature)
            label = self.stockMarketData.iloc[i + timeStep].values
            self.labels.append(label)

    def saveCsv(self):
        """
        Save the created dataset with features and labels to a CSV file.
        """
        data = {
            "features": self.features,
            "labels": self.labels
        }

        dataset = pd.DataFrame(data=data)
        
        dataset.to_csv(self.csvPath, index=False, header=False)
        logger.info(f"Dataset has been created and save into file {self.csvPath}")

if __name__ == "__main__":
    dataPath = "../data/stock-TCS.ns_yfinance_2012-01-01-2023-12-03.csv"
    csvPath = "../Dataset/stock-TCS.ns_yfinance_2012-01-01-2023-12-03.csv"
    columns = ["Open", "High", "Low", "Close"]
    
    createDatasetInstance = CreateDataset(dataPath, csvPath, columns)

    # stockMarketData = createDatasetInstance.loadData()
    # features, labels = createDatasetInstance.createDataset(stockMarketData)
    # createDatasetInstance.saveCsv(features, labels)
