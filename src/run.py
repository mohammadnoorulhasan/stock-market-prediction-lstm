import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("src")

from config import *

from StockDataDownloader import StockDataDownloader
from CreateDataset import CreateDataset
from trainModel import StockMarketLSTM

def main():
    stockDataDownloaderObj = StockDataDownloader(
        symbol, 
        startDate, 
        endDate, 
        downloadModule = downloadModule
    )
    stockMarketLSTMObj = StockMarketLSTM(
        dataFilePath,
        scalerPath,
        modelPath,
        nEpoches = nEpoches,
        patience = patience
    )
    stockMarketLSTMObj.main()
if __name__ == "__main__":
    main()