import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("src")

from config import *
import os
from time import time
import yfinance as yf
from nsepy import get_history
from datetime import timedelta
import datetime

class StockDataDownloader:
    """
    A class to download stock market data using yfinance or nsepython.

    Parameters:
    - symbol: Stock symbol (e.g., 'TCS', 'AAPL')
    - startDate: Start date in 'YYYY-MM-DD' format
    - endDate: End date in 'YYYY-MM-DD' format (default is yesterday)
    - saveCsvFlag: Flag to save the downloaded data to a CSV file (default is True)
    - downloadModule: Module to use for data download ('yfinance' or 'nsepython')
    """

    def __init__(self, symbol, startDate, endDate=None, saveCsvFlag=True, downloadModule='yfinance', interval = "1d"):
        self.symbol = symbol
        self.startDate = startDate
        self.dataFolder = '../Data'
        self.saveCsvFlag = saveCsvFlag
        self.downloadModule = downloadModule
        self.interval = interval

        if endDate is None:
            # Get the current date and time
            currentDatetime = datetime.datetime.now()

            # Calculate the end date of yesterday
            self.endDate = currentDatetime - timedelta(days=1)
            self.endDate = self.endDate.strftime('%Y-%m-%d')
        else:
            self.endDate = endDate
        
        self.downloadData()

    def downloadData(self):
        """Download stock market data and save it to a CSV file."""
        # Create CSV filename
        csvFilename = f"stock-{self.symbol}_{self.downloadModule}_{self.startDate}-{self.endDate}.csv"

        # Create file path
        filePath = dataFilePath
        secho(f"Downloading data from {downloadModule}")
        secho(f"Symbol : {self.symbol}")
        secho(f"Start Date : {self.startDate}")
        secho(f"End Date : {self.endDate}")
        
        # Download stock market data
        if self.downloadModule == 'nsepython':
            # Use nsepython for data download
            self.startDate = datetime.datetime.strptime(self.startDate, '%Y-%m-%d').date()
            self.endDate = datetime.datetime.strptime(self.endDate, '%Y-%m-%d').date()
            
            stockData = get_history(
                symbol=self.symbol,
                start=self.startDate,
                end=self.endDate
            )
        elif self.downloadModule == 'yfinance':
            # Use yfinance for data download
            stockData = yf.download(
                f"{self.symbol}", 
                start=self.startDate, 
                end=self.endDate,
                interval = self.interval
            )
        else:
            logger.warning("Invalid download module specified. Please use 'yfinance' or 'nsepython'.")
            return

        if self.saveCsvFlag:
            # Create Data folder if it doesn't exist
            if not os.path.exists(self.dataFolder):
                os.makedirs(self.dataFolder)
                logging.info(f"Directory Created: {self.dataFolder}")

            # Remove existing file if it exists
            if os.path.exists(filePath):
                os.remove(filePath)
                logger.warning(f"File Deleted: {filePath}")

            # Save data to CSV file
            stockData.to_csv(filePath)
            logger.info(f"File Saved: {filePath}")


# Example Usage:
# Instantiate the class with the desired parameters
if __name__ == "__main__":
    # Choose download module ('yfinance' or 'nsepython')
    # downloadModule = 'yfinance'  # Change to 'yfinance' if needed
    # downloadModule = 'nsepython'  
    
    downloader = StockDataDownloader(
        symbol=symbol, 
        startDate=startDate, 
        endDate= endDate, 
        downloadModule = downloadModule,
        interval=interval
    )

    # Download and save the data
    # downloader.downloadData()
