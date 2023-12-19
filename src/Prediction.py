import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("src")

from tensorflow.keras.models import load_model
from datetime import timedelta, datetime
from config import *
import yfinance as yf
import joblib
import numpy as np

class Prediction:
    def __init__(self, symbol, interval, modelFile, scalerPath,  downloadModule = 'yfinance', predictionDate = None):
        self.symbol = symbol
        self.downloadModule = downloadModule
        self.interval = interval
        if predictionDate is None:
            currentDatetime = datetime.now()
            # Calculate the end date of yesterday
            # endDate = currentDatetime - timedelta(days=1)
            self.predictionDate = currentDatetime.strftime('%Y-%m-%d')
        else:   
            self.predictionDate = predictionDate
        self.scaler = joblib.load(scalerPath)
        self.model = load_model(modelFile)
        self.predictionPoint()
        

    def predictionPoint(self):
        currentDatetime = datetime.now()
        # Calculate the end date of yesterday
        startDate = currentDatetime - timedelta(days = timeStep * 2)
        self.startDate = startDate.strftime('%Y-%m-%d')
        endate = currentDatetime
        self.endDate = endate.strftime('%Y-%m-%d')
        stockData = yf.download(
                f"{self.symbol}", 
                start=self.startDate, 
                end=self.endDate,
                interval = self.interval
            )
        print(columns)
        stockData = stockData[columns]
        stockData = stockData[-timeStep:]
        stockData = self.scaler.transform(stockData)
        features, labels = [],[]

        for i in range(len(stockData)):
            feature = stockData[i, :]
            # logger.info(f"Feature : {feature}")
            features.append(feature)
            # labels.append(stockData[i + timeStep, :])
        features = np.array(features)
        logger.info(f"length of features : {features.shape}")    
        logger.info(f"Features : {features}")     
        
        features = np.expand_dims(features, axis = 0)
        logger.info(f"shape of features : {features.shape}")    
        predictionPrice = self.model.predict(features)
        predictionPrice = self.scaler.inverse_transform(predictionPrice)
        print(predictionPrice)
        # print(features)
        # return np.array(features), np.array(labels)
        # return dataset

if __name__ == "__main__":
    bestModelPath = projectDir + "/ModelFile/stock-TCS.ns_yfinance_2012-01-01-2023-12-09/bestmodel.keras"
    scalerPath = f"{projectDir}/scaler/stock-TCS.ns_yfinance_2012-01-01-2023-12-09.pkl"
    predictionObj = Prediction(symbol, interval, bestModelPath, scalerPath)
