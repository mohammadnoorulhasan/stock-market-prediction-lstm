import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("src")

from config import *

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from click import secho
import plotly.graph_objects as go


class Inference:
    def __init__(self, dataPath, scalerPath, modelPath):
        self.dataPath = dataPath
        self.scalerPath = scalerPath
        self.modelPath = modelPath
        self.main()

    def loadData(self):
        self.columns = columns
        self.featuresLen = len(self.columns)
        stockMarketData = pd.read_csv(self.dataPath)
        stockMarketData = stockMarketData.reset_index()[self.columns]
        print(stockMarketData.head())
        logger.info(stockMarketData.head())
        return stockMarketData
    
    def scaleData(self, data):
        scaler = joblib.load(self.scalerPath)
        scaledData = scaler.transform(data)
        # logger.info(scaledData)
        # joblib.dump(scaler, self.scalerPath)
        return scaledData
    
    def createDataset(self, dataset, timeStep=10):
        features, labels = [],[]

        for i in range(len(dataset) - timeStep - 1):
            feature = dataset[i:(i + timeStep), :]
            features.append(feature)
            labels.append(dataset[i + timeStep, :])

        return np.array(features[0]), np.array(labels)
    
    def predictNDays(self, features, labels, N):
        predictionList = []
        scaler = joblib.load(self.scalerPath)
        
        for i in range(N-1):
            prediction = self.lstmModel.predict(features)

            # Update features for the next iteration
            features = np.concatenate([features[:, 1:], np.expand_dims(prediction, axis=0)], axis=1)

            # Inverse transform the prediction
            prediction = scaler.inverse_transform(prediction)
            
            predictionList.append(prediction)

        logger.info(len(predictionList))
        logger.info(labels.shape)

        # Flatten the labels if they are not already 1-dimensional
        labels_flat = labels.flatten()
        labels = scaler.inverse_transform(labels)
        # Create DataFrame with 1-dimensional arrays

        # df = pd.DataFrame(
        #     data={
        #         'Actual': labels,
        #         'Predicted': predictionList
        #     }
        # )
        
        # df.to_csv("test.csv", index=False)
        predictionList  = np.array(predictionList)
        print(predictionList.shape)
        predictionList = np.squeeze(predictionList, axis = 1)
        for i, j in zip (predictionList, labels):
            print(j,i)

        print(f"Shape of actual labels : {labels.shape}")
        print(f"Shape of prediction labels : {predictionList.shape}")
        df = pd.DataFrame(
            data = {
                'ActualOpen': labels[:,0],
                'PredictedOpen': predictionList[:,0],
                'ActualHigh': labels[:,1],
                'PredictedHigh': predictionList[:,1],
                'ActualLow': labels[:,2],
                'PredictedLow': predictionList[:,2],
                'ActualClose': labels[:,3],
                'PredictedClose': predictionList[:,3],
            }
        )
        print(df)
        return df

    def plotCandleStickChart(self, df):
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['ActualOpen'],
                high=df['ActualHigh'],
                low=df['ActualLow'],
                close=df['ActualClose'],
                name='Actual',
                increasing_line_color='green',  # Color for increasing values
                decreasing_line_color='red'    # Color for decreasing values
            ),
        go.Candlestick(x=df.index,
            open=df['PredictedOpen'],
            high=df['PredictedHigh'],
            low=df['PredictedLow'],
            close=df['PredictedClose'],
            name='Predicted',
            increasing_line_color='blue',  # Color for increasing values
            decreasing_line_color='orange' # Color for decreasing values
        )])

        # Update layout for better visualization
        fig.update_layout(
            title='Actual vs Predicted Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )

        # Show the plot
        fig.show()

    def main(self):
        stockMarketData = self.loadData()
        scaledData = self.scaleData(stockMarketData)

        trainingSize = int(len(scaledData) * 0.95)
        scaledDataLen = len(scaledData)
        timeStep = 150
        N = 10
        inferenceData = scaledData[scaledDataLen-timeStep-N:]
        # trainingData = scaledData[:trainingSize]
        # testingData = scaledData[trainingSize:]
        logger.info(f"InferenceData.shape : {inferenceData.shape}")
        
        trainingFeatures, trainingLabels = self.createDataset(inferenceData, timeStep)
        # testingFeatures, testingLabels = self.createDataset(testingData, timeStep)
        trainingFeatures = np.expand_dims(trainingFeatures, axis=0)
        secho(f"Shape of trainingFeatures before reshaping: {trainingFeatures.shape}", fg="blue")
        secho(f"Shape of trainingLabels before reshaping: {trainingLabels.shape}", fg="blue")
        
        # trainingLabels = np.expand_dims(trainingLabels, axis=1)
        
        self.lstmModel = load_model(bestModelPath)
        df = self.predictNDays(trainingFeatures, trainingLabels, 10)
        self.plotCandleStickChart(df)
        

if __name__ == "__main__":
    print(scalerPath)
    Inference(dataFilePath, scalerPath, bestModelPath)