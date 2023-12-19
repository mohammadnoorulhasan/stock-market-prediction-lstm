import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("src")

from config import *

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from click import secho
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow as tf

class StockMarketLSTM:
    def __init__(self, dataPath, scalerPath, modelPath, nEpoches, patience):
        self.dataPath = dataPath
        self.scalerPath = scalerPath
        self.modelPath = modelPath
        self.columns = columns
        self.nEpoches =  nEpoches
        self.patience = patience

    def loadData(self):
        
        self.featuresLen = len(self.columns)
        stockMarketData = pd.read_csv(self.dataPath)
        stockMarketData = stockMarketData.reset_index()[self.columns]
        return stockMarketData

    def scaleData(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaledData = scaler.fit_transform(data)
        joblib.dump(scaler, self.scalerPath)
        return scaledData

    def createDataset(self, dataset, timeStep=10):
        features, labels = [],[]

        for i in range(len(dataset) - timeStep - 1):
            feature = dataset[i:(i + timeStep), :]
            features.append(feature)
            labels.append(dataset[i + timeStep, :])

        return np.array(features), np.array(labels)

    def buildModel(self, timeStep):
        model = Sequential()
        model.add(
            LSTM(
                200, 
                return_sequences=True, 
                input_shape=(timeStep, self.featuresLen), 
                activation = tf.nn.leaky_relu
            )
        )
        model.add(
            LSTM(
                200, 
                return_sequences=True,
                activation = tf.nn.leaky_relu
            )
        )
        model.add(
            LSTM(
                50, 
                activation = tf.nn.leaky_relu
            )
        )
        model.add(
            Dense(
                100,
                activation = tf.nn.leaky_relu
            )
        )
        model.add(
            Dense(
                50,
                activation = tf.nn.leaky_relu
            )
        )
        model.add(Dense(self.featuresLen))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = tf.keras.metrics.RootMeanSquaredError())
        return model

    def trainModel(self, trainingFeatures, trainingLabels, testingFeatures, testingLabels):
        checkpoint = ModelCheckpoint(
            self.modelPath + '/epoch_{epoch}_model_{val_loss:4f}.keras',
            monitor='val_loss', save_best_only=True, mode='min', verbose=1
        )
        # checkpointBestModel = ModelCheckpoint(bestModelPath,
        #     monitor='val_loss', 
        #     save_best_only=True, 
        #     mode='min', 
        #     verbose=1
        # )
        earlyStopping = EarlyStopping(
            monitor='val_loss', 
            patience=self.patience, 
            restore_best_weights=True,
            verbose = 1
        )
        self.lstmModelHistory = self.lstmModel.fit(
            trainingFeatures, 
            trainingLabels, 
            validation_data=(testingFeatures, testingLabels),
            epochs=self.nEpoches, 
            batch_size=64, 
            verbose=1, 
            callbacks=[checkpoint, earlyStopping]
        )

        self.lstmModel.save(bestModelPath)
        

    def main(self):
        stockMarketData = self.loadData()
        scaledData = self.scaleData(stockMarketData)

        trainingSize = int(len(scaledData) * 0.90)
        trainingData = scaledData[:trainingSize]
        testingData = scaledData[trainingSize:]

        
        trainingFeatures, trainingLabels = self.createDataset(trainingData, timeStep)
        testingFeatures, testingLabels = self.createDataset(testingData, timeStep)
        secho(f"Shape of trainingFeatures before reshaping: {trainingFeatures.shape}", fg="blue")
        secho(f"Shape of trainingLabels before reshaping: {trainingLabels.shape}", fg="blue")
        secho(f"Shape of testingFeatures before reshaping: {testingFeatures.shape}", fg="blue")
        secho(f"Shape of testingLabels before reshaping: {testingLabels.shape}", fg="blue")

        # trainingFeatures = trainingFeatures.reshape(trainingFeatures.shape[0], trainingFeatures.shape[1], 1)
        # testingFeatures = testingFeatures.reshape(testingFeatures.shape[0], testingFeatures.shape[1], 1)

        secho("*" * 50, fg="green")
        secho(f"Total Data Points: {len(scaledData)}")
        secho(f"Total Training Points: {len(trainingFeatures)}")
        secho(f"Total Testing Points: {len(testingFeatures)}")
        secho("*" * 50, fg="green")

        self.lstmModel = self.buildModel(timeStep)
        self.trainModel( trainingFeatures, trainingLabels, testingFeatures, testingLabels)
        scaler = joblib.load(self.scalerPath)
        self.lstmModel = load_model(bestModelPath)
        trainPredict=self.lstmModel.predict(trainingFeatures)
        testPredict=self.lstmModel.predict(testingFeatures)
        ##Transformback to original form
        trainPredict=scaler.inverse_transform(trainPredict)
        testPredict=scaler.inverse_transform(testPredict)
        trainingLabels=scaler.inverse_transform(trainingLabels)
        testingLabels=scaler.inverse_transform(testingLabels)

        # trainingLabels_str = [str(row) for row in trainingLabels]
        # trainPredict_str = [str(row) for row in trainPredict]
        print(trainPredict)
        df = pd.DataFrame(
            data = {
                'ActualOpen': trainingLabels[:,0],
                'PredictedOpen': trainPredict[:,0],
                'DifferenceOpen': abs(trainingLabels[:,0]-trainPredict[:,0]),
                'ActualHigh': trainingLabels[:,1],
                'PredictedHigh': trainPredict[:,1],
                'DifferenceHigh': abs(trainingLabels[:,1]-trainPredict[:,1]),
                'ActualLow': trainingLabels[:,2],
                'PredictedLow': trainPredict[:,2],
                'DifferenceLow': abs(trainingLabels[:,2]-trainPredict[:,2]),
                'ActualClose': trainingLabels[:,3],
                'PredictedClose': trainPredict[:,3],
                'DifferenceClose': abs(trainingLabels[:,3]-trainPredict[:,3])
            }
        )
        logger.info(f"shape of dataframe : {df.shape}")
        print(df.head())
        # Save the DataFrame to a CSV file
        df.to_csv('TCS_train.csv')

        # testingLabels_str = [str(row) for row in testingLabels]
        # testingPredict_str = [str(row) for row in testPredict]

        # df = pd.DataFrame(data = {'Label': testingLabels_str,
        #                 'Prediction': testingPredict_str})
        df = pd.DataFrame(
            data = {
                'ActualOpen': testingLabels[:,0],
                'PredictedOpen': testPredict[:,0],
                'DifferenceOpen': testingLabels[:,0]-testPredict[:,0],
                'ActualHigh': testingLabels[:,1],
                'PredictedHigh': testPredict[:,1],
                'DifferenceHigh': testingLabels[:,1]-testPredict[:,1],
                'ActualLow': testingLabels[:,2],
                'PredictedLow': testPredict[:,2],
                'DifferenceLow':testingLabels[:,2]-testPredict[:,2],
                'ActualClose': testingLabels[:,3],
                'PredictedClose': testPredict[:,3],
                'DifferenceClose': testingLabels[:,3]-testPredict[:,3]
            }
        )
        # # Save the DataFrame to a CSV file
        df.to_csv('TCS_testing.csv')

if __name__ == "__main__":
    # dataPath = "../data/stock-TCS.ns_yfinance_2012-01-01-2023-12-03.csv"
    # scalerPath = '../scaler/TCS.pkl'
    # modelPath = '../modelFiles/TCS/'
    
    stockMarketLSTM = StockMarketLSTM(dataFilePath, scalerPath, modelPath)
    stockMarketLSTM.main()
