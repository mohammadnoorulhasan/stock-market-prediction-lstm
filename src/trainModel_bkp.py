import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from click import secho
import joblib
import logging

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

filePath = "../data/stock-TCS.ns_yfinance_2012-01-01-2023-12-03.csv"
stockMarketData = pd.read_csv(filePath)

stockMarketData = stockMarketData.reset_index()[["Open","High","Low","Close","Volume"]]
logging.info(stockMarketData.head())
scaler=MinMaxScaler(feature_range=(0,1))
stockMarketData=scaler.fit_transform(np.array(stockMarketData).reshape(-1,1))
logging.info(stockMarketData)
# print(stockMarketData.head())
trainingSize = int(len(stockMarketData)*0.80)
testingSize = len(stockMarketData) - trainingSize
logger.info(f"trainingSize : {trainingSize}")
trainingStockMarketData = stockMarketData[0: trainingSize]
testingStockMarketData = stockMarketData[trainingSize: len(stockMarketData)]

# print(trainingStockMarketData)
# convert an array of values into a dataset matrix
def createDataset(dataset, timeStep=10):
	features, labels = [], []

	for i in range(len(dataset)-timeStep-1):
		feature = dataset[i:(i+timeStep),:]   ###i=0, 0,1,2,3-----99   100 
		features.append(feature)
		labels.append(dataset[i + timeStep,:])
		break
	return np.array(features), np.array(labels)

timeStep = 100
trainingFeatures, trainingLabels = createDataset(trainingStockMarketData, timeStep)
logger.info(trainingFeatures)
logger.info(trainingLabels)
testingFeatures, testingLabels = createDataset(testingStockMarketData, timeStep)

trainingFeatures =trainingFeatures.reshape(trainingFeatures.shape[0],trainingFeatures.shape[1] , 1)
testingFeatures = testingFeatures.reshape(testingFeatures.shape[0],testingFeatures.shape[1] , 1)
secho(f"*"*50, fg = "green")
secho(f"Total Data Points : {len(stockMarketData)}")
secho(f"Total Training Points : {len(trainingFeatures)}")
secho(f"Total Testing Points : {len(testingFeatures)}")
# secho(f"Total Data Points : {len(stockMarketData)}")
secho(f"*"*50, fg = "green")


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(timeStep,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

checkpoint = ModelCheckpoint('modelFiles/TCS/epoch_{epoch}_model_{val_loss:4f}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with the callback
model.fit(trainingFeatures, trainingLabels, validation_data=(testingFeatures, testingLabels), epochs=100, batch_size=64, verbose=1, callbacks=[checkpoint])

joblib.dump(scaler, '../scaler/TCS.pkl')
