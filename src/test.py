import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from click import secho
from matplotlib import pyplot as plt
# Load the best model
model = load_model('../modelFiles/TCS/epoch_90_model_0.000141.h5')
scaler = joblib.load('../scaler/TCS.pkl')


filePath = "../data/stock-TCS.ns_yfinance_2012-01-01-2023-12-03.csv"
stockMarketData = pd.read_csv(filePath)

stockMarketData = stockMarketData.reset_index()['Close']

# scaler=MinMaxScaler(feature_range=(0,1))
stockMarketData=scaler.fit_transform(np.array(stockMarketData).reshape(-1,1))


def createDataset(dataset, timeStep=10):
	features, labels = [], []
	for i in range(len(dataset)-timeStep-1):
		feature = dataset[i:(i+timeStep)]   ###i=0, 0,1,2,3-----99   100 
		features.append(feature)
		labels.append(dataset[i + timeStep])
	
	return np.array(features), np.array(labels)

trainingSize = int(len(stockMarketData)*0.65)
testingSize = len(stockMarketData) - trainingSize
trainingStockMarketData = stockMarketData[0: trainingSize]
testingStockMarketData = stockMarketData[trainingSize: len(stockMarketData)]

timeStep = 100
trainingFeatures, trainingLabels = createDataset(trainingStockMarketData, timeStep)
testingFeatures, testingLabels = createDataset(testingStockMarketData, timeStep)
trainPredict=model.predict(trainingFeatures)
testPredict=model.predict(testingFeatures)

##Transformback to original form
trainPredict=scaler.inverse_transform(trainPredict)
testPredict=scaler.inverse_transform(testPredict)
trainingLabels=scaler.inverse_transform(trainingLabels)
testingLabels=scaler.inverse_transform(testingLabels)

# secho(f"MSE : {math.sqrt(mean_squared_error(trainingLabels,trainPredict))}", fg = 'green')
# secho(f"MSE : {math.sqrt(mean_squared_error(testingLabels,testPredict))}", fg = 'green')

# secho(f"*"*50, fg = "green")
# secho(f"Total Data Points : {len(stockMarketData)}")
# secho(f"Total Training Points : {len(trainPredict)}")
# secho(f"Total Testing Points : {len(testPredict)}")

# look_back=100
# trainPredictPlot = np.empty_like(stockMarketData)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(stockMarketData)
# testPredictPlot[:, :] = np.nan
# print(f"len(testPredict)+(look_back*2)+1 {len(testPredict)+(look_back*2)+1}")
# testPredictPlot[len(testPredict)+(look_back*2)+1:len(stockMarketData)-1] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(stockMarketData))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
trainingLabels_str = [str(row) for row in trainingLabels]
trainPredict_str = [str(row) for row in trainPredict]

df = pd.DataFrame(data = {'Label': trainingLabels_str,
                   'Prediction': trainPredict_str})

# Save the DataFrame to a CSV file
df.to_csv('TCS_train.csv', index=False)

testingLabels_str = [str(row) for row in testingLabels]
testingPredict_str = [str(row) for row in testPredict]

df = pd.DataFrame(data = {'Label': testingLabels_str,
                   'Prediction': testingPredict_str})

# Save the DataFrame to a CSV file
df.to_csv('TCS_testing.csv', index=False)