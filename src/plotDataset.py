import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from click import secho
import os
# %matplotlib notebook

filePath = f"Dataset/AAPL_2022-12-01-2023-01-01.csv"
plotFolder = f"Plot"
plotFilename = filePath.split("/")[-1].replace("csv", "png")
plotFilePath = f"{plotFolder}/{plotFilename}"
stockMarketDataSet = pd.read_csv(filePath)
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=[f'Candlestick Chart'])

fig = go.Figure(data=[go.Candlestick(x=stockMarketDataSet.Date,
                open=stockMarketDataSet['Open'],
                high=stockMarketDataSet['High'],
                low=stockMarketDataSet['Low'],
                close=stockMarketDataSet['Close'])])

fig.update_layout(title=f'Stock Price - OHLC Chart',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  xaxis_rangeslider_visible=True)

fig.show()

if not os.path.exists(plotFolder):
    os.makedirs(plotFolder)
    secho(f"Directory Created : {plotFolder}", fg="green")

fig.write_image(plotFilePath)
secho(f"Plot saved successfully : {plotFilePath}", fg = 'green')