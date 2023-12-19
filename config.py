import logging
import os
from datetime import timedelta, datetime
from colorlog import ColoredFormatter
import click
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

symbol = "EURUSD=X"
symbol = "TCS.ns"
startDate = '2012-01-01'
endDate = None
interval = "1d"
# endDate = '2023-12-04'
# downloadModule = "nsepython"
downloadModule = "yfinance"
columns = ["Open", "High", "Low", "Close", "Volume"]
timeStep = 150
nEpoches = 1000
patience = 20
if endDate is None:
    currentDatetime = datetime.now()
    # Calculate the end date of yesterday
    endDate = currentDatetime - timedelta(days=1)
    endDate = endDate.strftime('%Y-%m-%d')

projectDir = "/Users/niki/Work/Noor/Stock-Market/LSTM/"
dataFolder = os.path.join(projectDir, "Data")   
datasetFolder = os.path.join(projectDir, "Dataset")
filename = f"stock-{symbol}_{downloadModule}_{startDate}-{endDate}"
csvFilename = f"{filename}.csv"
# Create file path
dataFilePath = os.path.join(dataFolder, csvFilename)
datasetFilePath = os.path.join(datasetFolder, csvFilename)

scalerDir = os.path.join(projectDir, "scaler")
scalerPath = os.path.join(scalerDir, f"{filename}.pkl")
modelPath = os.path.join(projectDir, f"ModelFile/{filename}/")
bestModelPath = os.path.join(projectDir, f"ModelFile/{filename}/bestmodel.keras")

DEBUG_FLAG = True
PRINT_FLAG = True

# Define Function

def secho(message, fg=None, bg=None, bold=False, showTimeStamp = True):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if showTimeStamp:
        formattedMessage = f"{timestamp}-{message}"
        secho(formattedMessage, fg=fg, bg=bg, bold=bold)
    else:
        click.secho(message, fg=fg, bg=bg, bold=bold)