import pandas as pd
from matplotlib import pyplot as plt
# % matplotlib inline
# data = pd.read_csv("TCS_train.csv")
df = pd.read_csv('TCS_train.csv')

# Assuming your CSV has 'Actual' and 'Prediction' columns
actual_values = df['Label']
prediction_values = df['Prediction']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(actual_values, label='Label')
plt.plot(prediction_values, label='Prediction', linestyle='dashed', marker='o')
plt.title('Actual vs Prediction')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.legend()
plt.show()