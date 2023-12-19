import pandas as pd
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('TCS_train.csv')[2600:]


# Create a candlestick chart
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
