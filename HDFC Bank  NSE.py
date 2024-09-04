#!/usr/bin/env python
# coding: utf-8

# # Data Visulization for NSE(National Stock Exchange) of HDFC Bank

# why data visualization for the NSE of HDFC Bank?
# 
# Data visualization enhances the ability to interpret, analyze, and act on financial data, making it a powerful tool for anyone involved to invest by seeeing the graphs and charts.
# Visualization tools like line charts and bar graphs help investors and analysts quickly see trends in HDFC Bank’s stock price, trading volume, and other key metrics over time. This can aid in identifying patterns and making predictions about future performance.
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# In[2]:


df=pd.read_csv('Quote-Equity-HDFCBANK-EQ-27-08-2023-to-27-08-2024.csv')
df.head()


# # Data understanding

# Date: The specific trading day for which the data is reported.
# 
# Series: The trading series or segment where the stock is listed, such as EQ for Equity, BE for Buyback, etc.
# 
# OPEN: The price at which the stock first trades upon the opening of the market.
# 
# HIGH: The highest price at which the stock traded during the trading session.
# 
# LOW: The lowest price at which the stock traded during the trading session.
# 
# PREV. CLOSE: The closing price of the stock on the previous trading day.
# 
# ltp (Last Traded Price): The most recent price at which the stock was traded during the session.
# 
# close: The final price at which the stock was traded on the given trading day.
# 
# vwap (Volume Weighted Average Price): The average price at which the stock was traded, weighted by the volume of trades. It provides an indication of the average price at which most trading occurred.
# 
# 52W H: The highest price at which the stock has traded over the past 52 weeks.
# 
# 52W L: The lowest price at which the stock has traded over the past 52 weeks.
# 
# VOLUME: The total number of shares traded during the trading session.
# 
# VALUE: The total monetary value of the shares traded during the trading session. It is calculated as VOLUME x ltP.
# 
# No of trades: The number of individual trades executed during the trading session.

# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df.rename(columns={'OPEN ': 'Open'}, inplace=True)
df.rename(columns={'HIGH ': 'High'}, inplace=True)
df.rename(columns={'LOW ': 'Low'}, inplace=True)
df.rename(columns={'close ': 'Close'}, inplace=True)
df.rename(columns={'vwap ': 'vwap'}, inplace=True)
df.rename(columns={'52W H ': '52WH'}, inplace=True)
df.rename(columns={'52W L ': '52WL'}, inplace=True)
df.rename(columns={'VOLUME ': 'Volume'}, inplace=True)
df.rename(columns={'Date ': 'Date'}, inplace=True)
df.rename(columns={'series ': 'series'}, inplace=True)
df.rename(columns={'PREV. CLOSE ': 'Prev.Close'}, inplace=True)
df.rename(columns={'ltp ': 'ltp'}, inplace=True)
df.rename(columns={'VALUE ': 'Value'}, inplace=True)
df.rename(columns={'No of trades ': 'No of trades'}, inplace=True)


# In[9]:


df.head()


# In[10]:


df.dtypes


# In[11]:


for col in ['Date', 'Open', 'High', 'Low', 'Prev.Close', 'ltp', 'Close', 'Volume', 'vwap', '52WH', '52WL', 'Value', 'No of trades']:
    df[col] = df[col].astype(str).str.replace(',', '')  # Remove commas
    df[col] = df[col].str.strip()  # Remove any whitespace
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric

# Handle NaN values
df.fillna(0, inplace=True)


# In[12]:


df.dtypes


# # Important Columns for Data Visualization
# "Date": Essential for time series analysis. Always keep this column as it forms the basis of time-based plots.
# 
# "Open, High, Low, Close":
# 
# "Open": Useful to show where the stock price started the day.
# 
# "High and Low": Important for understanding the range of price movements during the day.
# "Close": Often the most significant price point, used for daily trends and calculations like moving averages.
# "Last Traded Price (LTP)": Similar to Close, it shows the most recent price at the end of trading hours. Important for understanding the final traded price.
# 
# 
# "Volume": Essential for analyzing trading activity. High volume can indicate strong interest or significant events.
# 
# "Use Case: Bar charts to show trading volume trends."
# 
# 
# "VWAP (Volume Weighted Average Price)": Useful for understanding the average price at which most trading occurred, weighted by volume.
# 
# "Use Case: Line charts to compare with price trends."
# 
# "52W High, 52W Low": Shows the stocks highest and lowest prices over the past year. Important for understanding long-term price trends and stock volatility.
# 
# Use Case: Reference lines or annotations in price trend charts.

# In[13]:


df.drop(columns=['series', 'Prev.Close', 'ltp', 'Value', 'No of trades'], axis=1, inplace=True)


# In[14]:


df.columns


# # EDA 

# # For this NSE of HDFC Bank we Analysie and plot these charts and graphs
1.Time series Data
2.Moving Averages
3.Daily Returns
4.Candelstick chart
5.Combined Price and volume chart
6.Histogram Return
7.Correlation Analysis
8.Volume Analysis
9.Rolling Statistics
10.Box Plots
11.Pairwise Relationship
12.Volatility Analysis
13.Cummulative return
14.calculate Rolling Correlation
15.For multiple Rolling Correlation
16.Violin Plots
# # Plot Time Series Data

# In[15]:


# Plot Open, High, Low, Close*, and Adj Close** prices
plt.figure(figsize=(14, 8))
plt.plot(df['Open'], label='Open', color='blue')
plt.plot(df['High'], label='High', color='green')
plt.plot(df['Low'], label='Low', color='red')
plt.plot(df['Close'], label='Close', color='orange')
plt.plot(df['vwap'], label='vwap', color='purple')

plt.title('Stock Prices Over Time')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# # Moving Averages

# In[16]:


# Calculate moving averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_100'] = df['Close'].rolling(window=100).mean()


# In[17]:


plt.figure(figsize=(14, 8))
plt.plot(df['Close'], label='Close', color='orange')
plt.plot(df['SMA_50'], label='50-Day SMA', color='red')
plt.plot(df['SMA_100'], label='100-Day SMA', color='green')

plt.title('Close* Price with Moving Averages')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# # Daily Returns

# In[18]:


# Calculate daily returns
df['Daily_Return'] = df['Close'].pct_change()


# In[19]:


plt.figure(figsize=(14, 8))
plt.plot(df['Daily_Return'], label='Daily Return', color='orange')
plt.title('Daily Returns')
plt.xlabel('Days')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True)
plt.show()


# # Candlestick Chart

# In[20]:


fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

candlestick = go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Candlestick')
fig.add_trace(candlestick)

fig.update_layout(title='Candlestick Chart',
                  xaxis_title='Days',
                  yaxis_title='Price')
fig.show()


# # Combined Price and Volume Chart

# In[21]:


fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.1, subplot_titles=('OHLC', 'Volume'), #'OHLC'='open','Low','high','close'
                    row_width=[0.2, 0.7])

fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Candlestick'), row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

fig.update_layout(title='Stock Price and Volume',
                  xaxis_title='Days',
                  yaxis_title='Price',
                  yaxis2_title='Volume')

fig.show()


# # Histogram of Returns

# In[22]:


plt.figure(figsize=(14, 8))
sns.histplot(df['Daily_Return'].dropna(), bins=50, kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# # Correlation Analysis

# In[23]:


# Compute correlation matrix
correlation_matrix = df[['Open', 'High', 'Low', 'Close', 'vwap', 'Volume','52WH','52WL']].corr()


# In[24]:


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# # Volume Analysis

# In[25]:


plt.figure(figsize=(14, 8))
plt.plot(df['Volume'], label='Volume', color='blue')
plt.title('Trading Volume Over Time')
plt.xlabel('DaYS')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()


# # Rolling Statistics

# In[26]:


# Calculate rolling mean and rolling standard deviation
df['Rolling_Mean'] = df['Close'].rolling(window=30).mean()
df['Rolling_Std'] = df['Close'].rolling(window=30).std()


# In[27]:


plt.figure(figsize=(14, 8))
plt.plot(df['Close'], label='Close Price', color='orange')
plt.plot(df['Rolling_Mean'], label='30-Day Rolling Mean', color='red')
plt.plot(df['Rolling_Std'], label='30-Day Rolling Std Dev', color='blue')
plt.title('Close Price with Rolling Statistics')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


# # Box Plots

# In[28]:


plt.figure(figsize=(14, 8))
plt.subplot(2, 3, 1)
sns.boxplot(df['Open'])
plt.title('Box Plot of Open Prices')

plt.subplot(2, 3, 2)
sns.boxplot(df['High'])
plt.title('Box Plot of High Prices')

plt.subplot(2, 3, 3)
sns.boxplot(df['Low'])
plt.title('Box Plot of Low Prices')

plt.subplot(2, 3, 4)
sns.boxplot(df['Close'])
plt.title('Box Plot of Close* Prices')

plt.subplot(2, 3, 5)
sns.boxplot(df['Volume'])
plt.title('Box Plot of Volume')

plt.tight_layout()
plt.show()


# # Pairwise Relationships

# In[29]:


sns.pairplot(df[['Open', 'High', 'Low', 'Close', 'Volume']])
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()


# # Volatility Analysis

# In[30]:


# Calculate volatility (30-day rolling standard deviation of daily returns)
df['Volatility'] = df['Daily_Return'].rolling(window=30).std()


# In[31]:


plt.figure(figsize=(14, 8))
plt.plot(df['Volatility'], label='30-Day Volatility', color='purple')
plt.title('Volatility Over Time')
plt.xlabel('Days')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


# # Cumulative Returns

# In[32]:


# Calculate cumulative returns
df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()


# In[33]:


plt.figure(figsize=(14, 8))
plt.plot(df['Cumulative_Return'], label='Cumulative Return', color='green')
plt.title('Cumulative Returns Over Time')
plt.xlabel('Days')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()


# # Calculate Rolling Correlation

# In[34]:


# Calculate rolling correlation between 'Close*' and 'Volume'
window_size = 30  # Define the rolling window size
df['Rolling_Correlation'] = df['Close'].rolling(window=window_size).corr(df['Volume'])


# In[35]:


plt.figure(figsize=(14, 8))
plt.plot(df['Rolling_Correlation'], label=f'Rolling Correlation (window={window_size} days)', color='purple')
plt.title('Rolling Correlation Between Close and Volume')
plt.xlabel('Days')
plt.ylabel('Rolling Correlation')
plt.legend()
plt.grid(True)
plt.show()


# # for mulitple rolling correlation

# In[36]:


pairs = [('Open', 'Close'), ('High', 'Volume')]


# In[37]:


plt.figure(figsize=(16, 10))

for i, (col1, col2) in enumerate(pairs, 1):
    df[f'Rolling_Correlation_{col1}_{col2}'] = df[col1].rolling(window=window_size).corr(df[col2])
    
    # Plot each rolling correlation
    plt.subplot(len(pairs), 1, i)
    plt.plot(df[f'Rolling_Correlation_{col1}_{col2}'], label=f'Rolling Correlation ({col1} vs {col2})', color='blue')
    plt.title(f'Rolling Correlation Between {col1} and {col2}')
    plt.xlabel('Days')
    plt.ylabel('Rolling Correlation')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


# # Violin Plots

# In[38]:


columns = ['Open', 'High', 'Low', 'Close', 'vwap', 'Volume']


# In[39]:


plt.figure(figsize=(16, 12))
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink']


for i, column in enumerate(columns, 1):
    plt.subplot(2, 3, i)  # 2 rows, 3 columns, i-th subplot
    color = colors[(i - 1) % len(colors)]
    sns.violinplot(y=df[column], color=color)
    plt.title(f'Violin Plot of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


# # Stream Lit App 

# In[40]:


import streamlit as st


# In[42]:


def plot_line_chart(data):
    # Create a line chart for 'Open', 'High', 'Low', 'Close'
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Open'], label='Open', color='blue')
    ax.plot(data['Date'], data['High'], label='High', color='green')
    ax.plot(data['Date'], data['Low'], label='Low', color='red')
    ax.plot(data['Date'], data['Close'], label='Close', color='black')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Prices')
    ax.legend()
    st.pyplot(fig)
    
def plot_candlestick_chart(data):
    # Create a candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    
def main():
    st.title("NSE Stock Data Visualization")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)

        # Ensure 'Date' column is in datetime format
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Show data preview
        st.write("Data Preview:", data.head())

        # Plot Line Chart
        st.subheader("Line Chart of Stock Prices")
        plot_line_chart(data)

        # Plot Candlestick Chart
        st.subheader("Candlestick Chart")
        plot_candlestick_chart(data)

if __name__ == "__main__":
    main()


# In[ ]:




