import streamlit as st
import yfinance as yf  
#from yahoo_fin.stock_info import get_data

# data analysis and wrangling
import pandas as pd
from pandas_datareader import data
import numpy as np
import random as rnd
import datetime

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

# plotly
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
# import cufflinks as cf
# cf.go_offline()

#Technical Analysis
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

#Library For modelling
from scipy import stats
from scipy.stats import zscore
from pmdarima.arima.utils import ndiffs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product
from tqdm.notebook import tqdm
import datetime
from datetime import date, timedelta

#Warning
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

st.markdown("<h1 style='text-align: center; color: blue;'>STOCK TECHNICAL ANALYSIS AND FORECASTING</h1>", unsafe_allow_html=True)

#Staart and end date
st.subheader("Entered Time Period")
start_date = st.sidebar.date_input('start date', datetime.date(2019,1,1))
st.write(start_date)
end_date = st.sidebar.date_input('end date', pd.to_datetime('today', format='%Y-%m-%d'))
st.write(end_date)

#stock to analize
st.subheader('Stock Ticker Name Yahoo')
tickerSymbol=st.text_input('Enter stock ticker',value='KOTAKBANK.NS')
#st.write('The current movie title is', titleSymbol)

#Data for stock
df = data.DataReader(tickerSymbol, 'yahoo', start_date, end_date)
#data= get_data(tickerSymbol, start_date=start_date, end_date=end_date, index_as_date = True, interval="1d")

#plotting stock
st.subheader('Candlestick Plot of Stock')
fig1=ff.create_candlestick(df['Open'],df['High'],df['Low'],df['Close'], dates=df.index)
st.plotly_chart(fig1, use_container_width=True)

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)   
plot_raw_data()

#Maximum and minimum price and date
st.subheader('Basic Stats')
max_price=round(df['Close'].max(),2)
max_date=df['Close'].idxmax().date()
min_price=round(df['Close'].min(),2)
min_date=df['Close'].idxmin().date()
st.write('The maximum price of the stock is ')
st.text(max_price)
st.write('attained on ')
st.text(max_date)
st.write('The minimum price of the stock is ')
st.text(min_price)
st.write('attained on ')
st.text(min_date)

#return for stock
returns = df['Close'].pct_change().dropna()

#return distribution
st.subheader('Return Distribution Of Stock')
fig2=plt.figure(figsize=(8,4))
ax = sns.distplot(returns, color='green', bins=50)
ax.set_title('Return distribution')
ax.set_xlabel('Returns')
ax.set_ylabel('Numbers of Returns')
st.pyplot(fig2)

#Moving Average
st.subheader('Moving Average')
n=int(st.text_input('Enter Rolling Window',value=30))
st.line_chart(df['Close'].rolling(window=n).mean().dropna())

# Bollinger Bands
indicator_bb = BollingerBands(df['Close'])
bb = df.copy()
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['Close','bb_h','bb_l']]

# Moving Average Convergence Divergence
macd = MACD(df['Close']).macd()

# Resistence Strength Indicator
rsi = RSIIndicator(df['Close']).rsi()

# Plot the prices and the bolinger bands
st.write('Stock Bollinger Bands')
st.line_chart(bb)

progress_bar = st.progress(0)

# Plot MACD
st.write('Stock Moving Average Convergence Divergence (MACD)')
st.area_chart(macd)

# Plot RSI
st.write('Stock RSI ')
st.line_chart(rsi)

# Data of recent days
st.write('Recent data ')
st.dataframe(df.tail(10))

# # Predict forecast with Prophet.
# df['Date']=df.index
# df_train = df[['Date','Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# # Show and plot forecast
# st.subheader('Forecast data')
# st.write(forecast.tail())
    
# st.write(f'Forecast plot for {n_years} years')
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1)

# st.write("Forecast components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)

def optimise_ARIMA(order_list, exog):
    results = []

    for order in tqdm(order_list):
        try:
            model = SARIMAX(exog, order=order).fit(disp=0)
        except:
            continue
        aic = model.aic
        results.append([order, model.aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df

# Parameters to iterate through
ps = range(0, 8, 1)
d = 1
qs = range(0, 8, 1)

# Create a list with all possible combination of parameters
parameters = product(ps, qs)
parameters_list = list(parameters)
order_list = []
for each in parameters_list:
    each = list(each)
    each.insert(1, 1)
    each = tuple(each)
    order_list.append(each)
result_df = optimise_ARIMA(order_list, exog=df['Close'])

# Print dataframe of results
print(result_df.head())
order=result_df[result_df['AIC']==result_df['AIC'].min()]
(p, d, q)=order['(p, d, q)'][0]

# Fit and print a summary of the best model, which is ARIMA (p, d, q)
best_model = sm.tsa.arima.ARIMA(df['Close'], order=(p, d, q)).fit()

# Forecast
st.subheader('Stock Forecast number of days')
n=st.number_input('Enter days',value=7)
d=best_model.forecast(n, alpha=0.05)
index_date = pd.date_range(start=df.index[-1], end=df.index[-1]+timedelta(days=n-1), freq="1D")
d.index = index_date
d=d.to_frame()
d.rename(columns = {'predicted_mean':'Forecast Close'}, inplace = True)
q=df['Close'].to_frame()
q.rename(columns = {'Close':'Close'}, inplace = True)
frames=[q,d]
c=pd.concat(frames)

# hist_data = [df['Close'],d]
#st.line_chart(c)
# st.plotly_chart(fig)

# Plot raw data
st.subheader('Forecast Plot')
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d['Forecast Close'], name="Forecast"))
    fig.add_trace(go.Scatter(x=c.index, y=df['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)   
plot_raw_data()


# Forecast
st.subheader('APP Recommendation')
fc=best_model.forecast(n, alpha=0.05) 
if ((df['Close'].iloc[(df.shape[0]-30):]>fc.mean())[0]==True):
    st.write('BUY')
else:
    st.write('SELL')
