from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt
#
# # Import data.
# PATH = "data/"
# FILE = "drugSales.csv"
# df = pd.read_csv(PATH + FILE, parse_dates=['date'], index_col='date')
# type(df.index)
#
# # Perform decomposition using multiplicative decomposition.  #디버깅해보면 1개의 칼럼으로 보일 것. 날짜별 판매량이.
# tseries = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend="freq")
#
# tseries.plot()
# plt.show()
#
#
#
# # 위에 multiplicative로 했던 것이라 actual_values가 저 셋의 곱으로 나옴.
# dfComponents = pd.concat([tseries.seasonal, tseries.trend,
#                           tseries.resid, tseries.observed], axis=1)
# dfComponents.columns = ['seas', 'trend', 'resid', 'actual_values']
# print(dfComponents.head())
# print(tseries.observed[1])
#
#


from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
PATH      = "data/"
FILE      = "Airpassengers.csv"
df        = pd.read_csv(PATH + FILE, parse_dates=['date'], index_col='date')
tseries   = seasonal_decompose(df['value'], model='multiplicative',
                               extrapolate_trend='freq')

plt.plot(df['value'])
plt.title("Drug Sales", fontsize=16)
plt.show()

deseasonalized = df.value.values / tseries.seasonal
plt.plot(deseasonalized)
plt.title('Air passengers After De-Seasonalizing', fontsize=16)
plt.show()




#
#
#
# from pandas import read_csv
# import matplotlib.pyplot as plt
#
# PATH = "data/"
# FILE = 'daily-total-female-births.csv'
# series = read_csv(PATH + FILE, header=0, index_col=0)
# print(series.head())
# series.plot(rot=45)
# plt.show()
#
# # Calculate rolling moving average 3 steps back.
# print("\n*** Rolling mean")
# rolling = series.rolling(window=3)
# rolling_mean = rolling.mean()
# print(rolling_mean.head(5))
#
# # Plot actual and rolling mean values.
# plt.plot(series, color='blue', label='female births')
# plt.plot(rolling_mean, color='red', label='rolling mean')
# plt.legend()
# plt.show()
#
# from pandas_datareader import data as pdr
# import yfinance as yfin  # Work around until
# # pandas_datareader is fixed.
# import datetime
# import matplotlib.pyplot as plt
#
#
# def getStock(stk, ttlDays):
#     numDays = int(ttlDays)
#     # Only gets up until day before during
#     # trading hours
#     dt = datetime.date.today()
#     # For some reason, must add 1 day to get current stock prices
#     # during trade hours. (Prices are about 15 min behind actual prices.)
#     dtNow = dt + datetime.timedelta(days=1)
#     dtNowStr = dtNow.strftime("%Y-%m-%d")
#     dtPast = dt + datetime.timedelta(days=-numDays)
#     dtPastStr = dtPast.strftime("%Y-%m-%d")
#     yfin.pdr_override()
#     df = pdr.get_data_yahoo(stk, start=dtPastStr, end=dtNowStr)
#     return df
#
#
# df = getStock('MSFT', 200)
#
# # Calculating the moving averages.
# # rolling_mean = df['Close'].rolling(window=20).mean()
# rolling_mean2 = df['Close'].rolling(window=50).mean()
#
# # Calculate the exponentially smoothed series.
# # exp20 = df['Close'].ewm(span=20, adjust=False).mean()
# exp50 = df['Close'].ewm(span=50, adjust=False).mean()
#
# # plt.figure(figsize=(10,30))
# df['Close'].plot(label='MSFT Close ', color='gray', alpha=0.3)
# # rolling_mean.plot(label='MSFT 20 Day MA', style='--', color='orange')
# rolling_mean2.plot(label='MSFT 50 Day MA', style='--', color='red')
# # exp20.plot(label='MSFT 20 Day ES', style='--', color='green')
# exp50.plot(label='MSFT 50 Day ES', style='--', color='blue', alpha=0.5)
# plt.legend()
# plt.show()
