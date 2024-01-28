import datareader
import pandas as pd
from datetime import datetime


#Datetime Object
dt1 = datetime(year=2015, month=12, day=4)
dt2 = pd.to_datetime('12/8/1952')
dt3 = pd.to_datetime('12/8/1952', dayfirst=True)

print(dt1)
print(dt2)
print(dt3)


#Datetime -> Date obj
d1 = dt1.date()
print(d1)
d2 = dt2.date()
print(d2)
d3 = dt3.date()
print(d3)




#Dataframe용 datetime format, DatetimeIndex로의 변환까지.
import pandas as pd

df = pd.DataFrame(
    data={
        "Dates":['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'],
        "Temperature":[28,27,29,26]
    }
)
print(df)

#스트링 칼럼 -> datetime 포맷
df['Dates'] = pd.to_datetime(df['Dates'])
df          = df.set_index('Dates')
print(df)
print(type(df))
print("Index data type: ")
print(type(df.index))



import pandas as pd
PATH = "data/"
FILE = "aritzia.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['Date'], index_col='Date')
print(type(df.index)) # Verify the data type.
print(df)

df['year']    = df.index.year
df['month']   = df.index.month
df['day']     = df.index.day
df['dayName'] = df.index.strftime("%A")
print(df)


#Frequency
print("""FREQUENCY!!

""")
co2 = [
342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27, 344.21,
342.88, 342.58, 343.99, 345.31, 345.98, 346.72, 347.63, 349.24, 349.83, 349.10,
347.52, 345.43, 344.48, 343.89, 345.29, 346.54, 347.66, 348.07, 349.12, 350.55,
351.34, 350.80, 349.10, 347.54, 346.20, 346.20, 347.44, 348.67]

df = pd.DataFrame({'CO2':co2}, index=pd.date_range(
     start='09-01-2023', periods=len(co2), freq='W-MON'))
print(df)



#
#
# #TODO: 임시, TEMP로 pdr =>yfinance
import yfinance as yf
import pandas as pd
import datetime

# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
#
def getStock(stk, ttlDays):
    numDays = int(ttlDays)
    # Only gets up until day before during
    # trading hours
    dt = datetime.date.today()
    # For some reason, must add 1 day to get current stock prices
    # during trade hours. (Prices are about 15 min behind actual prices.)
    dtNow = dt + datetime.timedelta(days=1)
    dtNowStr = dtNow.strftime("%Y-%m-%d")
    dtPast = dt + datetime.timedelta(days=-numDays)
    dtPastStr = dtPast.strftime("%Y-%m-%d")

    # Use yfinance to directly fetch data
    df = yf.download(stk, start=dtPastStr, end=dtNowStr)
    return df

#
# # # Canadian stocks have the suffix
# # # .TO, .V or .CN for Canadian markets.
# # NUM_DAYS = 10
# # df = getStock('TD.TO', NUM_DAYS)
# # print("Toronto Dominion bank stock")
# # print(df)
#
#
# #TODO:시계열 표현
#
#
# today = datetime.date.today()
# start_date = datetime.date(2022, 1, 1)
# NUM_DAYS = (today - start_date).days
#
# df       = getStock('AMZN', NUM_DAYS)
# print("AMZN")
# print(df)
#
import matplotlib.pyplot as plt
#
#
def showStock(df, title):
    plt.plot(df.index, df['Close'])
    plt.title(title)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()
# showStock(df, "Amazon Close Prices")
#




#TODO: SUMMARY DATAFRAME

# Get Southwestern stock for last 60 days
NUM_DAYS = 1200
df       = getStock('LUV', NUM_DAYS)
print("Southwest Airlines")
print(df)

# Create weekly summary of closing price standard deviations
from pandas.tseries.frequencies import to_offset
series = df['Close'].resample('BM').mean()
series.index = series.index + to_offset("1M")
summaryDf = series.to_frame()

# Convert datetime index to date and then graph it.
summaryDf.index = summaryDf.index.date
print(summaryDf)
showStock(summaryDf, "Monthly Avg. Southwest Airlines")


#TODO: rescaled using a percent change.
import matplotlib.pyplot as plt
def showStocks(df, stock, title):
    plt.plot(df.index, df['Close'], label=stock)
    plt.xticks(rotation=70)

NUM_DAYS = 20
df = getStock('AMZN', NUM_DAYS)
df['Close'] = df['Close'].pct_change()
showStocks(df,'AMZN',"AMZN Close Prices")

df = getStock('AAPL', NUM_DAYS)
df['Close'] = df['Close'].pct_change()
showStocks(df, 'AAPL', "AAPL Close Prices")

df = getStock('MSFT', NUM_DAYS)
df['Close'] = df['Close'].pct_change()
showStocks(df, 'MSFT', "MSFT Close Prices")
# Make graphs appear.
plt.legend()
plt.show()





import pandas as pd

co2 = [342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27]
df  = pd.DataFrame({'CO2':co2}, index=pd.date_range('09-01-2022',
                                periods=len(co2), freq='B'))
df['CO2_t-1'] = df['CO2'].shift(periods=1)
df['CO2_t-2'] = df['CO2'].shift(periods=2)
df  = df.dropna()
print(df)
