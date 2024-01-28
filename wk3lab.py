# import warnings
#
# warnings.filterwarnings("ignore")
#
# import numpy as np
# from scipy import stats
# import pandas as pd
# import matplotlib.pyplot as plt
#
# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA
#
# dta = sm.datasets.sunspots.load_pandas().data
#
# import datetime
# from pandas_datareader import data as pdr
# import yfinance as yfin  # Work around until
#
#
# # pandas_datareader is fixed.
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
# stkName = 'MSFT'
# dfStock = getStock(stkName, 400)
#
# # Split the data.
# NUM_TEST_DAYS = 5
# lenData = len(dfStock)
# dfTrain = dfStock.iloc[0:lenData - NUM_TEST_DAYS, :]
# dfTest = dfStock.iloc[lenData - NUM_TEST_DAYS:, :]
#
# plt.plot(dfStock.index, dfStock['Open'])
# plt.show()
#
#
#
#
#
#
# import warnings
# warnings.filterwarnings("ignore")
#
# import statsmodels.api as sm
# import statsmodels.tsa.arima.model as sma
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# import numpy as np
#
#
#
# def buildModel(df, ar, i, ma):
#     model = sma.ARIMA(df['Open'], order=(ar, i, ma)).fit()  #ARMIA 모델에 어떤 걸 줄지. Ar, ma는 아까 위에서 본 p,q 밸류라고 함. P-value랑 혼동 주의.
#     return model
#
# def predictAndEvaluate(model, test, title):
#     print("\n***" + title)
#     print(model.summary())
#     start       = len(dfTrain)
#     end         = start + len(dfTest) -1
#
#     predictions = model.predict(start=start, end=end, dynamic=True)   #일자.
#     mse = mean_squared_error(predictions, test['Open'])
#
#     rmse = np.sqrt(mse)
#     print("RMSE: " + str(rmse))
#     return rmse, predictions
#
# train, test = dfTrain, dfTest
#
# def showPredictedAndActual(actual, predictions, ar, ma):
#     indicies = list(actual.index)
#     plt.title("AR: " + str(ar) + " MA: " + str(ma))
#     plt.plot(indicies, predictions, label='predictions', marker='o')
#     plt.plot(indicies, actual, label = 'actual', marker='o')
#     plt.legend()
#     plt.xticks(rotation=70)
#     plt.tight_layout()
#     plt.show()
#
# modelStats = []
# for ar in range(0, 5):
#     for ma in range(0, 5):
#         model = buildModel(train, ar, 0, ma)
#         title = str(ar) + "_0_" + str(ma)
#         rmse, predictions = predictAndEvaluate(model, test, title)
#         if(ar==3 and ma==2):  #제일 좋은 모델만 표시함. (0?, 0)
#             showPredictedAndActual(test['Open'], predictions, ar, ma)
#         modelStats.append({"ar":ar, "ma":ma, "rmse":rmse})   #디버깅해보셈ㅎㅎ
#
# dfSolutions = pd.DataFrame(data=modelStats)
# dfSolutions = dfSolutions.sort_values(by=['rmse'])
# print(dfSolutions)





#WALK FORWARD EXAMPLE
# from pandas import read_csv
# import matplotlib.pyplot as plt
# import statsmodels.tsa.arima.model as sma
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# PATH = "data/"
# df = read_csv(PATH + 'daily-min-temperatures.csv', header=0, index_col=0)
#
# TEST_DAYS = 5
#
# X_train = df[0:len(df)-TEST_DAYS]
# y_train = df[0:len(df)-TEST_DAYS]
# X_test  = df[len(df)-TEST_DAYS:]
# y_test  = df[len(df)-TEST_DAYS:]
#
# # Create a list with the training array.
# predictions = []  #디버거 걸어보셈.
#
# for i in range(len(X_train)):
#     print("History length: " + str(len(X_train)))
#
#     #################################################################
#     # Model building and prediction section.
#     model       = sma.ARIMA(X_train, order=(1, 0, 0)).fit()
#     yhat        = model.predict(start=len(X_train), end=len(X_train))  #index of next things(5).. 3645개의row가 있으니.
#
#     if(i<len(X_test)):   #이렇게 계속 new rows of data from test_data and put it into train each iteration)
#         test_row = X_test.iloc[i]
#         X_train = X_train._append(test_row, ignore_index=True )
#         predictions.append(yhat.iloc[0])     #즉 X_test에서 끌고와서 X_train에 계속 얹고 있는 거임. 최신 자료 얹기 ㅋㅋ
#     else:
#         break
#
#     #################################################################
#
# rmse = sqrt(mean_squared_error(X_test, predictions))
# print('Test RMSE: %.3f' % rmse)
#
# indices = list(X_test.index)
# plt.plot(indices, X_test, label='Actual', marker='o', color='blue')
# plt.plot(indices, predictions, label='Predictions', marker='o', color='orange')
# plt.legend()
# plt.title("AR Model")
# plt.xticks(rotation=70)
# plt.tight_layout()
# plt.show()



#ADF
import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data
df = pd.read_csv(
"https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv", \
                  names=['value'], header=0)
print(df)
df.value.plot()
plt.title("www usage")
plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


#differentiate it
dfDifferenced  = df.diff()
plt.plot(dfDifferenced )
plt.xticks(rotation=75)
plt.show()



from statsmodels.tsa.stattools import adfuller
result = adfuller(dfDifferenced.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])





#differentiate it
dfDifferenced  = dfDifferenced.diff()
plt.plot(dfDifferenced )
plt.xticks(rotation=75)
plt.show()



from statsmodels.tsa.stattools import adfuller
result = adfuller(dfDifferenced.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

