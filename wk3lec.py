# # dataframe opertations - pandas
# import pandas as pd
# # plotting data - matplotlib
# from matplotlib import pyplot as plt
# # time series - statsmodels
# # Seasonality decomposition
# from statsmodels.tsa.seasonal import seasonal_decompose
# # holt winters
# # single exponential smoothing
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# # double and triple exponential smoothing
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
#
#
#
# path = 'data/DailyDelhiClimateTest.csv'
# DelhiTemp = pd.read_csv(path, index_col='date', parse_dates=True)
# # finding shape of the dataframe
# print(DelhiTemp.shape)
# # having a look at the data
# print(DelhiTemp.head())
# # plotting the original data
#
# # DelhiTemp.index_freq = 'D'
# # Split into train and test set.
# train_Delhi    = DelhiTemp[:90]
# test_Delhi     = DelhiTemp[90:]
#
#
#
#
#
# # Build HWES3 model with multiplicative decomposition.
# fitted_model     = ExponentialSmoothing(train_Delhi['meantemp'],
#                    trend='mul',seasonal='mul',seasonal_periods=12).fit()
# test_predictions = fitted_model.forecast(24)
# # Plot raw train, test and predictions.
# train_Delhi['meantemp'].plot(legend=True,label='TRAIN')
# test_Delhi['meantemp'].plot(legend=True,label='TEST',figsize=(6,4))
# test_predictions.plot(legend=True,label='PREDICTION')
# plt.title('Train, Test and Predicted Test using Holt Winters')
# plt.show()
#
#
#
#





import matplotlib.pyplot as plt
import statsmodels.api as sm

dta = sm.datasets.sunspots.load_pandas().data
print(dta)
# dta['SUNACTIVITY_t-1'] = (dta['SUNACTIVITY'].shift(1))
# print(dta)
# plt.plot(dta['YEAR'], dta['SUNACTIVITY'])
# plt.show()





#
#
# import matplotlib.pyplot as plt
# import pandas            as pd
# import numpy             as np
# import statsmodels.api   as sm
# from sklearn.preprocessing import MinMaxScaler
#
# # Show all columns.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
#
# # Do not show warning.
# pd.options.mode.chained_assignment = None
#
#
#
# # Create back-shifted columns for an attribute.
# def addBackShiftedColumns(df, colName, timeLags):
#       for i in range(1, timeLags+1):
#             newColName = colName + "_t-" + str(i)
#             df[newColName] = df[colName].shift(i)
#       return df
#
# # Build dataframe for modelling.
# columns = ['SUNACTIVITY']
# modelDf = dta.copy()
# NUM_TIME_STEPS = 1
# for i in range(0, len(columns)):
#       modelDf = addBackShiftedColumns(modelDf, columns[i],
#                                       NUM_TIME_STEPS)
# modelDf = modelDf.dropna()
# y = modelDf[['SUNACTIVITY']]
# X = modelDf[[ 'SUNACTIVITY_t-1']]
#
# # Add intercept for OLS regression.
# X         = sm.add_constant(X)
# TEST_DAYS = 10
#
# # Split into test and train sets. The test data includes
# # the latest values in the data.
# lenData = len(X)
# X_train = X[0:lenData-TEST_DAYS]
# y_train = y[0:lenData-TEST_DAYS]
# X_test  = X[lenData-TEST_DAYS:]
# y_test  = y[lenData-TEST_DAYS:]
#
# # Model and make predictions.
# model       = sm.OLS(y_train, X_train).fit()
# print(model.summary())
# predictions = model.predict(X_test)
#
#
#
# # Show RMSE.
# from sklearn  import metrics
# print('Root Mean Squared Error:',
#       np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#
# # Plot the data.
# xaxisValues = list(y_test.index)
# plt.plot(xaxisValues, y_test, label='Actual', marker='o')
# plt.plot(xaxisValues, predictions, label='Predicted', marker='o')
# plt.xticks(rotation=45)
# plt.legend(loc='best')
# plt.title("Mean temperature in Dehli")
# plt.show()
#
#
#
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(dta['SUNACTIVITY'], lags=50)  #50정도가 보기 나쁘지 않을듯. last 15 samples 해보든가 ㅎㅎ
# plt.show()
#
# # Show partial-autocorrelation function.
# # Shows correlation of 1st lag with past lags.
# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(dta['SUNACTIVITY'], lags=50)
# plt.show()
#
# print(dta)



# #EXercise 8
# import  pandas as pd
# import  matplotlib.pyplot as plt
# import  statsmodels.api as sm
# from    statsmodels.tsa.arima.model import ARIMA
# from    sklearn.metrics import mean_squared_error
# import  numpy as np
# import warnings
# warnings.filterwarnings("ignore")
#
# dta       = sm.datasets.sunspots.load_pandas().data
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
#
# # Show autocorrelation function.
# # General correlation of lags with past lags.
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(dta['SUNACTIVITY'], lags=50)
# plt.show()
#
# # Split the data.
# NUM_TEST_YEARS = 10
# lenData        = len(dta)
# dfTrain        = dta.iloc[0:lenData - NUM_TEST_YEARS, :]
# dfTest         = dta.iloc[lenData-NUM_TEST_YEARS:,:]
#
# def buildModelAndMakePredictions(AR_time_steps, dfTrain, dfTest):
#     # This week we will use the ARIMA model.
#
#     model  = ARIMA(dfTrain['SUNACTIVITY'], order=(AR_time_steps, 0, 0),
#                    freq='Y').fit()
#     print("\n*** Evaluating ARMA(" + str(AR_time_steps) + ",0,0)")
#     print('Coefficients: %s' %model.params)
#
#     # Strings which can be converted to time stamps are passed in.
#     # For this case the entire time range for the test set is represented.
#     predictions = model.predict('1999-12-31', '2008-12-31', dynamic=True)
#     rmse = np.sqrt(mean_squared_error(dfTest['SUNACTIVITY'].values,
#                                       np.array(predictions)))
#     print('Test RMSE: %.3f' % rmse)
#     print('Model AIC %.3f' % model.aic)
#     print('Model BIC %.3f' % model.bic)
#     return model, predictions
#
# print(dfTest)
# arma_mod20, predictionsARMA_20 = buildModelAndMakePredictions(2, dfTrain, dfTest)
# arma_mod30, predictionsARMA_30 = buildModelAndMakePredictions(3, dfTrain, dfTest)
# arma_mod90, predictionsARMA_90 = buildModelAndMakePredictions(9, dfTrain, dfTest)
# plt.plot(dfTest.index, dfTest['SUNACTIVITY'],
#          label='Actual Values', color='blue')
# plt.plot(dfTest.index, predictionsARMA_30,
#          label='Predicted Values AR(30)', color='brown')
# plt.legend(loc='best')
# # plt.show()
#
#
#
# plt.plot(dfTest.index, predictionsARMA_20,
#          label='Predicted Values AR(20)', color='orange')
# plt.legend(loc='best')
# # plt.show()
#
#
# plt.plot(dfTest.index, predictionsARMA_90,
#          label='Predicted Values AR(90)', color='red')
# plt.legend(loc='best')
# plt.show()
#
# print("\n*** Comparing AR Model Time Steps")
# import warnings
# warnings.filterwarnings("ignore")
# res = sm.tsa.arma_order_select_ic(dta['SUNACTIVITY'], max_ar=10, max_ma=0, ic='bic')
# print(res.values())
#
#


# #Exercise10

from pandas                     import read_csv
import matplotlib.pyplot        as plt
from statsmodels.tsa.ar_model   import AutoReg
from sklearn.metrics            import mean_squared_error
from   math                     import sqrt
import warnings
warnings.filterwarnings("ignore")

# Load the data.
PATH = "data/"
series = read_csv(PATH + 'daily-min-temperatures.csv',
                  header=0, index_col=0, parse_dates=True)

# Plot ACF.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=20)
plt.show()

# Plot PACF.
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series, lags=20)
plt.show()



NUM_TEST_DAYS = 7

# Split dataset into test and train.
X       = series.values
lenData = len(X)
train   = X[0:lenData-NUM_TEST_DAYS]
test    = X[lenData-NUM_TEST_DAYS:]

# Train.
model     = AutoReg(train, lags=7)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

print(model_fit.summary())   #약간 pacf 결과랑 유사해보이긴 함.

# Make predictions. 구간은 노가다로 알아서 잘..
predictions = model_fit.predict(start=len(train),
                                end=len(train)+len(test)-1,
                                dynamic=False)

for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot results. 걍 직선으로 나오네..
plt.plot(test, marker='o', label='actual')
plt.plot(predictions, color='brown', linewidth=4,
         marker='o', label='predicted')

plt.legend()
plt.show()


#Exercise11

# Use model coefficients from autoregression to make a prediction.
def makePrediction(t_1, t_2, t_3, t_4, t_5, t_6, t_7):
    intercept =   1.1142
    t1Coeff   =   0.6263
    t2Coeff   = -0.0751
    t3Coeff   =    0.0741
    t4Coeff   =   0.0617
    t5Coeff   =  0.0660
    t6Coeff   =   0.0442
    t7Coeff   =   0.1030

    prediction = intercept + t1Coeff*t_1 + t2Coeff*t_2 + t3Coeff*t_3 + t4Coeff*t_4 + t5Coeff*t_5 + t6Coeff*t_6 + t7Coeff*t_7

    return prediction

testLen    = len(test)

t_1 = test[testLen-1]
t_2 = test[testLen-2]
t_3 = test[testLen-3]
t_4 = test[testLen-4]
t_5 = test[testLen-5]
t_6 = test[testLen-6]
t_7  = test[testLen-7]

futurePredictions = []
for i in range(0, NUM_TEST_DAYS):
    prediction = makePrediction(t_1, t_2, t_3, t_4, t_5, t_6, t_7)
    futurePredictions.append(prediction)
    t_7 = t_6
    t_6 = t_5
    t_5 = t_4
    t_4 = t_3
    t_3 = t_2
    t_2 = t_1
    t_1 = prediction

print("Here is a one week temperature forecast: ")
# print(futurePredictions)
formatted_result = [f"{arr[0]:.6f}" for arr in futurePredictions]
print(formatted_result)