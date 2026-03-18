import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# All code was run in a jupyter notebook


#reports variaous metrics and values of y predictions and true y values
def regression_metrics(y_true, y_pred, shift, alphaval, ratio):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero by adding a tiny number)
    eps = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE(%)": mape, "Minutes Shifted": shift*5, "Alpha Value": alphaval, "L1 Ratio": ratio}

#aggregates speed, volume, volume per lane, and occupancy by hour
def byHour(df):

  df["DateTime"] = pd.to_datetime(df["DateTime"])

  df['Hour'] = df['DateTime'].apply(lambda n: n.hour)
  df['Date'] = df['DateTime'].apply(lambda n: n.date())
  dates = df['Date'].unique()

  res = []
  for d in dates:
    temp = df.where(df['Date'] == d).groupby(['Hour'])[['Speed',
                                                        'Volume',
                                                        'Volume Per Lane',
                                                        'Occupancy']].mean().reset_index()
    temp['Date'] = d
    temp['DateTime'] = temp.apply(lambda row:
                                  pd.to_datetime(f"{row['Date']}" + " " +
                                  f"{dt.time(hour=int(row['Hour'])).strftime('%H:%M:%S')}"), axis = 1)
    res.append(temp)

  return res

#takes in two dataframes (formatted identically, with columns of DateTime (assumed to be seoparated by 5 minute increments),
#Speed, Volume, Volume Per Lane, and Occupancy) and a integer number

#returns first of the two dataframes with the Speed, Volume, Volume Per Lane, and Occupancy datafields of the second dataframe 
#appendend to it and shifted by the number passed in as SpeedLag1, VolLag1, VPLLag1, OccLag1
def timelag(df1, df2, shiftn):

  #make sure DateTime fields are in datetime format
  df1["DateTime"] = pd.to_datetime(df1["DateTime"])
  df2["DateTime"] = pd.to_datetime(df2["DateTime"])

  #add speed, volume, volume per lane, and occupancy data from second dataframe to first
  #shift by shiftn rows
  df1[['SpeedLag1',
       'VolLag1',
       'VPLLag1',
       'OccLag1']] = df2[['Speed',
                          'Volume',
                          'Volume Per Lane',
                          'Occupancy']].shift(shiftn)
  df1 = df1.dropna().reset_index(drop=True)

  if 'Unnamed: 0' in df1.columns:
    df1 = df1.drop('Unnamed: 0', axis = 1)
  return df1

#ensures the dataframe passed in is sorted and there are no rows with missing values.
# returns dataframes for the training data, validation data, and test data based on a ration of 70:15:15
def dfSort(df):
  sorted = df.dropna().reset_index(drop=True)
  cutoff_train = sorted["DateTime"].quantile(0.70)
  cutoff_val= sorted["DateTime"].quantile(0.85)

  train = sorted[sorted["DateTime"] < cutoff_train]
  val   = sorted[(sorted["DateTime"] >= cutoff_train) & (sorted["DateTime"] < cutoff_val)]
  test  = sorted[sorted["DateTime"] >= cutoff_val]
  return sorted, train, val, test

#trains the model passed, takes training data, validation data, testing data, a model, the value to predict, 
#the number of rows to shift, an alpha value, and a L1 ratio
#outputs regression metrics if desired
def trainModel(traindat, valdat, testdat, model, pred_val, shiftn, alpha, ratio):
  xval = traindat[['SpeedLag1', 'VolLag1', 'VPLLag1', 'OccLag1']]
  #fits model to the xvalues
  model.fit(xval, traindat[pred_val].to_frame())

  y_val = valdat[pred_val].values
  y_test = testdat[pred_val].values

  pred_val = model.predict(valdat[['SpeedLag1', 'VolLag1', 'VPLLag1', 'OccLag1']])
  pred_test = model.predict(testdat[['SpeedLag1', 'VolLag1', 'VPLLag1', 'OccLag1']])

  lin_val = regression_metrics(y_val, pred_val, shiftn, alpha, ratio)
  lin_test = regression_metrics(y_test, pred_test, shiftn, alpha, ratio)

  res = pd.DataFrame([lin_val, lin_test], index=[f"{model} (Val)", f"{model} (Test)"])
  return res

filename_i5 = '005es16732_loop_cloutput.csv'
filename_520 = '520es00972_loop_cloutput.csv'

dfi5 = pd.read_csv(filename_i5)
df520 = pd.read_csv(filename_520)

#range of vrtime shifts representing 5 minute increments.
shiftns =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#range of alpha values for Lasso and ElasticNet models
alphavals = 0.1 #[0.1, 1, 5, 10, 20, 100, 200, 500]

#range of L1 ratios for ElasticNet model
l1ratios = 0.45 #[0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

spdfulllr = []
volfulllr = []
spdfulllas = []
volfulllas = []
spdfulleln = []
volfulleln = []

#Testing of diffrent time shifts

for i in range(len(shiftns)):
    lr = LinearRegression()
    lrlasso = Lasso(alpha=alphavals)
    lreln = ElasticNet(alpha=alphavals, l1_ratio = l1ratios)
    dftemp = timelag(dfi5.copy(), df520.copy(), shiftns[i])
    dftemp_sorted, trainm, valm, testm = dfSort(dftemp)
    resvollr = trainModel(trainm, valm, testm, lr, 'Volume', shiftns[i], alphavals, l1ratios)
    resspdlr = trainModel(trainm, valm, testm, lr, 'Speed', shiftns[i], alphavals, l1ratios)
    resvollas = trainModel(trainm, valm, testm, lrlasso, 'Volume', shiftns[i], alphavals, l1ratios)
    resspdlas = trainModel(trainm, valm, testm, lrlasso, 'Speed', shiftns[i], alphavals, l1ratios)
    resvoleln = trainModel(trainm, valm, testm, lreln, 'Volume', shiftns[i], alphavals, l1ratios)
    resspdeln = trainModel(trainm, valm, testm, lreln, 'Speed', shiftns[i], alphavals, l1ratios)
    volfulllr.append(resvollr)
    spdfulllr.append(resspdlr)
    volfulllas.append(resvollas)
    spdfulllas.append(resspdlas)
    volfulleln.append(resvoleln)
    spdfulleln.append(resspdeln)

#Testing of different alpha values

# for i in range(len(alphavals)):
#     lr = LinearRegression()
#     lrlasso = Lasso(alpha=alphavals[i])
#     lreln = ElasticNet(alpha=alphavals[i], l1_ratio = l1ratios)
#     dftemp = timelag(dfi5.copy(), df520.copy(), shiftns)
#     dftemp_sorted, trainm, valm, testm = dfSort(dftemp)
#     resvollr = trainModel(trainm, valm, testm, lr, 'Volume', shiftns, alphavals[i], l1ratios)
#     resspdlr = trainModel(trainm, valm, testm, lr, 'Speed', shiftns, alphavals[i], l1ratios)
#     resvollas = trainModel(trainm, valm, testm, lrlasso, 'Volume', shiftns, alphavals[i], l1ratios)
#     resspdlas = trainModel(trainm, valm, testm, lrlasso, 'Speed', shiftns, alphavals[i], l1ratios)
#     resvoleln = trainModel(trainm, valm, testm, lreln, 'Volume', shiftns, alphavals[i], l1ratios)
#     resspdeln = trainModel(trainm, valm, testm, lreln, 'Speed', shiftns, alphavals[i], l1ratios)
#     volfulllr.append(resvollr)
#     spdfulllr.append(resspdlr)
#     volfulllas.append(resvollas)
#     spdfulllas.append(resspdlas)
#     volfulleln.append(resvoleln)
#     spdfulleln.append(resspdeln)

#Testing of different L1 ratios

# for i in range(len(l1ratios)):
#     lr = LinearRegression()
#     lrlasso = Lasso(alpha=alphavals)
#     lreln = ElasticNet(alpha=alphavals, l1_ratio = l1ratios[i])
#     dftemp = timelag(dfi5.copy(), df520.copy(), shiftns)
#     dftemp_sorted, trainm, valm, testm = dfSort(dftemp)
#     resvollr = trainModel(trainm, valm, testm, lr, 'Volume', shiftns, alphavals, l1ratios[i])
#     resspdlr = trainModel(trainm, valm, testm, lr, 'Speed', shiftns, alphavals, l1ratios[i])
#     resvollas = trainModel(trainm, valm, testm, lrlasso, 'Volume', shiftns, alphavals, l1ratios[i])
#     resspdlas = trainModel(trainm, valm, testm, lrlasso, 'Speed', shiftns, alphavals, l1ratios[i])
#     resvoleln = trainModel(trainm, valm, testm, lreln, 'Volume', shiftns, alphavals, l1ratios[i])
#     resspdeln = trainModel(trainm, valm, testm, lreln, 'Speed', shiftns, alphavals, l1ratios[i])
#     volfulllr.append(resvollr)
#     spdfulllr.append(resspdlr)
#     volfulllas.append(resvollas)
#     spdfulllas.append(resspdlas)
#     volfulleln.append(resvoleln)
#     spdfulleln.append(resspdeln)

spdfinlr = pd.concat(spdfulllr)
volfinlr = pd.concat (volfulllr)
spdfinlas = pd.concat(spdfulllas)
volfinlas = pd.concat (volfulllas)
spdfineln = pd.concat(spdfulleln)
volfineln = pd.concat (volfulleln)

print('Linear Regression \nTraining to Predict Speed:\n')
display(spdfinlr)

print('Linear Regression \nTraining to Predict Volume:\n')
display(volfinlr)

print('Linear Regression w/ Lasso Regularization\nTraining to Predict Speed:\n')
display(spdfinlas)

print('Training to Predict Volume:\n')
display(volfinlas)