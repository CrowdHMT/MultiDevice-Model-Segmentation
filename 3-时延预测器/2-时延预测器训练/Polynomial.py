# -*- coding: utf-8 -*-
# @FunctionName: Polynomial
# @Author: wanghongli
# @Time: 2022/3/10 11:14
#!/usr/bin/env python -W ignore::DeprecationWarning

import matplotlib.pyplot as plt
import numpy as np
from readdata import readdata
from readdata import readdata_flops
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from ploting import ploting
from sklearn.metrics import r2_score
import csv

def stdError_func(y_test, y):
  return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
  return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
  y_mean = np.array(y)
  y_mean[:] = y.mean()
  return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)


def polynomial():
    features, labels, feature_list, train_features, train_labels, test_features, test_labels = readdata()   # 正常预测
    # features, labels, feature_list, train_features, train_labels, test_features, test_labels = readdata_flops()
    print(test_features.shape)
    # cft = linear_model.LinearRegression()
    cft = PolynomialFeatures(degree=3)  # 几次多项式
    X_ploy = cft.fit_transform(train_features)
    lin_reg_2 = linear_model.LinearRegression() # 线性拟合？
    lin_reg_2.fit(X_ploy, train_labels)
    # cft.fit(train_features, train_labels)
    #print("model coefficients", cft.coef_)
    #print("model intercept", cft.intercept_)
    # predictions = cft.predict(test_features)
    X_test = cft.fit_transform(test_features)
    predictions = lin_reg_2.predict(X_test)
    strError = stdError_func(predictions, test_labels)
    #R2_1 = R2_1_func(predictions, test_labels)
    #R2_2 = R2_2_func(predictions, test_labels)
    score_train = r2_score(lin_reg_2.predict(cft.fit_transform(train_features)), train_labels);
    R_test = r2_score(predictions, test_labels)
    #score_train = cft.score(train_features, train_labels)  ##sklearn中自带的模型评估，与R2_1逻辑相同
    #score_test = cft.score(test_features, test_labels)

    print('strError={:.5f}, R2_train={:.5f}, R2_test={:.5f}'.format(
        strError, score_train, R_test))
    # feature_list = list(features.columns)  # 特征的名字
    #
    # Calculate the absolute errors 计算损失
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)   MAE
    print('Mean Absolute Error:', round(np.mean(errors), 6), 'degrees.')


    ploting(features, test_features, feature_list, labels, predictions)
    ## 为了画图
    prediction_all = lin_reg_2.predict(cft.fit_transform(features))
    write_data(features, feature_list, labels, prediction_all)
    pp = predictions.reshape(-1)
    error = []
    for i in range(len(test_labels)):
        error.append(test_labels[i] - pp[i])
    squaredError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
    print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE


def write_data(features, feature_list, labels,predictions_all):
    f = open('conv_prediction_actual_polynomial.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    headers = ['index', 'prediction', 'actual']
    csv_writer.writerow(headers)
    indexs = features[:, feature_list.index("index")]
    print(len(indexs),len(predictions_all), len(labels))
    for i in range(len(indexs)):
        rows = [indexs[i], predictions_all[i], labels[i]]
        csv_writer.writerow(rows)
    f.close()

if __name__ == '__main__':
    polynomial()