# -*- coding: utf-8 -*-
# @FunctionName: Decision_tree
# @Author: wanghongli
# @Time: 2022/3/4 15:39


# Pandas is used for data manipulation
import pandas as pd
import csv
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import pydot
import os
from ploting import ploting

features = []
feature_list = []
rf = RandomForestRegressor(n_estimators=500, max_depth=30, n_jobs=-1)
'''
train_features = []
test_features = []
train_labels = []
test_labels = []
'''

# 1.数据采集
def readdata():
    global features
    # Read in data and display first 5 rows 读数据
    features = pd.read_csv('conv_example.csv')
    print(features.head(5))
    print('The shape of our features is:', features.shape)
    # print(features.describe()) 抽象描述
    # print(np.isnan(features).any())

# 2.数据预处理
def preprocess():
    global features
    # One-hot encode the data using pandas get_dummies  变为热编码
    features = pd.get_dummies(features)
    # Display the first 5 rows of the last 12 columns
    print(features.iloc[:, 0:].head(5))

# 3.训练决策树
def trainDecisionTree():
    global features
    global feature_list
    global rf
    # global train_features
    # global test_features
    # global train_labels
    # global test_labels

    # Labels are the values we want to predict 选择时延为待预测时延并去除特征
    labels = np.array(features['latency'])
    features = features.drop('latency', axis=1)
    feature_list = list(features.columns)  # 特征的名字
    # Convert to numpy array 转化为 np数组格式
    features = np.array(features)
    # random代表如何取随机数
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)
    # 检查划分的数据集是否 正确
    print('Training Features Shape：', train_features.shape)
    print('Training Labels Shape：', train_labels.shape)
    print('Testing Features Shape：', test_features.shape)
    print('Testing Labels Shape：', test_labels.shape)
    # 取平均值作为基线的预测值
    # baseline_preds = []
    # for i in range(21):
    #     baseline_preds.append(0.03)

    # baseline_errors = abs(baseline_preds - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))
    # rf = RandomForestRegressor(n_estimators=200, random_state=42)
    # 创建随机森林模型 50颗决策树组成的随机森林 使用全局的决策树模型
    rf.fit(train_features, train_labels)  # 决策森林训练模型

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors 计算损失
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)   MAE指标
    print('Mean Absolute Error:', round(np.mean(errors), 6), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)  计算百分比的精度
    # print(errors, test_labels.shape)
    # mape = 100 * (errors / test_labels)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')

    # rf.sccore
    print("*" * 30 + " 准确率 " + "*" * 30)
    # train_score 和 test_score
    print(rf.score(test_features, test_labels))
    print(rf.score(train_features, train_labels))
    # ploting(features, test_features, feature_list, labels, predictions)
    prediction_all = rf.predict(features)
    write_data(features, feature_list, labels, prediction_all)
    #
    pp = predictions.reshape(-1)
    error = []
    for i in range(len(test_labels)):
        error.append(test_labels[i] - pp[i])
    squaredError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
    print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE

# 4.超参数的优选
def parameter_choose():
    global features
    # 网格搜索
    # Labels are the values we want to predict 预测时延并去除特征
    labels = np.array(features['latency'])
    features = features.drop('latency', axis=1)
    # Convert to numpy array 转化为 np数组格式
    features = np.array(features)
    # random代表如何取随机数
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=1234)
    print('The shape of our features is:', train_features.shape)
    print('The shape of our features is:', test_features.shape)
    print('The shape of our features is:', train_labels.shape)
    print('The shape of our features is:', test_labels.shape)
    # n_estimators: 决策树数目
    # max_depth: 树最大深度  识别树
    rf = RandomForestRegressor(n_jobs=-1)
    param = {
        "n_estimators": [500],
        "max_depth": [30]
    }
    # 2折交叉验证
    search = GridSearchCV(rf, param_grid=param, cv=2)
    print("*" * 30 + " 超参数网格搜索 " + "*" * 30)
    start_time = time.time()
    search.fit(train_features, train_labels)
    print("耗时：{}".format(time.time() - start_time))
    print("最优参数：{}".format(search.best_params_))

    print("*" * 30 + " 准确率 " + "*" * 30)
    print(search.score(test_features, test_labels))


# 5.可视化决策树
def visual_decisiontree():
    # 由森林中抽取一棵树
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
    # 由dot创建图
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')


# 6.决策树中各标签的重要性
def featuresimportance():
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    for pair in feature_importances:
        print('Variable: {:20} Importance: {}'.format(*pair))

def write_data(features, feature_list, labels,predictions_all):
    f = open('conv_prediction_actual.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    headers = ['index', 'prediction', 'actual']
    csv_writer.writerow(headers)
    indexs = features[:, feature_list.index("index")]
    print(len(indexs),len(predictions_all), len(labels))
    for i in range(len(indexs)):
        rows = [indexs[i], predictions_all[i], labels[i]]
        csv_writer.writerow(rows)
    f.close()


if __name__ == "__main__":
    readdata()
    preprocess()
    #parameter_choose() # 超参数选择
    trainDecisionTree()
    #visual_decisiontree()
    # featuresimportance()