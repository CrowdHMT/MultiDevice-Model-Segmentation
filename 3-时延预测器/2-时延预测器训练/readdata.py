# -*- coding: utf-8 -*-
# @FunctionName: readdata
# @Author: wanghongli
# @Time: 2022/3/10 11:18


# Pandas is used for data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = []
feature_list = []
'''
train_features = []
test_features = []
train_labels = []
test_labels = []
'''


# 1.数据采集
def readdata():
    global features
    # Read in data and display first 5 rows
    # features = pd.read_csv('conv_example_raspberry.csv')
    features = pd.read_csv('conv_example.csv')

    print(features.head(5))
    print('The shape of our features is:', features.shape)
    # print(features.describe()) 抽象描述
    # print(np.isnan(features).any())

    # 2.数据预处理
    # One-hot encode the data using pandas get_dummies  变为热编码
    features = pd.get_dummies(features, dtype=float)
    # Display the first 5 rows of the last 12 columns
    print(features.iloc[:, 0:].head(5))

    global feature_list

    # Labels are the values we want to predict 预测时延并去除特征
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
    return features, labels, feature_list, train_features, train_labels, test_features, test_labels


# 1.数据采集
def readdata_flops():
    global features
    # Read in data and display first 5 rows
    # features = pd.read_csv('conv_example_raspberry.csv')
    features = pd.read_csv('conv_example_flops.csv')

    print(features.head(2))
    print('The shape of our features is:', features.shape)
    # print(features.describe()) 抽象描述
    # print(np.isnan(features).any())

    # 2.数据预处理
    # One-hot encode the data using pandas get_dummies  变为热编码
    features = pd.get_dummies(features, dtype=float)
    # Display the first 5 rows of the last 12 columns
    print(features.iloc[:, 0:].head(2))

    global feature_list

    # Labels are the values we want to predict 预测时延并去除特征
    labels = np.array(features['latency'])
    features = features.drop('latency', axis=1)
    feature_list = list(features.columns)  # 特征的名字
    # Convert to numpy array 转化为 np数组格式
    features = np.array(features)
    # random代表如何取随机数
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    # 检查划分的数据集是否 正确
    print('Training Features Shape：', train_features.shape)
    print('Training Labels Shape：', train_labels.shape)
    print('Testing Features Shape：', test_features.shape)
    print('Testing Labels Shape：', test_labels.shape)
    return features, labels, feature_list, train_features, train_labels, test_features, test_labels
