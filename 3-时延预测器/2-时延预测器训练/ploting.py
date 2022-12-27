# -*- coding: utf-8 -*-
# @FunctionName: ploting
# @Author: wanghongli
# @Time: 2022/3/9 21:28
# Use datetime for creating date objects for plotting
import datetime
import pandas as pd
from matplotlib import pyplot as plt


def ploting(features, test_features, feature_list, labels, predictions):
    indexs = features[:, feature_list.index('index')]
    print(indexs)
    # Dates of training values
    true_data = pd.DataFrame(data={'index': indexs, 'actual': labels})
    # Dataframe with predictions and dates
    indexs = test_features[:, feature_list.index('index')]
    predictions_data = pd.DataFrame(data={'index': indexs, 'prediction': predictions})
    plt.plot(true_data['index'], true_data['actual'], 'bo', label='actual')
    plt.plot(predictions_data['index'], predictions_data['prediction'], 'ro', label='prediction', linewidth=2)
    plt.xticks(rotation='60');
    plt.legend()
    # Graph labels
    plt.xlabel('Index'); plt.ylabel('Excution Latency (s)'); plt.title('Actual and Predicted Values')
    plt.show()
