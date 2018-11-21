# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午2:03
# @Author  : xuef
# @FileName: test1.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_42118777/article
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys


def calc_line_of_best_fit(data):
    x_av = np.mean(data['x'])
    y_av = np.mean(data['y'])

    slope = sum([(data['x'][i] - x_av) * (data['y'][i] - y_av) for i in range(len(data))]) \
            / \
            sum([x ** 2 - x_av ** 2 for x in data['x']])

    y_intercept = y_av - (slope * x_av)

    return slope, y_intercept


def plot_l_o_bf(data, slope, y_intercept):
    data.plot(kind='scatter', x='x', y='y', color='red')  # Plot data

    min_max_x = pd.DataFrame({'x': np.array([min(data['x']), max(data['x'])])})  # Bounds of line

    plt.plot(min_max_x, y_intercept + slope * min_max_x, c="blue", linestyle='-')  # Plot line


if __name__ == '__main__':
    # TODO - Make n dimensional based on data

    data = pd.DataFrame({'x': range(5, 15), 'y': [6, 8, 4, 6, 10, 11, 5, 8, 1, 5]})

    slope, y_intercept = calc_line_of_best_fit(data)

    print("f-best_fit(x) = {}x + {}".format(slope, y_intercept))
    plot_l_o_bf(data, slope, y_intercept)

    plt.show()
