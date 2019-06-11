import csv, os, sys
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

# 统计个数 MEAN_TEMP_AVG_LAST_

def readthecsv():

    # list_1 = [1.802, 1.885, 1.86, 1.826, 1.851, 1.863, 1.874, 1.959, 1.842, 1.864, 1.985, 1.87, 1.872]
    #
    # list_2 = [1.934, 1.928]
    #
    # list_3 = [1.881]
    #
    # list_4 = [2.012, 1.989]
    #
    #
    #
    #
    #
    # x_1 = [1989, 1990, 1991, 1992, 1993, 1994, 1998, 1999, 2000, 2000, 2001, 2002, 2004]
    # x_2 = [2009, 2011]
    # x_3 = [2012]
    # x_4 = [2015, 2016]


    list_1 = [1.08, 1.042, 1.082, 1.02, 1.019, 1.036, 1.065, 1.032, 1.037, 1.017]

    list_2 = [0.905]

    list_3 = [0.815, 0.918, 1.004]

    list_4 = [0.759]





    x_1 = [1989.75, 1991.5, 1992.75, 1993.75, 1997.5, 1998.5, 1999.5, 2000.5, 2000.75, 2001.25]
    x_2 = [2001.5]
    x_3 = [2002.75, 2009.5, 2011.25]
    x_4 = [2015.75]

    middle_1_2_x = [2001.25, 2001.5]
    middle_1_2_y = [1.017, 0.905]

    middle_2_3_x = [2001.5, 2002.75]
    middle_2_3_y = [0.905, 0.815]

    middle_3_4_x = [2011.25, 2015.75]
    middle_3_4_y = [1.004, 0.759]

    list_1_1 = [1.08, 1.042, 1.082, 1.02, 1.019, 1.036, 1.065, 1.032, 1.037, 1.017]
    x_1_1= [1989.75, 1991.5, 1992.75, 1993.75, 1997.5, 1998.5, 1999.5, 2000.5, 2000.75, 2001.25]
    list_3_3 = [0.815, 0.918, 1.004]
    x_3_3 = [2002.75, 2009.5, 2011.25]



    plt.figure()
    # plt.plot(x_1, y_1, color='red', label='y=x')

    plt.plot(x_1_1, list_1_1, 'o-', color='blue')
    plt.plot(x_3_3, list_3_3, 'o-', color='green')

    plt.scatter(x_1, list_1, label='Construction=1', color='blue')
    plt.scatter(x_2, list_2, label='Construction=2', color='red')
    plt.scatter(x_3, list_3, label='Construction=3', color='green')
    plt.scatter(x_4, list_4, label='Construction=4', color='orange')

    plt.plot(middle_1_2_x, middle_1_2_y, '--', color='grey')
    plt.plot(middle_2_3_x, middle_2_3_y, '--', color='grey')
    plt.plot(middle_3_4_x, middle_3_4_y, '--', color='grey')
    plt.xlabel("Time")
    plt.ylabel("IRI")

    plt.legend()

    plt.show()



if __name__ == '__main__':
    readthecsv()