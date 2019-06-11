import csv, os, sys
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

# 统计个数 MEAN_TEMP_AVG_LAST_

def readthecsv():
    # path_from_1 = "final_version_structure.csv"
    # list_1 = []
    # with open(path_from_1, "r") as f:
    #     csv_read = csv.reader(f)
    #     for line in csv_read:
    #         list_1.append(line[0] + line[1])
    # print(len(list_1))
    # list_1 = list(set(list_1))
    # print(len(list_1))

    path_from_1 = "final_version_structure.csv"
    list_1 = []
    with open(path_from_1, "r") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            for i in range(len(line)):
                if 'WET_DAYS_LAST_' in line[i]:
                    list_1.append(i)
            break
    print(len(list_1))
    list_1 = list(set(list_1))
    print(len(list_1))
    dataset = read_csv("final_version_structure.csv")
    dataset.dropna(axis=0, how='any', inplace=True)
    using = dataset[list_1]
    # print(using.values)
    list_2 = []
    length = len(using.values)
    sum = 0
    for i in range(length):
        print('doing', i/length)
        for j in range(len(list_1)):
            num = float(using.values[i][j])
            if num != -100:
                list_2.append(num)
                sum += num
                # print(sum, '- - -', num)
    list_2.sort()
    print(list_2[0])
    print(list_2[-1])
    # print(list_2)
    print(np.mean(list_2))

    # x_1 = [x/100 for x in range(len(list_2))]
    # y_1 = [x/100 for x in range(1, 500)]
    #
    # plt.figure()
    # # plt.plot(x_1, y_1, color='red', label='y=x')
    # plt.plot(x_1, list_2, "-")
    # plt.xlabel("Real value")
    # plt.ylabel("Predicted value")
    #
    # plt.show()



if __name__ == '__main__':
    readthecsv()