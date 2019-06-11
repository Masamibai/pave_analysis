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

    path_from_1 = "year_changed.csv"
    list_1 = []
    flag = 0
    with open(path_from_1, "r") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            for i in range(len(line)):
                if flag != 0:
                    str_1 = str(line[0]) + str(line[2])
                    list_1.append(str_1)
                flag += 1

    list_1 = list(set(list_1))
    year_list = []
    year = 0

    for i in range(len(list_1)):
        print('doing', i/len(list_1))
        with open(path_from_1, "r") as f:
            csv_read = csv.reader(f)
            for line in csv_read:
                if str(line[0]) + str(line[2]) == list_1[i]:
                    year_list.append(int(int(line[1].split('/')[2]) - year))
                    year = int(line[1].split('/')[2])
    print(year_list)

    # 删除100以上的

    # 统计2和2以上的

    # 可以排序雨后绘制个数


















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