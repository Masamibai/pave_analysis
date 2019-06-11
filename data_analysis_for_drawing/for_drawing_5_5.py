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

    the_str_1 = 'MEAN_ANN_TEMP_AVG'
    the_str_2 = 'DAYS_ABOVE_32_C_YR'
    the_str_3 = 'TOTAL_ANN_PRECIP'

    path_from_1 = "temp_drawing_1.csv"
    list_1 = []
    with open(path_from_1, "r") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            for i in range(len(line)):
                if the_str_1 == line[i]:
                    list_1.append(i)
            break
    print(len(list_1))
    list_1 = list(set(list_1))
    print(len(list_1))

    dataset = read_csv("temp_drawing_1.csv")
    using = dataset[list_1]
    # print(using.values)
    list_2 = []
    length = len(using.values)
    print(length)
    sum = 0
    for i in range(length):
        print('doing', i/length)
        for j in range(len(list_1)):
            num = float(using.values[i][j])
            if num != -100:
                list_2.append(num)
                sum += num
                # print(sum, '- - -', num)
    # list_2.sort()
    print(list_2[0])
    print(list_2[-1])
    # print(list_2)
    print(np.mean(list_2))






    path_from_1 = "temp_drawing_1.csv"
    list_3 = []
    with open(path_from_1, "r") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            for i in range(len(line)):
                if the_str_2 == line[i]:
                    list_3.append(i)
            break
    print(len(list_3))
    list_1 = list(set(list_3))
    print(len(list_3))

    dataset = read_csv("temp_drawing_1.csv")
    using = dataset[list_3]
    # print(using.values)
    list_4 = []
    length = len(using.values)
    print(length)
    sum = 0
    for i in range(length):
        print('doing', i / length)
        for j in range(len(list_3)):
            num = float(using.values[i][j])
            if num != -100:
                list_4.append(num)
                sum += num
                # print(sum, '- - -', num)
    # list_2.sort()
    print(list_4[0])
    print(list_4[-1])
    # print(list_2)
    print(np.mean(list_4))





    path_from_1 = "precip_drawing_1.csv"
    list_5 = []
    with open(path_from_1, "r") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            for i in range(len(line)):
                if the_str_3 == line[i]:
                    list_5.append(i)
            break
    print(len(list_5))
    list_1 = list(set(list_5))
    print(len(list_5))

    dataset = read_csv("temp_drawing_1.csv")
    using = dataset[list_5]
    # print(using.values)
    list_6 = []
    length = len(using.values)
    print(length)
    sum = 0
    for i in range(length):
        print('doing', i / length)
        for j in range(len(list_5)):
            num = float(using.values[i][j])
            if num != -100:
                list_6.append(num)
                sum += num
                # print(sum, '- - -', num)
    # list_2.sort()
    print(list_6[0])
    print(list_6[-1])
    # print(list_2)
    print(np.mean(list_6))











    x_1 = [x+1958 for x in range(len(list_2))]
    y_1 = [x/100 for x in range(1, 500)]

    plt.figure()
    # plt.plot(x_1, y_1, color='red', label='y=x')
    plt.subplot(311)
    plt.plot(x_1, list_2, "-")
    plt.xlabel("Year")
    plt.ylabel('AAT')

    plt.subplot(312)
    plt.plot(x_1, list_4, "-")
    plt.xlabel("Year")
    plt.ylabel('ADT32')

    plt.subplot(313)
    plt.plot(x_1, list_6, "-")
    plt.xlabel("Year")
    plt.ylabel('TAP')




    plt.show()



if __name__ == '__main__':
    readthecsv()