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

    the_str = 'DAYS_BELOW_0_C_YR'

    path_from_1 = "mri_counting.csv"
    list_1 = []
    flag = 0
    with open(path_from_1, "r") as f:
        csv_read = csv.reader(f)
        for line in csv_read:
            print("doing", flag)
            if flag == 0:
                list_1.append(line)
                list_1[0].append('DATE_changed')
                flag += 1
            else:
                line.append(str(line[1].split('/')[2]) + str(line[1].split('/')[0]) + str(line[1].split('/')[1]))
                list_1.append(line)
                flag += 1

    with open("year_changed.csv", "w") as f_2:
        for i in range(len(list_1)):
            print("doing", i/len(list_1))
            for j in range(len(list_1[i])):
                if j == 0:
                    f_2.write(list_1[i][j])
                else:
                    f_2.write(',' + str(list_1[i][j]))
            f_2.write('\n')


if __name__ == "__main__":
    readthecsv()