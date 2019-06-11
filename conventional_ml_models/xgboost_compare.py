import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
import random
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import ensemble
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.model_selection import train_test_split, KFold

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from graphviz import Digraph

r_2_list = []
rmse_list = []



def only_climate():
    dataset = read_csv('final_version_structure.csv', header=0, index_col=0)
    dataset = shuffle(dataset)  # 打乱顺序
    dataset.dropna(axis=0, how='any', inplace=True)
    using = dataset[
        ['MRI', 'MRI0', 'DAYS_BETWEEN', 'LAT', 'LONGT', 'ELEV', 'AGE_TO_MRI_MEASURE', 'TRUCK_TRAFFIC', 'KESAL_TRAFFIC'
            , '1_layer_exists', '1_layer_thickness', '1_type_AC', '1_type_EF', '1_type_GB', '1_type_GS', '1_type_PC',
         '1_type_SS', '1_type_TB', '1_type_TS', '2_layer_exists', '2_layer_thickness', '2_type_AC', '2_type_EF',
         '2_type_GB',
         '2_type_GS', '2_type_PC', '2_type_SS', '2_type_TB', '2_type_TS', '3_layer_exists', '3_layer_thickness',
         '3_type_AC', '3_type_EF', '3_type_GB', '3_type_GS', '3_type_PC', '3_type_SS', '3_type_TB', '3_type_TS',
         '4_layer_exists', '4_layer_thickness', '4_type_AC', '4_type_EF', '4_type_GB', '4_type_GS', '4_type_PC',
         '4_type_SS',
         '4_type_TB', '4_type_TS', '5_layer_exists', '5_layer_thickness', '5_type_AC', '5_type_EF', '5_type_GB',
         '5_type_GS', '5_type_PC', '5_type_SS', '5_type_TB', '5_type_TS', '6_layer_exists', '6_layer_thickness',
         '6_type_AC',
         '6_type_EF', '6_type_GB', '6_type_GS', '6_type_PC', '6_type_SS', '6_type_TB', '6_type_TS', '7_layer_exists',
         '7_layer_thickness', '7_type_AC', '7_type_EF', '7_type_GB', '7_type_GS', '7_type_PC', '7_type_SS', '7_type_TB',
         '7_type_TS', '8_layer_exists', '8_layer_thickness', '8_type_AC', '8_type_EF', '8_type_GB', '8_type_GS',
         '8_type_PC',
         '8_type_SS', '8_type_TB', '8_type_TS', '9_layer_exists', '9_layer_thickness', '9_type_AC', '9_type_EF',
         '9_type_GB', '9_type_GS', '9_type_PC', '9_type_SS', '9_type_TB', '9_type_TS', '10_layer_exists',
         '10_layer_thickness', '10_type_AC', '10_type_EF', '10_type_GB', '10_type_GS', '10_type_PC', '10_type_SS',
         '10_type_TB', '10_type_TS', '11_layer_exists', '11_layer_thickness', '11_type_AC', '11_type_EF', '11_type_GB',
         '11_type_GS', '11_type_PC', '11_type_SS', '11_type_TB', '11_type_TS', '12_layer_exists', '12_layer_thickness',
         '12_type_AC', '12_type_EF', '12_type_GB', '2_type_GS', '12_type_PC', '12_type_SS', '12_type_TB', '12_type_TS',
         '13_layer_exists', '13_layer_thickness', '13_type_AC', '13_type_EF', '13_type_GB', '13_type_GS', '13_type_PC',
         '13_type_SS', '13_type_TB', '13_type_TS', '14_layer_exists', '14_layer_thickness', '14_type_AC', '14_type_EF',
         '14_type_GB', '14_type_GS', '14_type_PC', '14_type_SS', '14_type_TB', '14_type_TS'
         # cliamte
            , 'MEAN_TEMP_AVG_LAST_1', 'FREEZE_INDEX_LAST_1', 'FREEZE_THAW_LAST_1', 'MAX_TEMP_AVG_LAST_1',
         'MIN_TEMP_AVG_LAST_1', 'MAX_TEMP_LAST_1', 'MIN_TEMP_LAST_1', 'DAYS_ABOVE_32_LAST_1', 'DAYS_BELOW_0_LAST_1',
         'TOTAL_PRECIP_LAST_1', 'TOTAL_SNOWFALL_LAST_1', 'INTENSE_PRECIP_DAYS_LAST_1', 'WET_DAYS_LAST_1'
            , 'MEAN_TEMP_AVG_LAST_2', 'FREEZE_INDEX_LAST_2', 'FREEZE_THAW_LAST_2', 'MAX_TEMP_AVG_LAST_2',
         'MIN_TEMP_AVG_LAST_2', 'MAX_TEMP_LAST_2', 'MIN_TEMP_LAST_2', 'DAYS_ABOVE_32_LAST_2', 'DAYS_BELOW_0_LAST_2',
         'TOTAL_PRECIP_LAST_2', 'TOTAL_SNOWFALL_LAST_2', 'INTENSE_PRECIP_DAYS_LAST_2', 'WET_DAYS_LAST_2'
            , 'MEAN_TEMP_AVG_LAST_3', 'FREEZE_INDEX_LAST_3', 'FREEZE_THAW_LAST_3', 'MAX_TEMP_AVG_LAST_3',
         'MIN_TEMP_AVG_LAST_3', 'MAX_TEMP_LAST_3', 'MIN_TEMP_LAST_3', 'DAYS_ABOVE_32_LAST_3', 'DAYS_BELOW_0_LAST_3',
         'TOTAL_PRECIP_LAST_3', 'TOTAL_SNOWFALL_LAST_3', 'INTENSE_PRECIP_DAYS_LAST_3', 'WET_DAYS_LAST_3'
            , 'MEAN_TEMP_AVG_LAST_4', 'FREEZE_INDEX_LAST_4', 'FREEZE_THAW_LAST_4', 'MAX_TEMP_AVG_LAST_4',
         'MIN_TEMP_AVG_LAST_4', 'MAX_TEMP_LAST_4', 'MIN_TEMP_LAST_4', 'DAYS_ABOVE_32_LAST_4', 'DAYS_BELOW_0_LAST_4',
         'TOTAL_PRECIP_LAST_4', 'TOTAL_SNOWFALL_LAST_4', 'INTENSE_PRECIP_DAYS_LAST_4', 'WET_DAYS_LAST_4'
            , 'MEAN_TEMP_AVG_LAST_5', 'FREEZE_INDEX_LAST_5', 'FREEZE_THAW_LAST_5', 'MAX_TEMP_AVG_LAST_5',
         'MIN_TEMP_AVG_LAST_5', 'MAX_TEMP_LAST_5', 'MIN_TEMP_LAST_5', 'DAYS_ABOVE_32_LAST_5', 'DAYS_BELOW_0_LAST_5',
         'TOTAL_PRECIP_LAST_5', 'TOTAL_SNOWFALL_LAST_5', 'INTENSE_PRECIP_DAYS_LAST_5', 'WET_DAYS_LAST_5'
            , 'MEAN_TEMP_AVG_LAST_6', 'FREEZE_INDEX_LAST_6', 'FREEZE_THAW_LAST_6', 'MAX_TEMP_AVG_LAST_6',
         'MIN_TEMP_AVG_LAST_6', 'MAX_TEMP_LAST_6', 'MIN_TEMP_LAST_6', 'DAYS_ABOVE_32_LAST_6', 'DAYS_BELOW_0_LAST_6',
         'TOTAL_PRECIP_LAST_6', 'TOTAL_SNOWFALL_LAST_6', 'INTENSE_PRECIP_DAYS_LAST_6', 'WET_DAYS_LAST_6'
            , 'MEAN_TEMP_AVG_LAST_7', 'FREEZE_INDEX_LAST_7', 'FREEZE_THAW_LAST_7', 'MAX_TEMP_AVG_LAST_7',
         'MIN_TEMP_AVG_LAST_7', 'MAX_TEMP_LAST_7', 'MIN_TEMP_LAST_7', 'DAYS_ABOVE_32_LAST_7', 'DAYS_BELOW_0_LAST_7',
         'TOTAL_PRECIP_LAST_7', 'TOTAL_SNOWFALL_LAST_7', 'INTENSE_PRECIP_DAYS_LAST_7', 'WET_DAYS_LAST_7'
            , 'MEAN_TEMP_AVG_LAST_8', 'FREEZE_INDEX_LAST_8', 'FREEZE_THAW_LAST_8', 'MAX_TEMP_AVG_LAST_8',
         'MIN_TEMP_AVG_LAST_8', 'MAX_TEMP_LAST_8', 'MIN_TEMP_LAST_8', 'DAYS_ABOVE_32_LAST_8', 'DAYS_BELOW_0_LAST_8',
         'TOTAL_PRECIP_LAST_8', 'TOTAL_SNOWFALL_LAST_8', 'INTENSE_PRECIP_DAYS_LAST_8', 'WET_DAYS_LAST_8'
            , 'MEAN_TEMP_AVG_LAST_9', 'FREEZE_INDEX_LAST_9', 'FREEZE_THAW_LAST_9', 'MAX_TEMP_AVG_LAST_9',
         'MIN_TEMP_AVG_LAST_9', 'MAX_TEMP_LAST_9', 'MIN_TEMP_LAST_9', 'DAYS_ABOVE_32_LAST_9', 'DAYS_BELOW_0_LAST_9',
         'TOTAL_PRECIP_LAST_9', 'TOTAL_SNOWFALL_LAST_9', 'INTENSE_PRECIP_DAYS_LAST_9', 'WET_DAYS_LAST_9'
            , 'MEAN_TEMP_AVG_LAST_10', 'FREEZE_INDEX_LAST_10', 'FREEZE_THAW_LAST_10', 'MAX_TEMP_AVG_LAST_10',
         'MIN_TEMP_AVG_LAST_10', 'MAX_TEMP_LAST_10', 'MIN_TEMP_LAST_10', 'DAYS_ABOVE_32_LAST_10',
         'DAYS_BELOW_0_LAST_10', 'TOTAL_PRECIP_LAST_10', 'TOTAL_SNOWFALL_LAST_10', 'INTENSE_PRECIP_DAYS_LAST_10',
         'WET_DAYS_LAST_10'

         # # cliamte_reverse_for_right_time_series
         # , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'
         # , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
         # , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
         # , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
         # , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
         # , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
         # , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
         # , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
         # , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
         # , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'

         ]]
    pca = PCA(n_components=5, whiten=True, random_state=42)
    XList = np.concatenate([np.array(using.values[:, 1]).reshape(using.values.shape[0], 1), pca.fit_transform(np.array(using.values[:, 9:149]))],
                           axis=1)
    print(np.array(XList).shape)
    yList = using.values[:, 0]
    print(np.array(yList).shape)

    # 13类气候特征，先append上所有的均值方差，然后做reshape，concat

    list_features = []

    middle_list = []
    analysis_features = []

    data = using.values
    # 清零list
    for i in range(len(data)):
        for j in range(13):
            for k in range(10):
                if str(data[i][149 + k * 13]) != '-100':
                    middle_list.append(float(data[i][149 + j + k * 13]))
            analysis_features.append(np.mean(middle_list))
            analysis_features.append(np.var(middle_list))
            if not analysis_features:
                analysis_features.append(float(0))
                analysis_features.append(float(0))
            if len(analysis_features) == 1:
                analysis_features.append(float(0))
            list_features.append(analysis_features)
            middle_list = []
            analysis_features = []
    print(len(list_features))

    middle_list = []
    analysis_features = []
    count = 0
    new_list_for_concat = []
    for i in range(len(list_features)):
        middle_list.append(list_features[i][0])
        middle_list.append(list_features[i][1])
        count += 1
        if count == 13:
            analysis_features.append(middle_list)
            middle_list = []
            count = 0
    print(len(analysis_features))
    XList = np.concatenate([XList, analysis_features], axis=1)


    return XList, yList  # 1738




def loadDataset(filePath):
    df = pd.read_csv(filepath_or_buffer=filePath)
    df.dropna(axis=0, how='any', inplace=True)
    df = shuffle(df)  # 打乱顺序
    missing_val_count_by_column = (df.isnull().sum())
    print(missing_val_count_by_column)
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # 缺失值删除
    # df.fillna(0)


    missing_val_count_by_column = (df.isnull().sum())
    print(missing_val_count_by_column)
    print(missing_val_count_by_column[missing_val_count_by_column > 0])


    return df


def featureSet(data):
    data_num = len(data)
    XList = []
    count_1 = 0
    print("loading train data...")
    print(data_num)
    dataset = read_csv('final_version_structure.csv', header=0, index_col=0)
    dataset = shuffle(dataset)  # 打乱顺序
    dataset.dropna(axis=0, how='any', inplace=True)
    using = dataset[['MRI', 'MRI0', 'DAYS_BETWEEN', 'LAT', 'LONGT', 'ELEV', 'AGE_TO_MRI_MEASURE', 'TRUCK_TRAFFIC', 'KESAL_TRAFFIC'
                     ,'1_layer_exists', '1_layer_thickness', '1_type_AC', '1_type_EF', '1_type_GB', '1_type_GS', '1_type_PC', '1_type_SS', '1_type_TB', '1_type_TS', '2_layer_exists', '2_layer_thickness', '2_type_AC', '2_type_EF', '2_type_GB',
                     '2_type_GS', '2_type_PC', '2_type_SS', '2_type_TB', '2_type_TS', '3_layer_exists', '3_layer_thickness', '3_type_AC', '3_type_EF', '3_type_GB', '3_type_GS', '3_type_PC', '3_type_SS', '3_type_TB', '3_type_TS', '4_layer_exists', '4_layer_thickness', '4_type_AC', '4_type_EF', '4_type_GB', '4_type_GS', '4_type_PC', '4_type_SS',
                     '4_type_TB', '4_type_TS', '5_layer_exists', '5_layer_thickness', '5_type_AC', '5_type_EF', '5_type_GB', '5_type_GS', '5_type_PC', '5_type_SS', '5_type_TB', '5_type_TS', '6_layer_exists', '6_layer_thickness', '6_type_AC',
                     '6_type_EF', '6_type_GB', '6_type_GS', '6_type_PC', '6_type_SS', '6_type_TB', '6_type_TS', '7_layer_exists', '7_layer_thickness', '7_type_AC', '7_type_EF', '7_type_GB', '7_type_GS', '7_type_PC', '7_type_SS', '7_type_TB', '7_type_TS', '8_layer_exists', '8_layer_thickness', '8_type_AC', '8_type_EF', '8_type_GB', '8_type_GS', '8_type_PC',
                     '8_type_SS', '8_type_TB', '8_type_TS', '9_layer_exists', '9_layer_thickness', '9_type_AC', '9_type_EF', '9_type_GB', '9_type_GS', '9_type_PC', '9_type_SS', '9_type_TB', '9_type_TS', '10_layer_exists', '10_layer_thickness', '10_type_AC', '10_type_EF', '10_type_GB', '10_type_GS', '10_type_PC', '10_type_SS', '10_type_TB', '10_type_TS', '11_layer_exists', '11_layer_thickness', '11_type_AC', '11_type_EF', '11_type_GB', '11_type_GS', '11_type_PC', '11_type_SS', '11_type_TB', '11_type_TS', '12_layer_exists', '12_layer_thickness', '12_type_AC', '12_type_EF', '12_type_GB', '2_type_GS', '12_type_PC', '12_type_SS', '12_type_TB', '12_type_TS', '13_layer_exists', '13_layer_thickness', '13_type_AC', '13_type_EF', '13_type_GB', '13_type_GS', '13_type_PC', '13_type_SS', '13_type_TB', '13_type_TS', '14_layer_exists', '14_layer_thickness', '14_type_AC', '14_type_EF', '14_type_GB', '14_type_GS', '14_type_PC', '14_type_SS', '14_type_TB', '14_type_TS'
                     #cliamte
                     , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'
                     , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'

                     # # cliamte_reverse_for_right_time_series
                     # , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'
                     # , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     # , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     # , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     # , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     # , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     # , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     # , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     # , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     # , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'

                     ]]

    for i in range(9):
        print('f' + str(i-1) + ':' + using.columns[i])
    for i in range(5):
        print('f' + str(i+8) + ':' + "structure - " + str(i+1))
    the_climate_name = ['MEAN_TEMP_AVG','FREEZE_INDEX','FREEZE_THAW','MAX_TEMP_AVG','MIN_TEMP_AVG','MAX_TEMP','MIN_TEMP','DAYS_ABOVE_32','DAYS_BELOW_0','TOTAL_PRECIP','TOTAL_SNOWFALL','INTENSE_PRECIP_DAYS','WET_DAYS']
    for i in range(13):
        print('f' + str(2 * i+13) + ':' + 'MEAN_' + the_climate_name[i])
        print('f' + str(2 * i+14) + ':' + 'VAR_' + the_climate_name[i])

    # 9 + 5 + 26
    pca = PCA(n_components=5, whiten=True, random_state=42)
    XList = np.concatenate([np.array(using.values[:, 1:9]), pca.fit_transform(np.array(using.values[:, 9:149]))], axis=1)
    print(np.array(XList).shape)

    # 13类气候特征，先append上所有的均值方差，然后做reshape，concat

    list_features = []

    middle_list = []
    analysis_features = []

    data = using.values
    # 清零list
    for i in range(len(data)):
        for j in range(13):
            for k in range(10):
                if str(data[i][149 + k * 13]) != '-100':
                    middle_list.append(float(data[i][149 + j + k * 13]))
            analysis_features.append(np.mean(middle_list))
            analysis_features.append(np.var(middle_list))
            if not analysis_features:
                analysis_features.append(float(0))
                analysis_features.append(float(0))
            if len(analysis_features) == 1:
                analysis_features.append(float(0))
            list_features.append(analysis_features)
            middle_list = []
            analysis_features = []
    print(len(list_features))

    middle_list = []
    analysis_features = []
    count = 0
    new_list_for_concat = []
    for i in range(len(list_features)):
        middle_list.append(list_features[i][0])
        middle_list.append(list_features[i][1])
        count += 1
        if count == 13:
            analysis_features.append(middle_list)
            middle_list = []
            count = 0
    print(len(analysis_features))
    XList = np.concatenate([XList, analysis_features], axis=1)



    yList = using.values[:, 0]

    return XList, yList  # 1738

















    # for row in range(0, data_num-500):
    #     tmp_list = []
    #     print("training data: ", count_1)
    #     count_1 += 1
    #
    #     # # temp_annual
    #     # tmp_list.append(data.iloc[row]['MEAN_ANN_TEMP_AVG'])  # f8               f0`
    #     # tmp_list.append(data.iloc[row]['FREEZE_INDEX_YR'])    # f9               f1`
    #     # tmp_list.append(data.iloc[row]['FREEZE_THAW_YR'])    # f10               f2`
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_AVG'])    # f11             f3
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_AVG'])    # f12             f4
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP'])    # f13                 f5`
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP'])    # f14                 f6`
    #     # tmp_list.append(data.iloc[row]['DAYS_ABOVE_32_C_YR'])    # f15           f7`
    #     # tmp_list.append(data.iloc[row]['DAYS_BELOW_0_C_YR'])    # f16            f8
    #     #
    #     # # precipitation
    #     # tmp_list.append(data.iloc[row]['TOTAL_ANN_PRECIP'])  # f8                f9`
    #     # tmp_list.append(data.iloc[row]['TOTAL_SNOWFALL_YR'])  # f9               f10`
    #     # tmp_list.append(data.iloc[row]['INTENSE_PRECIP_DAYS_YR'])  # f10         f11
    #     # tmp_list.append(data.iloc[row]['WET_DAYS_YR'])  # f11                    f12`
    #     # # tmp_list.append(data.iloc[row]['DAYS_DELTA'])  # f11                    f12`*********
    #     #
    #     # # window
    #     # tmp_list.append(data.iloc[row]['MRI_LAST_2'])  # f8                      f13
    #     # tmp_list.append(data.iloc[row]['DAYS_AWAY_2'])  # f8                      f13
    #     # tmp_list.append(data.iloc[row]['MEAN_ANN_TEMP_AVG_LAST_2'])    # f10     f15
    #     # tmp_list.append(data.iloc[row]['FREEZE_INDEX_YR_2'])    # f11            f16
    #     # tmp_list.append(data.iloc[row]['FREEZE_THAW_YR_2'])    # f12             f17
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_AVG_2'])    # f13           f18
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_AVG_2'])    # f14           f19
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_2'])    # f15               f20
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_2'])    # f16               f21
    #     # tmp_list.append(data.iloc[row]['DAYS_ABOVE_32_C_YR_2'])  # f8            f22
    #     # tmp_list.append(data.iloc[row]['DAYS_BELOW_0_C_YR_2'])  # f8             f23
    #     #
    #     # tmp_list.append(data.iloc[row]['TOTAL_ANN_PRECIP_2'])  # f10             f24
    #     # tmp_list.append(data.iloc[row]['TOTAL_SNOWFALL_YR_2'])  # f11            f25
    #     # tmp_list.append(data.iloc[row]['INTENSE_PRECIP_DAYS_YR_2'])  # f11       f26
    #     # tmp_list.append(data.iloc[row]['WET_DAYS_YR_2'])  # f11                  f27
    #
    #
    #     # # window
    #     # tmp_list.append(data.iloc[row]['MRI_LAST_3'])  # f8                      f13
    #     # tmp_list.append(data.iloc[row]['MEAN_ANN_TEMP_AVG_LAST_3'])  # f10     f15
    #     # tmp_list.append(data.iloc[row]['FREEZE_INDEX_YR_3'])  # f11            f16
    #     # tmp_list.append(data.iloc[row]['FREEZE_THAW_YR_3'])  # f12             f17
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_AVG_3'])  # f13           f18
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_AVG_3'])  # f14           f19
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_3'])  # f15               f20
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_3'])  # f16               f21
    #     # tmp_list.append(data.iloc[row]['DAYS_ABOVE_32_C_YR_3'])  # f8            f22
    #     # tmp_list.append(data.iloc[row]['DAYS_BELOW_0_C_YR_3'])  # f8             f23
    #     #
    #     # tmp_list.append(data.iloc[row]['TOTAL_ANN_PRECIP_3'])  # f8                f9`
    #     # tmp_list.append(data.iloc[row]['TOTAL_SNOWFALL_YR_3'])  # f9               f10`
    #     # tmp_list.append(data.iloc[row]['INTENSE_PRECIP_DAYS_YR_3'])  # f10         f11
    #     # tmp_list.append(data.iloc[row]['WET_DAYS_YR_3'])  # f11                    f12`
    #
    #     # MRI,CONS_NO_THIS_TIME,MRI0_DAYS_TO_NOW,,,,,,
    #     tmp_list.append(data.iloc[row]['MRI0'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['DAYS_BETWEEN'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['LAT'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['LONGT'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['ELEV'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['AGE_TO_MRI_MEASURE'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['TRUCK_TRAFFIC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['KESAL_TRAFFIC'])  # f11                    f12`
    #     # ,,,,,,
    #     tmp_list.append(data.iloc[row]['1_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['2_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['3_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['4_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['5_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['6_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['7_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['8_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['9_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['10_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['11_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['12_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['13_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['14_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_TS'])  # f11                    f12`
    #
    #
    #     XList.append(tmp_list)
    #yList = data.MRI.values
    # yList = using.values[:, 0]
    # print(np.array(yList).shape)
    # return XList, yList #1738


def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data.dropna(axis=0, how='any', inplace=True)
    print("loading test data...")
    data = shuffle(data)  # 打乱顺序
    data_num = len(data)
    count_1 = 0
    XList = []
    dataset = read_csv('final_version_structure.csv', header=0, index_col=0)
    dataset.dropna(axis=0, how='any', inplace=True)
    using = dataset[['MRI', 'MRI0', 'DAYS_BETWEEN', 'LAT', 'LONGT', 'ELEV', 'AGE_TO_MRI_MEASURE', 'TRUCK_TRAFFIC', 'KESAL_TRAFFIC'
                     ,'1_layer_exists', '1_layer_thickness', '1_type_AC', '1_type_EF', '1_type_GB', '1_type_GS', '1_type_PC', '1_type_SS', '1_type_TB', '1_type_TS', '2_layer_exists', '2_layer_thickness', '2_type_AC', '2_type_EF', '2_type_GB',
                     '2_type_GS', '2_type_PC', '2_type_SS', '2_type_TB', '2_type_TS', '3_layer_exists', '3_layer_thickness', '3_type_AC', '3_type_EF', '3_type_GB', '3_type_GS', '3_type_PC', '3_type_SS', '3_type_TB', '3_type_TS', '4_layer_exists', '4_layer_thickness', '4_type_AC', '4_type_EF', '4_type_GB', '4_type_GS', '4_type_PC', '4_type_SS',
                     '4_type_TB', '4_type_TS', '5_layer_exists', '5_layer_thickness', '5_type_AC', '5_type_EF', '5_type_GB', '5_type_GS', '5_type_PC', '5_type_SS', '5_type_TB', '5_type_TS', '6_layer_exists', '6_layer_thickness', '6_type_AC',
                     '6_type_EF', '6_type_GB', '6_type_GS', '6_type_PC', '6_type_SS', '6_type_TB', '6_type_TS', '7_layer_exists', '7_layer_thickness', '7_type_AC', '7_type_EF', '7_type_GB', '7_type_GS', '7_type_PC', '7_type_SS', '7_type_TB', '7_type_TS', '8_layer_exists', '8_layer_thickness', '8_type_AC', '8_type_EF', '8_type_GB', '8_type_GS', '8_type_PC',
                     '8_type_SS', '8_type_TB', '8_type_TS', '9_layer_exists', '9_layer_thickness', '9_type_AC', '9_type_EF', '9_type_GB', '9_type_GS', '9_type_PC', '9_type_SS', '9_type_TB', '9_type_TS', '10_layer_exists', '10_layer_thickness', '10_type_AC', '10_type_EF', '10_type_GB', '10_type_GS', '10_type_PC', '10_type_SS', '10_type_TB', '10_type_TS', '11_layer_exists', '11_layer_thickness', '11_type_AC', '11_type_EF', '11_type_GB', '11_type_GS', '11_type_PC', '11_type_SS', '11_type_TB', '11_type_TS', '12_layer_exists', '12_layer_thickness', '12_type_AC', '12_type_EF', '12_type_GB', '2_type_GS', '12_type_PC', '12_type_SS', '12_type_TB', '12_type_TS', '13_layer_exists', '13_layer_thickness', '13_type_AC', '13_type_EF', '13_type_GB', '13_type_GS', '13_type_PC', '13_type_SS', '13_type_TB', '13_type_TS', '14_layer_exists', '14_layer_thickness', '14_type_AC', '14_type_EF', '14_type_GB', '14_type_GS', '14_type_PC', '14_type_SS', '14_type_TB', '14_type_TS'
                     #cliamte
                     , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'
                     , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'

                     # # cliamte_reverse_for_right_time_series
                     # , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'
                     # , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     # , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     # , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     # , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     # , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     # , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     # , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     # , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     # , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'

                     ]]


    # XList = using.values[data_num-500:, 1:149]

    pca = PCA(n_components=5, whiten=True, random_state=42)
    XList = np.concatenate([np.array(using.values[data_num-500:, 1:9]), pca.fit_transform(np.array(using.values[data_num-500:, 9:149]))], axis=1)



    # for row in range(0, data_num-500):
    #     tmp_list = []
    #     print("training data: ", count_1)
    #     count_1 += 1
    #
    #     # # temp_annual
    #     # tmp_list.append(data.iloc[row]['MEAN_ANN_TEMP_AVG'])  # f8               f0`
    #     # tmp_list.append(data.iloc[row]['FREEZE_INDEX_YR'])    # f9               f1`
    #     # tmp_list.append(data.iloc[row]['FREEZE_THAW_YR'])    # f10               f2`
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_AVG'])    # f11             f3
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_AVG'])    # f12             f4
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP'])    # f13                 f5`
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP'])    # f14                 f6`
    #     # tmp_list.append(data.iloc[row]['DAYS_ABOVE_32_C_YR'])    # f15           f7`
    #     # tmp_list.append(data.iloc[row]['DAYS_BELOW_0_C_YR'])    # f16            f8
    #     #
    #     # # precipitation
    #     # tmp_list.append(data.iloc[row]['TOTAL_ANN_PRECIP'])  # f8                f9`
    #     # tmp_list.append(data.iloc[row]['TOTAL_SNOWFALL_YR'])  # f9               f10`
    #     # tmp_list.append(data.iloc[row]['INTENSE_PRECIP_DAYS_YR'])  # f10         f11
    #     # tmp_list.append(data.iloc[row]['WET_DAYS_YR'])  # f11                    f12`
    #     # # tmp_list.append(data.iloc[row]['DAYS_DELTA'])  # f11                    f12`*********
    #     #
    #     # # window
    #     # tmp_list.append(data.iloc[row]['MRI_LAST_2'])  # f8                      f13
    #     # tmp_list.append(data.iloc[row]['DAYS_AWAY_2'])  # f8                      f13
    #     # tmp_list.append(data.iloc[row]['MEAN_ANN_TEMP_AVG_LAST_2'])    # f10     f15
    #     # tmp_list.append(data.iloc[row]['FREEZE_INDEX_YR_2'])    # f11            f16
    #     # tmp_list.append(data.iloc[row]['FREEZE_THAW_YR_2'])    # f12             f17
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_AVG_2'])    # f13           f18
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_AVG_2'])    # f14           f19
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_2'])    # f15               f20
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_2'])    # f16               f21
    #     # tmp_list.append(data.iloc[row]['DAYS_ABOVE_32_C_YR_2'])  # f8            f22
    #     # tmp_list.append(data.iloc[row]['DAYS_BELOW_0_C_YR_2'])  # f8             f23
    #     #
    #     # tmp_list.append(data.iloc[row]['TOTAL_ANN_PRECIP_2'])  # f10             f24
    #     # tmp_list.append(data.iloc[row]['TOTAL_SNOWFALL_YR_2'])  # f11            f25
    #     # tmp_list.append(data.iloc[row]['INTENSE_PRECIP_DAYS_YR_2'])  # f11       f26
    #     # tmp_list.append(data.iloc[row]['WET_DAYS_YR_2'])  # f11                  f27
    #
    #
    #     # # window
    #     # tmp_list.append(data.iloc[row]['MRI_LAST_3'])  # f8                      f13
    #     # tmp_list.append(data.iloc[row]['MEAN_ANN_TEMP_AVG_LAST_3'])  # f10     f15
    #     # tmp_list.append(data.iloc[row]['FREEZE_INDEX_YR_3'])  # f11            f16
    #     # tmp_list.append(data.iloc[row]['FREEZE_THAW_YR_3'])  # f12             f17
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_AVG_3'])  # f13           f18
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_AVG_3'])  # f14           f19
    #     # tmp_list.append(data.iloc[row]['MAX_ANN_TEMP_3'])  # f15               f20
    #     # tmp_list.append(data.iloc[row]['MIN_ANN_TEMP_3'])  # f16               f21
    #     # tmp_list.append(data.iloc[row]['DAYS_ABOVE_32_C_YR_3'])  # f8            f22
    #     # tmp_list.append(data.iloc[row]['DAYS_BELOW_0_C_YR_3'])  # f8             f23
    #     #
    #     # tmp_list.append(data.iloc[row]['TOTAL_ANN_PRECIP_3'])  # f8                f9`
    #     # tmp_list.append(data.iloc[row]['TOTAL_SNOWFALL_YR_3'])  # f9               f10`
    #     # tmp_list.append(data.iloc[row]['INTENSE_PRECIP_DAYS_YR_3'])  # f10         f11
    #     # tmp_list.append(data.iloc[row]['WET_DAYS_YR_3'])  # f11                    f12`
    #
    #     # MRI,CONS_NO_THIS_TIME,MRI0_DAYS_TO_NOW,,,,,,
    #     tmp_list.append(data.iloc[row]['MRI0'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['DAYS_BETWEEN'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['LAT'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['LONGT'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['ELEV'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['AGE_TO_MRI_MEASURE'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['TRUCK_TRAFFIC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['KESAL_TRAFFIC'])  # f11                    f12`
    #     # ,,,,,,
    #     tmp_list.append(data.iloc[row]['1_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['1_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['2_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['2_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['3_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['3_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['4_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['4_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['5_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['5_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['6_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['6_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['7_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['7_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['8_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['8_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['9_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['9_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['10_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['10_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['11_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['11_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['12_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['12_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['13_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['13_type_TS'])  # f11                    f12`
    #
    #     tmp_list.append(data.iloc[row]['14_layer_exists'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_layer_thickness'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_AC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_EF'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_GB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_GS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_PC'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_SS'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_TB'])  # f11                    f12`
    #     tmp_list.append(data.iloc[row]['14_type_TS'])  # f11                    f12`
    #
    #
    #     XList.append(tmp_list)
    #yList = data.MRI.values
    YList = using.values[:, 0]
    return XList, YList[data_num-500:data_num]


def trainandTest(X_train, y_train, X_test, Y_test):

    # XGBoost训练过程
    # model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')

    # excel版本
    # model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=80, max_depth=3, min_child_weight=6, seed=0,
    #                          subsample=0.7, colsample_bytree=0.9, gamma=0, reg_alpha=0.05, reg_lambda=3)


        # shuffle + （1）->（2）
    model = xgb.XGBRegressor(learning_rate=0.12, n_estimators=400, max_depth=4, min_child_weight=4, seed=0,
                                 subsample=0.7, colsample_bytree=0.9, gamma=0.1, reg_alpha=0.05, reg_lambda=3)

    # shuffle + （1）+ (2)->（3）
    # model = xgb.XGBRegressor(learning_rate=0.07, n_estimators=550, max_depth=6, min_child_weight=2, seed=0,
    #                          subsample=0.7, colsample_bytree=0.8, gamma=0.05, reg_alpha=2, reg_lambda=2)

    # 随机森林回归
    # model = ensemble.RandomForestRegressor(n_estimators=20)  # 使用20个决策树
    print("start training...")
    model.fit(X_train, y_train)

    # 对测试集进行预测
    print("start predicting...")
    ans = model.predict(X_test)

    r_2 = model.score(X_test, Y_test)

    # print(model.score(X_test, Y_test))
    rmse = sqrt(mean_squared_error(Y_test, ans))
    print('Test RMSE: %.3f' % rmse)
    print('Test R^2: %.3f' % r_2)

    rmse_list.append(rmse)
    # print('Test RMSE: %.3f' % rmse)
    r_2_list.append(r_2)

    plot_tree(model)
    plt.show()


    # ######################################################################
    # k_range = range(1, 31)
    # cv_scores = []
    # for n in k_range:
    #     scores = cross_val_score(model, X_train, y_train, cv=10)  # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
    #     cv_scores.append(scores.mean())
    #     print(n, ": ", scores.mean())
    # plt.plot(k_range, cv_scores)
    # plt.xlabel('K')
    # plt.ylabel('Accuracy')  # 通过图像选择最好的参数
    # plt.show()
    # ######################################################################


    # 交叉验证部分
    # list_cross_val = cross_val_score(model, X_train, y_train, cv=5)
    # print(list_cross_val)
    # total = 0.0
    # for each in list_cross_val:
    #     total += each
    # print(total/len(list_cross_val))

    x_1 = [x/100 for x in range(1, 500)]
    y_1 = [x/100 for x in range(1, 500)]



    plt.figure(figsize=(4.4, 4.0))
    plt.plot(Y_test, ans, "o", markersize=2.)
    # print(Y_test)
    # print(ans)
    plt.plot(x_1, y_1, color='red', label='Predicted IRI = Real IRI')
    plt.xlabel("Real IRI")
    plt.ylabel("Predicted IRI")
    plt.text(4, 0, r'$R^{2} = %.3f$' % r_2)  # 文本中注释
    plt.legend()
    plt.show()

    # #####################################################
    # # 写文件用的
    # ans_len = len(ans)
    # # id_list = np.arange(10441, 17441)
    # data_arr = []
    # for row in range(0, ans_len):
    #     data_arr.append([ans[row]])
    # np_data = np.array(data_arr)
    #
    # # # 写入文件
    # # pd_data = pd.DataFrame(np_data, columns=['y'])
    # # # print(pd_data)
    # # pd_data.to_csv('submit.csv', index=None)
    # #####################################################
    print(model.feature_importances_)




    # 显示重要特征-仅xgboost适用
    plot_importance(model)
    plt.show()





    # random forest regression
    ###################################################
    # importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    #
    # for f in range(len(X_train[1])):
    #     print("%d. feature x%d (%f)" % (f + 1, indices[f] + 1, importances[indices[f]]))
    ###################################################


if __name__ == '__main__':
    # trainFilePath = 'MRI_add_last_time_4_2_using.csv'
    # testFilePath = 'MRI_add_last_time_4_2_using.csv'

    X = np.array([[x] for x in range(0, 2200)])

    Y = np.array([x for x in range(0, 2200)])

    floder = KFold(n_splits=5, random_state=0, shuffle=False)

    # for train, test in floder.split(X, y):
    #     print('Train: %s | test: %s' % (train, test))
    #     print(" ")

    cvscores = []
    for (train_num, test_num) in floder.split(X, Y):
        trainFilePath = 'final_version_structure.csv'
        testFilePath = 'final_version_structure.csv'
        data = loadDataset(trainFilePath)


        X_train, y_train = featureSet(data)
        # X_test, Y_test = loadTestData(testFilePath)

        # X_train, y_train = only_climate()

        X, y = featureSet(data)


        # 与网格调参不共存
        trainandTest(X_train[train_num], y_train[train_num], X_train[test_num], y_train[test_num])



    print('************************')
    print("R^2 = ", '[', format(r_2_list[0], '.3f'), format(r_2_list[1], '.3f'), format(r_2_list[2], '.3f'),
          format(r_2_list[3], '.3f'), format(r_2_list[4], '.3f'), ']')
    print("mean: ", format(np.array(r_2_list).mean(), '.3f'))
    print("std: ", format(np.array(r_2_list).std(ddof=1), '.3f'))
    print("RMSE = ", '[', format(rmse_list[0], '.3f'), format(rmse_list[1], '.3f'), format(rmse_list[2], '.3f'),
          format(rmse_list[3], '.3f'), format(rmse_list[4], '.3f'), ']')
    print("mean: ", format(np.array(rmse_list).mean(), '.3f'))
    print("std: ", format(np.array(rmse_list).std(ddof=1), '.3f'))











    # trainFilePath = 'final_version_structure.csv'
    # testFilePath = 'final_version_structure.csv'
    # data = loadDataset(trainFilePath)
    # # X_train, y_train = featureSet(data)
    # # X_test, Y_test = loadTestData(testFilePath)
    # X, y = featureSet(data)
    # X_train, y_train = X[:-500], y[:-500]
    # X_test, Y_test = X[-500:], y[-500:]
    #
    #
    # # 与网格调参不共存
    # trainandTest(X_train, y_train, X_test, Y_test)


    # 网格调参
    # cv_params = {'n_estimators': [50, 100, 500, 1000]}
    # cv_params = {'n_estimators': [350, 400, 450, 500, 550, 600]}

    # cv_params = {'max_depth': [1, 2, 3, 4, 5, 6], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # cv_params = {'max_depth': [3], 'min_child_weight': [6, 7, 8, 9]}

    # cv_params = {'gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.4]}

    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # cv_params = {'subsample': [0.45, 0.48, 0.5, 0.52, 0.55, 0.57], 'colsample_bytree': [0.8]}

    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # cv_params = {'reg_alpha': [0.02, 0.04, 0.05], 'reg_lambda': [3, 4, 5, 6]}

    # cv_params = {'learning_rate': [0.07, 0.08, 0.1, 0.12, 0.13]}


    # other_params = {'learning_rate': 0.12, 'n_estimators': 100, 'max_depth': 6, 'min_child_weight': 2, 'seed': 0,
    #                 'subsample': 0.7, 'colsample_bytree': 0.8, 'gamma': 0.05, 'reg_alpha': 2, 'reg_lambda': 2}
    #
    # model = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    # optimized_GBM.fit(X_train, y_train)
    # evalute_result = optimized_GBM.cv_results_
    # # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    # print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
