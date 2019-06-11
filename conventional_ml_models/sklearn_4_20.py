from __future__ import print_function
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pandas import read_csv
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score


# KNeighbors
# iris = datasets.load_iris()
# iris_X = iris.data
# iris_y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(
#     iris_X, iris_y, test_size=0.3)
#
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# print(knn.predict(X_test))
# print(y_test)


# linear regression
# loaded_data = datasets.load_boston()
# data_X = loaded_data.data
# data_y = loaded_data.target
#
# model = LinearRegression()
# model.fit(data_X, data_y)
#
# print(model.predict(data_X[:4, :]))
# print(data_y[:4])



# make some data
# X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=8)
# plt.scatter(X, y)
# plt.show()


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



































def featureSet():
    dataset = read_csv('final_version_structure_5_22.csv', header=0, index_col=0)
    dataset = shuffle(dataset)  # 打乱顺序
    dataset.dropna(axis=0, how='any', inplace=True)
    using = dataset[['MRI', 'MRI0', 'DAYS_BETWEEN', 'LAT', 'LONGT', 'ELEV', 'AGE_TO_MRI_MEASURE', 'TRUCK_TRAFFIC', 'KESAL_TRAFFIC'
                     ,'1_layer_exists', '1_layer_thickness', '1_type_AC', '1_type_EF', '1_type_GB', '1_type_GS', '1_type_PC', '1_type_SS', '1_type_TB', '1_type_TS', '2_layer_exists', '2_layer_thickness', '2_type_AC', '2_type_EF', '2_type_GB',
                     '2_type_GS', '2_type_PC', '2_type_SS', '2_type_TB', '2_type_TS', '3_layer_exists', '3_layer_thickness', '3_type_AC', '3_type_EF', '3_type_GB', '3_type_GS', '3_type_PC', '3_type_SS', '3_type_TB', '3_type_TS', '4_layer_exists', '4_layer_thickness', '4_type_AC', '4_type_EF', '4_type_GB', '4_type_GS', '4_type_PC', '4_type_SS',
                     '4_type_TB', '4_type_TS', '5_layer_exists', '5_layer_thickness', '5_type_AC', '5_type_EF', '5_type_GB', '5_type_GS', '5_type_PC', '5_type_SS', '5_type_TB', '5_type_TS', '6_layer_exists', '6_layer_thickness', '6_type_AC',
                     '6_type_EF', '6_type_GB', '6_type_GS', '6_type_PC', '6_type_SS', '6_type_TB', '6_type_TS', '7_layer_exists', '7_layer_thickness', '7_type_AC', '7_type_EF', '7_type_GB', '7_type_GS', '7_type_PC', '7_type_SS', '7_type_TB', '7_type_TS', '8_layer_exists', '8_layer_thickness', '8_type_AC', '8_type_EF', '8_type_GB', '8_type_GS', '8_type_PC',
                     '8_type_SS', '8_type_TB', '8_type_TS', '9_layer_exists', '9_layer_thickness', '9_type_AC', '9_type_EF', '9_type_GB', '9_type_GS', '9_type_PC', '9_type_SS', '9_type_TB', '9_type_TS', '10_layer_exists', '10_layer_thickness', '10_type_AC', '10_type_EF', '10_type_GB', '10_type_GS', '10_type_PC', '10_type_SS', '10_type_TB', '10_type_TS', '11_layer_exists', '11_layer_thickness', '11_type_AC', '11_type_EF', '11_type_GB', '11_type_GS', '11_type_PC', '11_type_SS', '11_type_TB', '11_type_TS', '12_layer_exists', '12_layer_thickness', '12_type_AC', '12_type_EF', '12_type_GB', '2_type_GS', '12_type_PC', '12_type_SS', '12_type_TB', '12_type_TS', '13_layer_exists', '13_layer_thickness', '13_type_AC', '13_type_EF', '13_type_GB', '13_type_GS', '13_type_PC', '13_type_SS', '13_type_TB', '13_type_TS', '14_layer_exists', '14_layer_thickness', '14_type_AC', '14_type_EF', '14_type_GB', '14_type_GS', '14_type_PC', '14_type_SS', '14_type_TB', '14_type_TS'
                     # cliamte
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
    pca = PCA(n_components=5, whiten=True, random_state=42)
    XList = np.concatenate([np.array(using.values[:, 1:9]), pca.fit_transform(np.array(using.values[:, 9:149]))], axis=1)
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
                if str(data[i][149+k*13]) != '-100':
                    middle_list.append(float(data[i][149+j+k*13]))
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

    return XList, yList #1738




train, test = featureSet()
# train的前两列是x，后一列是y，这里的y有随机噪声
# train, test = only_climate()

x_train, y_train = train[:-500], test[:-500]
x_test, y_test = train[-500:], test[-500:]  # 同上，但这里的y没有噪声
print(np.array(x_train).shape)
print(np.array(y_train).shape)
y_train = np.array(y_train).reshape((len(y_train))).tolist()
y_test = np.array(y_test).reshape((len(y_test))).tolist()


# 回归部分
def try_different_method(model, method):




    X = np.array([[x] for x in range(0, 2200)])

    Y = np.array([x for x in range(0, 2200)])

    floder = KFold(n_splits=5, random_state=0, shuffle=False)

    # for train, test in floder.split(X, y):
    #     print('Train: %s | test: %s' % (train, test))
    #     print(" ")

    cvscores = []
    r_2_list = []
    rmse_list = []

    for (train_num, test_num) in floder.split(X, Y):
        model.fit(train[train_num], test[train_num])
        result = model.predict(train[test_num])
        rmse = sqrt(mean_squared_error(test[test_num], result))
        rmse_list.append(rmse)
        # print('Test RMSE: %.3f' % rmse)
        r_2 = r2_score(test[test_num], result)
        r_2_list.append(r_2)
        # print('Test R^2: %.3f' % r_2)
    print("R^2 = ", '[', format(r_2_list[0], '.3f'), format(r_2_list[1], '.3f'), format(r_2_list[2], '.3f'), format(r_2_list[3], '.3f'), format(r_2_list[4], '.3f'), ']')
    print("mean: ", format(np.array(r_2_list).mean(), '.3f'))
    print("std: ", format(np.array(r_2_list).std(ddof=1), '.3f'))
    print("RMSE = ", '[', format(rmse_list[0], '.3f'), format(rmse_list[1], '.3f'), format(rmse_list[2], '.3f'), format(rmse_list[3], '.3f'), format(rmse_list[4], '.3f'), ']')
    print("mean: ", format(np.array(rmse_list).mean(), '.3f'))
    print("std: ", format(np.array(rmse_list).std(ddof=1), '.3f'))









    # for i in range(5):
    #     print(np.array(x_train).shape)
    #     print(np.array(y_train).shape)
    #     model.fit(x_train, y_train)
    #     score = model.score(x_test, y_test)
    #     # print(score)
    #     result = model.predict(x_test)
    #     rmse = sqrt(mean_squared_error(y_test, result))
    #     print('Test RMSE: %.3f' % rmse)
    #     # print("length of test data: ", len(y_test))
    #     r_2 = r2_score(y_test, result)
    #     print('Test R^2: %.3f' % r_2)
    #     print('\n')


        # importances = model.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in model.estimators_],
        #      axis=0)
        # indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        # print("Feature ranking:")
        #
        # for f in range(x_train.shape[1]):
        #     print("%d. feature x%d (%f)" % (f + 1, indices[f] + 1, importances[indices[f]]))



    result = model.predict(x_test)
    x_1 = [x/100 for x in range(1, 500)]
    y_1 = [x/100 for x in range(1, 500)]
    rmse = sqrt(mean_squared_error(y_test, result))
    print('Test RMSE: %.3f' % rmse)
    print("length of test data: ", len(y_test))
    r_2 = r2_score(y_test, result)
    print('Test R^2: %.3f' % r_2)


    plt.figure(figsize=(4.4, 4.0))
    plt.plot(y_test, result, "o", markersize=2.)
    print(y_test)
    print(result)
    plt.plot(x_1, y_1, color='red', label='Predicted IRI = Real IRI')
    plt.xlabel("Real IRI")
    plt.ylabel("Predicted IRI")
    plt.text(4, 0, r'$R^{2} = %.3f$' % 0.867)  # 文本中注释
    plt.legend()
    plt.show()


    # plt.figure()
    # plt.plot(np.arange(len(result)), y_test, "go-", label="True value")
    # plt.plot(np.arange(len(result)), result, "ro-", label="Predict value")
    # plt.title("method: %s ---score: %s" % (method, score))
    # plt.legend(loc="best")
    # plt.show()

    # plt.figure(1)
    # ax1 = plt.subplot(211)
    # ax2 = plt.subplot(212)
    #
    # plt.sca(ax1)
    # plt.plot(np.arange(len(result)), y_test, "go-", label="True value")
    # plt.plot(np.arange(len(result)), result, "ro-", label="Predict value")
    # plt.title("%s: R^2: %s, MSE: %s" % (method, score, mean_squared_error(y_test, result)))
    # plt.legend(loc="best")
    #
    # plt.sca(ax2)
    # plt.plot(y_test, result, "o")
    # plt.plot([0, 1, 2, 3, 4, 5], linestyle='-')
    # # plt.legend([score, mean_squared_error(y_test, result)], loc="best")
    # plt.xlabel("Real value")
    # plt.ylabel("Predicted value")
    #
    # plt.show()

# ["R^2:", "MSE:"],



# 方法选择-lgbm、xgboost
# 1.决策树回归
from sklearn import tree

model_decision_tree_regression = tree.DecisionTreeRegressor()

# 2.线性回归
from sklearn.linear_model import LinearRegression

model_linear_regression = LinearRegression()

# 3.SVM回归
from sklearn import svm

model_svm = svm.SVR()

# 4.kNN回归
from sklearn import neighbors

model_k_neighbor = neighbors.KNeighborsRegressor()

# 5.随机森林回归
from sklearn import ensemble

model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=500)  # 使用20个决策树


# 6.Adaboost回归
from sklearn import ensemble

model_adaboost_regressor = ensemble.AdaBoostRegressor(n_estimators=500)  # 这里使用50个决策树

# 7.GBRT回归
from sklearn import ensemble

model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor(n_estimators=400)  # 这里使用100个决策树

# 8.Bagging回归
from sklearn import ensemble

model_bagging_regressor = ensemble.BaggingRegressor(n_estimators=500)

# 9.ExtraTree极端随机数回归
from sklearn.tree import ExtraTreeRegressor

model_extra_tree_regressor = ExtraTreeRegressor()




try_different_method(model_gradient_boosting_regressor, "DecisionTree")


# try_different_method(model_k_neighbor, "RandomForest")