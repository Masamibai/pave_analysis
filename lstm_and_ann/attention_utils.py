import keras.backend as K
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA










def get_activations(model, inputs_1, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    print(inp)
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp[0]] + [inp[1]] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    print(type(funcs))
    print(funcs)

    layer_outputs = [func([inputs_1[0], inputs_1[1], 1.])[0] for func in funcs]

    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


# def get_data(n, input_dim, attention_column=1):
#     """
#     Data generation. x is purely random except that it's first value equals the target y.
#     In practice, the network should learn that the target = x[attention_column].
#     Therefore, most of its attention should be focused on the value addressed by attention_column.
#     :param n: the number of samples to retrieve.
#     :param input_dim: the number of dimensions of each element in the series.
#     :param attention_column: the column linked to the target. Everything else is purely random.
#     :return: x: model inputs, y: model targets
#     """
#     x = np.random.standard_normal(size=(n, input_dim))
#     y = np.random.randint(low=0, high=2, size=(n, 1))
#     x[:, attention_column] = y[:, 0]
#     return x, y




def get_data_recurrent_ANN(n, time_steps, input_dim, NN_FEATURES, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    # x = np.random.standard_normal(size=(n, time_steps, input_dim))
    # y = np.random.randint(low=0, high=2, size=(n, 1))
    # x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    # nn_input = np.random.standard_normal(size=(n, NN_FEATURES))

    # dataset = read_csv('data_using_last_MRI_and_days.csv', header=0, index_col=0)
    # # using = dataset[['MRI', 'MRI_LAST_2', 'MRI_LAST_2', 'MRI_LAST_2', 'DAYS_AWAY_2', 'DAYS_AWAY_2'
    # #                 , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61'
    # #                 , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
    # #                 , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
    # #                 , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
    # #                 , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
    # #                 , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
    # #                 , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
    # #                 , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
    # #                 , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
    # #                 , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
    # #                 , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
    # #                 , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
    # #                 , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
    # #                 , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
    # #                 , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
    # #                 , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
    # #                 , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
    # #                 , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
    # #                 , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
    # #                 , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99']]
    # # using = dataset[['MRI', 'MRI_LAST_2', 'MRI_LAST_2', 'MRI_LAST_2', 'DAYS_AWAY_2', 'DAYS_AWAY_2'
    # using = dataset[['MRI', 'RMSVA_8', 'RMSVA_16', 'RMSVA_32', 'RMSVA_64', 'RMSVA_128'
    #                  #本来的
    #                 , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61'
    #                 , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
    #                 , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
    #                 , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
    #                 , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
    #                 , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
    #                 , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
    #                 , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
    #                 , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
    #                 , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
    #                 , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
    #                 , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
    #                 , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
    #                 , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
    #                 , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
    #                 , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
    #                 , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
    #                 , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
    #                 , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
    #                 , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99']]
    #
    #                  # 反过来的
    #                 # , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99'
    #                 # , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
    #                 # , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
    #                 # , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
    #                 # , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
    #                 # , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
    #                 # , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
    #                 # , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
    #                 # , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
    #                 # , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
    #                 # , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
    #                 # , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
    #                 # , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
    #                 # , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
    #                 # , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
    #                 # , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
    #                 # , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
    #                 # , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
    #                 # , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
    #                 # , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61']]

    # TRAIL_4_18 trail_version_DELTA_2
    # MRI,,,,,,,,,,,,,,,,,,,,,,,,,,,TOTAL_PRECIP_LAST_11,TOTAL_SNOWFALL_LAST_11,INTENSE_PRECIP_DAYS_LAST_11,WET_DAYS_LAST_11,TOTAL_PRECIP_LAST_12,TOTAL_SNOWFALL_LAST_12,INTENSE_PRECIP_DAYS_LAST_12,WET_DAYS_LAST_12,TOTAL_PRECIP_LAST_13,TOTAL_SNOWFALL_LAST_13,INTENSE_PRECIP_DAYS_LAST_13,WET_DAYS_LAST_13,TOTAL_PRECIP_LAST_14,TOTAL_SNOWFALL_LAST_14,INTENSE_PRECIP_DAYS_LAST_14,WET_DAYS_LAST_14,TOTAL_PRECIP_LAST_15,TOTAL_SNOWFALL_LAST_15,INTENSE_PRECIP_DAYS_LAST_15,WET_DAYS_LAST_15,TOTAL_PRECIP_LAST_16,TOTAL_SNOWFALL_LAST_16,INTENSE_PRECIP_DAYS_LAST_16,WET_DAYS_LAST_16,TOTAL_PRECIP_LAST_17,TOTAL_SNOWFALL_LAST_17,INTENSE_PRECIP_DAYS_LAST_17,WET_DAYS_LAST_17,TOTAL_PRECIP_LAST_18,TOTAL_SNOWFALL_LAST_18,INTENSE_PRECIP_DAYS_LAST_18,WET_DAYS_LAST_18,TOTAL_PRECIP_LAST_19,TOTAL_SNOWFALL_LAST_19,INTENSE_PRECIP_DAYS_LAST_19,WET_DAYS_LAST_19,TOTAL_PRECIP_LAST_20,TOTAL_SNOWFALL_LAST_20,INTENSE_PRECIP_DAYS_LAST_20,WET_DAYS_LAST_20

    # dataset = read_csv('trail_version_DELTA_2.csv', header=0, index_col=0)
    dataset = read_csv('final_version_structure.csv', header=0, index_col=0)
    dataset.dropna(axis=0, how='any', inplace=True)
    using = dataset[['MRI', 'MRI0', 'DAYS_BETWEEN', 'LAT', 'LONGT', 'ELEV', 'AGE_TO_MRI_MEASURE', 'TRUCK_TRAFFIC', 'KESAL_TRAFFIC'
                     ,'1_layer_exists', '1_layer_thickness', '1_type_AC', '1_type_EF', '1_type_GB', '1_type_GS', '1_type_PC', '1_type_SS', '1_type_TB', '1_type_TS', '2_layer_exists', '2_layer_thickness', '2_type_AC', '2_type_EF', '2_type_GB',
                     '2_type_GS', '2_type_PC', '2_type_SS', '2_type_TB', '2_type_TS', '3_layer_exists', '3_layer_thickness', '3_type_AC', '3_type_EF', '3_type_GB', '3_type_GS', '3_type_PC', '3_type_SS', '3_type_TB', '3_type_TS', '4_layer_exists', '4_layer_thickness', '4_type_AC', '4_type_EF', '4_type_GB', '4_type_GS', '4_type_PC', '4_type_SS',
                     '4_type_TB', '4_type_TS', '5_layer_exists', '5_layer_thickness', '5_type_AC', '5_type_EF', '5_type_GB', '5_type_GS', '5_type_PC', '5_type_SS', '5_type_TB', '5_type_TS', '6_layer_exists', '6_layer_thickness', '6_type_AC',
                     '6_type_EF', '6_type_GB', '6_type_GS', '6_type_PC', '6_type_SS', '6_type_TB', '6_type_TS', '7_layer_exists', '7_layer_thickness', '7_type_AC', '7_type_EF', '7_type_GB', '7_type_GS', '7_type_PC', '7_type_SS', '7_type_TB', '7_type_TS', '8_layer_exists', '8_layer_thickness', '8_type_AC', '8_type_EF', '8_type_GB', '8_type_GS', '8_type_PC',
                     '8_type_SS', '8_type_TB', '8_type_TS', '9_layer_exists', '9_layer_thickness', '9_type_AC', '9_type_EF', '9_type_GB', '9_type_GS', '9_type_PC', '9_type_SS', '9_type_TB', '9_type_TS', '10_layer_exists', '10_layer_thickness', '10_type_AC', '10_type_EF', '10_type_GB', '10_type_GS', '10_type_PC', '10_type_SS', '10_type_TB', '10_type_TS', '11_layer_exists', '11_layer_thickness', '11_type_AC', '11_type_EF', '11_type_GB', '11_type_GS', '11_type_PC', '11_type_SS', '11_type_TB', '11_type_TS', '12_layer_exists', '12_layer_thickness', '12_type_AC', '12_type_EF', '12_type_GB', '2_type_GS', '12_type_PC', '12_type_SS', '12_type_TB', '12_type_TS', '13_layer_exists', '13_layer_thickness', '13_type_AC', '13_type_EF', '13_type_GB', '13_type_GS', '13_type_PC', '13_type_SS', '13_type_TB', '13_type_TS', '14_layer_exists', '14_layer_thickness', '14_type_AC', '14_type_EF', '14_type_GB', '14_type_GS', '14_type_PC', '14_type_SS', '14_type_TB', '14_type_TS'
                     # #cliamte
                     # , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'
                     # , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     # , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     # , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     # , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     # , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     # , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     # , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     # , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     # , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'

                     # cliamte_reverse_for_right_time_series
                     , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'
                     , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'

                     ]]

                     # 反过来的
                    # , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99'
                    # , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
                    # , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
                    # , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
                    # , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
                    # , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
                    # , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
                    # , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
                    # , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
                    # , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
                    # , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
                    # , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
                    # , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
                    # , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
                    # , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
                    # , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
                    # , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
                    # , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
                    # , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
                    # , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61']]







    # missing_val_count_by_column = (using.isnull().sum())
    # print(missing_val_count_by_column)
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])


    values_3 = using.values
    values_3 = values_3
    values_3 = shuffle(values_3)
    # ensure all data is float
    values_3 = values_3.astype('float32')
    # # normalize features
    # scaler_3 = MinMaxScaler(feature_range=(-1, 1))
    # scaled_3 = scaler_3.fit_transform(values_3)
    # values_3 = scaled_3
    # x_1 = np.array(values_3[:, 149:]).reshape((values_3.shape[0], 10, 13))
    # y = np.array(values_3[:, 0])
    #
    # # PCA降维
    # pca = PCA(n_components=5, whiten=True, random_state=42)
    # nn_input = np.concatenate([np.array(values_3[:, 1:9]), pca.fit_transform(np.array(values_3[:, 9:149]))], axis=1)




    ##############################################
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
    print(list_features)
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
    print(analysis_features)
    print(len(analysis_features))

    # PCA降维
    pca = PCA(n_components=5, whiten=True, random_state=42)
    # y, nn input: 8+5+26,(气候前有40列)
    values_3 = np.concatenate([np.array(values_3[:, 0:9]), pca.fit_transform(np.array(values_3[:, 9:149])), analysis_features], axis=1)

    print(np.array(values_3).shape)
    # nn_input增加了26列（13个10年每个都浓缩为两个：均值、方差）
    # normalize features
    scaler_3 = MinMaxScaler(feature_range=(-1, 1))
    scaled_3 = scaler_3.fit_transform(values_3)
    values_3 = scaled_3
    # x_1 = np.array(values_3[:, 149:]).reshape((values_3.shape[0], 10, 13))
    # y = np.array(values_3[:, 0])
    # x_1 = np.array(values_3[:, 40:]).reshape((values_3.shape[0], 10, 13))
    nn_input = np.array(values_3[:, 1: NN_FEATURES + 1])
    y = np.array(values_3[:, 0])

    ##############################################


    # nn feature = 维度 + 8



    return nn_input, y, np.array(values_3[:, 1: ]), scaler_3







def get_data_recurrent(n, time_steps, input_dim, NN_FEATURES, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    # x = np.random.standard_normal(size=(n, time_steps, input_dim))
    # y = np.random.randint(low=0, high=2, size=(n, 1))
    # x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    # nn_input = np.random.standard_normal(size=(n, NN_FEATURES))

    # dataset = read_csv('data_using_last_MRI_and_days.csv', header=0, index_col=0)
    # # using = dataset[['MRI', 'MRI_LAST_2', 'MRI_LAST_2', 'MRI_LAST_2', 'DAYS_AWAY_2', 'DAYS_AWAY_2'
    # #                 , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61'
    # #                 , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
    # #                 , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
    # #                 , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
    # #                 , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
    # #                 , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
    # #                 , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
    # #                 , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
    # #                 , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
    # #                 , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
    # #                 , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
    # #                 , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
    # #                 , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
    # #                 , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
    # #                 , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
    # #                 , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
    # #                 , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
    # #                 , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
    # #                 , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
    # #                 , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99']]
    # # using = dataset[['MRI', 'MRI_LAST_2', 'MRI_LAST_2', 'MRI_LAST_2', 'DAYS_AWAY_2', 'DAYS_AWAY_2'
    # using = dataset[['MRI', 'RMSVA_8', 'RMSVA_16', 'RMSVA_32', 'RMSVA_64', 'RMSVA_128'
    #                  #本来的
    #                 , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61'
    #                 , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
    #                 , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
    #                 , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
    #                 , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
    #                 , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
    #                 , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
    #                 , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
    #                 , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
    #                 , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
    #                 , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
    #                 , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
    #                 , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
    #                 , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
    #                 , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
    #                 , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
    #                 , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
    #                 , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
    #                 , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
    #                 , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99']]
    #
    #                  # 反过来的
    #                 # , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99'
    #                 # , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
    #                 # , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
    #                 # , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
    #                 # , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
    #                 # , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
    #                 # , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
    #                 # , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
    #                 # , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
    #                 # , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
    #                 # , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
    #                 # , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
    #                 # , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
    #                 # , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
    #                 # , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
    #                 # , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
    #                 # , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
    #                 # , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
    #                 # , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
    #                 # , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61']]

    # TRAIL_4_18 trail_version_DELTA_2
    # MRI,,,,,,,,,,,,,,,,,,,,,,,,,,,TOTAL_PRECIP_LAST_11,TOTAL_SNOWFALL_LAST_11,INTENSE_PRECIP_DAYS_LAST_11,WET_DAYS_LAST_11,TOTAL_PRECIP_LAST_12,TOTAL_SNOWFALL_LAST_12,INTENSE_PRECIP_DAYS_LAST_12,WET_DAYS_LAST_12,TOTAL_PRECIP_LAST_13,TOTAL_SNOWFALL_LAST_13,INTENSE_PRECIP_DAYS_LAST_13,WET_DAYS_LAST_13,TOTAL_PRECIP_LAST_14,TOTAL_SNOWFALL_LAST_14,INTENSE_PRECIP_DAYS_LAST_14,WET_DAYS_LAST_14,TOTAL_PRECIP_LAST_15,TOTAL_SNOWFALL_LAST_15,INTENSE_PRECIP_DAYS_LAST_15,WET_DAYS_LAST_15,TOTAL_PRECIP_LAST_16,TOTAL_SNOWFALL_LAST_16,INTENSE_PRECIP_DAYS_LAST_16,WET_DAYS_LAST_16,TOTAL_PRECIP_LAST_17,TOTAL_SNOWFALL_LAST_17,INTENSE_PRECIP_DAYS_LAST_17,WET_DAYS_LAST_17,TOTAL_PRECIP_LAST_18,TOTAL_SNOWFALL_LAST_18,INTENSE_PRECIP_DAYS_LAST_18,WET_DAYS_LAST_18,TOTAL_PRECIP_LAST_19,TOTAL_SNOWFALL_LAST_19,INTENSE_PRECIP_DAYS_LAST_19,WET_DAYS_LAST_19,TOTAL_PRECIP_LAST_20,TOTAL_SNOWFALL_LAST_20,INTENSE_PRECIP_DAYS_LAST_20,WET_DAYS_LAST_20

    # dataset = read_csv('trail_version_DELTA_2.csv', header=0, index_col=0)
    dataset = read_csv('final_version_structure_5_22.csv', header=0, index_col=0)
    dataset.dropna(axis=0, how='any', inplace=True)
    using = dataset[['MRI', 'MRI0', 'DAYS_BETWEEN', 'LAT', 'LONGT', 'ELEV', 'AGE_TO_MRI_MEASURE', 'TRUCK_TRAFFIC', 'KESAL_TRAFFIC'
                     ,'1_layer_exists', '1_layer_thickness', '1_type_AC', '1_type_EF', '1_type_GB', '1_type_GS', '1_type_PC', '1_type_SS', '1_type_TB', '1_type_TS', '2_layer_exists', '2_layer_thickness', '2_type_AC', '2_type_EF', '2_type_GB',
                     '2_type_GS', '2_type_PC', '2_type_SS', '2_type_TB', '2_type_TS', '3_layer_exists', '3_layer_thickness', '3_type_AC', '3_type_EF', '3_type_GB', '3_type_GS', '3_type_PC', '3_type_SS', '3_type_TB', '3_type_TS', '4_layer_exists', '4_layer_thickness', '4_type_AC', '4_type_EF', '4_type_GB', '4_type_GS', '4_type_PC', '4_type_SS',
                     '4_type_TB', '4_type_TS', '5_layer_exists', '5_layer_thickness', '5_type_AC', '5_type_EF', '5_type_GB', '5_type_GS', '5_type_PC', '5_type_SS', '5_type_TB', '5_type_TS', '6_layer_exists', '6_layer_thickness', '6_type_AC',
                     '6_type_EF', '6_type_GB', '6_type_GS', '6_type_PC', '6_type_SS', '6_type_TB', '6_type_TS', '7_layer_exists', '7_layer_thickness', '7_type_AC', '7_type_EF', '7_type_GB', '7_type_GS', '7_type_PC', '7_type_SS', '7_type_TB', '7_type_TS', '8_layer_exists', '8_layer_thickness', '8_type_AC', '8_type_EF', '8_type_GB', '8_type_GS', '8_type_PC',
                     '8_type_SS', '8_type_TB', '8_type_TS', '9_layer_exists', '9_layer_thickness', '9_type_AC', '9_type_EF', '9_type_GB', '9_type_GS', '9_type_PC', '9_type_SS', '9_type_TB', '9_type_TS', '10_layer_exists', '10_layer_thickness', '10_type_AC', '10_type_EF', '10_type_GB', '10_type_GS', '10_type_PC', '10_type_SS', '10_type_TB', '10_type_TS', '11_layer_exists', '11_layer_thickness', '11_type_AC', '11_type_EF', '11_type_GB', '11_type_GS', '11_type_PC', '11_type_SS', '11_type_TB', '11_type_TS', '12_layer_exists', '12_layer_thickness', '12_type_AC', '12_type_EF', '12_type_GB', '2_type_GS', '12_type_PC', '12_type_SS', '12_type_TB', '12_type_TS', '13_layer_exists', '13_layer_thickness', '13_type_AC', '13_type_EF', '13_type_GB', '13_type_GS', '13_type_PC', '13_type_SS', '13_type_TB', '13_type_TS', '14_layer_exists', '14_layer_thickness', '14_type_AC', '14_type_EF', '14_type_GB', '14_type_GS', '14_type_PC', '14_type_SS', '14_type_TB', '14_type_TS'
                     # #cliamte
                     # , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'
                     # , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     # , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     # , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     # , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     # , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     # , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     # , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     # , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     # , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'

                     # cliamte_reverse_for_right_time_series
                     , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10'
                     , 'MEAN_TEMP_AVG_LAST_9','FREEZE_INDEX_LAST_9','FREEZE_THAW_LAST_9','MAX_TEMP_AVG_LAST_9','MIN_TEMP_AVG_LAST_9','MAX_TEMP_LAST_9','MIN_TEMP_LAST_9','DAYS_ABOVE_32_LAST_9','DAYS_BELOW_0_LAST_9','TOTAL_PRECIP_LAST_9','TOTAL_SNOWFALL_LAST_9','INTENSE_PRECIP_DAYS_LAST_9','WET_DAYS_LAST_9'
                     , 'MEAN_TEMP_AVG_LAST_8','FREEZE_INDEX_LAST_8','FREEZE_THAW_LAST_8','MAX_TEMP_AVG_LAST_8','MIN_TEMP_AVG_LAST_8','MAX_TEMP_LAST_8','MIN_TEMP_LAST_8','DAYS_ABOVE_32_LAST_8','DAYS_BELOW_0_LAST_8','TOTAL_PRECIP_LAST_8','TOTAL_SNOWFALL_LAST_8','INTENSE_PRECIP_DAYS_LAST_8','WET_DAYS_LAST_8'
                     , 'MEAN_TEMP_AVG_LAST_7','FREEZE_INDEX_LAST_7','FREEZE_THAW_LAST_7','MAX_TEMP_AVG_LAST_7','MIN_TEMP_AVG_LAST_7','MAX_TEMP_LAST_7','MIN_TEMP_LAST_7','DAYS_ABOVE_32_LAST_7','DAYS_BELOW_0_LAST_7','TOTAL_PRECIP_LAST_7','TOTAL_SNOWFALL_LAST_7','INTENSE_PRECIP_DAYS_LAST_7','WET_DAYS_LAST_7'
                     , 'MEAN_TEMP_AVG_LAST_6','FREEZE_INDEX_LAST_6','FREEZE_THAW_LAST_6','MAX_TEMP_AVG_LAST_6','MIN_TEMP_AVG_LAST_6','MAX_TEMP_LAST_6','MIN_TEMP_LAST_6','DAYS_ABOVE_32_LAST_6','DAYS_BELOW_0_LAST_6','TOTAL_PRECIP_LAST_6','TOTAL_SNOWFALL_LAST_6','INTENSE_PRECIP_DAYS_LAST_6','WET_DAYS_LAST_6'
                     , 'MEAN_TEMP_AVG_LAST_5','FREEZE_INDEX_LAST_5','FREEZE_THAW_LAST_5','MAX_TEMP_AVG_LAST_5','MIN_TEMP_AVG_LAST_5','MAX_TEMP_LAST_5','MIN_TEMP_LAST_5','DAYS_ABOVE_32_LAST_5','DAYS_BELOW_0_LAST_5','TOTAL_PRECIP_LAST_5','TOTAL_SNOWFALL_LAST_5','INTENSE_PRECIP_DAYS_LAST_5','WET_DAYS_LAST_5'
                     , 'MEAN_TEMP_AVG_LAST_4','FREEZE_INDEX_LAST_4','FREEZE_THAW_LAST_4','MAX_TEMP_AVG_LAST_4','MIN_TEMP_AVG_LAST_4','MAX_TEMP_LAST_4','MIN_TEMP_LAST_4','DAYS_ABOVE_32_LAST_4','DAYS_BELOW_0_LAST_4','TOTAL_PRECIP_LAST_4','TOTAL_SNOWFALL_LAST_4','INTENSE_PRECIP_DAYS_LAST_4','WET_DAYS_LAST_4'
                     , 'MEAN_TEMP_AVG_LAST_3','FREEZE_INDEX_LAST_3','FREEZE_THAW_LAST_3','MAX_TEMP_AVG_LAST_3','MIN_TEMP_AVG_LAST_3','MAX_TEMP_LAST_3','MIN_TEMP_LAST_3','DAYS_ABOVE_32_LAST_3','DAYS_BELOW_0_LAST_3','TOTAL_PRECIP_LAST_3','TOTAL_SNOWFALL_LAST_3','INTENSE_PRECIP_DAYS_LAST_3','WET_DAYS_LAST_3'
                     , 'MEAN_TEMP_AVG_LAST_2','FREEZE_INDEX_LAST_2','FREEZE_THAW_LAST_2','MAX_TEMP_AVG_LAST_2','MIN_TEMP_AVG_LAST_2','MAX_TEMP_LAST_2','MIN_TEMP_LAST_2','DAYS_ABOVE_32_LAST_2','DAYS_BELOW_0_LAST_2','TOTAL_PRECIP_LAST_2','TOTAL_SNOWFALL_LAST_2','INTENSE_PRECIP_DAYS_LAST_2','WET_DAYS_LAST_2'
                     , 'MEAN_TEMP_AVG_LAST_1','FREEZE_INDEX_LAST_1','FREEZE_THAW_LAST_1','MAX_TEMP_AVG_LAST_1','MIN_TEMP_AVG_LAST_1','MAX_TEMP_LAST_1','MIN_TEMP_LAST_1','DAYS_ABOVE_32_LAST_1','DAYS_BELOW_0_LAST_1','TOTAL_PRECIP_LAST_1','TOTAL_SNOWFALL_LAST_1','INTENSE_PRECIP_DAYS_LAST_1','WET_DAYS_LAST_1'

                     ]]

                     # 反过来的
                    # , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99'
                    # , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
                    # , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
                    # , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
                    # , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
                    # , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
                    # , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
                    # , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
                    # , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
                    # , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
                    # , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
                    # , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
                    # , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
                    # , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
                    # , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
                    # , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
                    # , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
                    # , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
                    # , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
                    # , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61']]







    # missing_val_count_by_column = (using.isnull().sum())
    # print(missing_val_count_by_column)
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])


    values_3 = using.values
    values_3 = values_3
    values_3 = shuffle(values_3)
    # ensure all data is float
    values_3 = values_3.astype('float32')
    # # normalize features
    # scaler_3 = MinMaxScaler(feature_range=(-1, 1))
    # scaled_3 = scaler_3.fit_transform(values_3)
    # values_3 = scaled_3
    # x_1 = np.array(values_3[:, 149:]).reshape((values_3.shape[0], 10, 13))
    # y = np.array(values_3[:, 0])
    #
    # # PCA降维
    # pca = PCA(n_components=5, whiten=True, random_state=42)
    # nn_input = np.concatenate([np.array(values_3[:, 1:9]), pca.fit_transform(np.array(values_3[:, 9:149]))], axis=1)




    ##############################################
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
    print(list_features)
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
    print(analysis_features)
    print(len(analysis_features))

    # PCA降维
    pca = PCA(n_components=5, whiten=True, random_state=42)
    # y, nn input: 8+5+26,(气候前有40列)
    values_3 = np.concatenate([np.array(values_3[:, 0:9]), pca.fit_transform(np.array(values_3[:, 9:149])), analysis_features, np.array(values_3[:, 149:])], axis=1)

    print(np.array(values_3).shape)
    # nn_input增加了26列（13个10年每个都浓缩为两个：均值、方差）
    # normalize features
    scaler_3 = MinMaxScaler(feature_range=(-1, 1))
    scaled_3 = scaler_3.fit_transform(values_3)
    values_3 = scaled_3
    # x_1 = np.array(values_3[:, 149:]).reshape((values_3.shape[0], 10, 13))
    # y = np.array(values_3[:, 0])
    x_1 = np.array(values_3[:, 40:]).reshape((values_3.shape[0], 10, 13))
    nn_input = np.array(values_3[:, 1:NN_FEATURES + 1])
    y = np.array(values_3[:, 0])

    ##############################################


    # nn feature = 维度 + 8



    return x_1, nn_input, y, np.array(values_3[:, 1:]), scaler_3


def get_data_recurrent_1(n, count, time_steps, input_dim, NN_FEATURES, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    # x = np.random.standard_normal(size=(n, time_steps, input_dim))
    # y = np.random.randint(low=0, high=2, size=(n, 1))
    # x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    # nn_input = np.random.standard_normal(size=(n, NN_FEATURES))

    # dataset = read_csv('trail_version_DELTA_2.csv', header=0, index_col=0)
    # using = dataset[['MRI', 'RMSVA_32', 'RMSVA_64', 'RMSVA_128', 'RMSVA_8', 'RMSVA_16'
    #                 , 'FREEZE_THAW_YR_21', 'INTENSE_PRECIP_DAYS_YR_61'
    #                 , 'FREEZE_THAW_YR_23', 'INTENSE_PRECIP_DAYS_YR_63'
    #                 , 'FREEZE_THAW_YR_25', 'INTENSE_PRECIP_DAYS_YR_65'
    #                 , 'FREEZE_THAW_YR_27', 'INTENSE_PRECIP_DAYS_YR_67'
    #                 , 'FREEZE_THAW_YR_29', 'INTENSE_PRECIP_DAYS_YR_69'
    #                 , 'FREEZE_THAW_YR_31', 'INTENSE_PRECIP_DAYS_YR_71'
    #                 , 'FREEZE_THAW_YR_33', 'INTENSE_PRECIP_DAYS_YR_73'
    #                 , 'FREEZE_THAW_YR_35', 'INTENSE_PRECIP_DAYS_YR_75'
    #                 , 'FREEZE_THAW_YR_37', 'INTENSE_PRECIP_DAYS_YR_77'
    #                 , 'FREEZE_THAW_YR_39', 'INTENSE_PRECIP_DAYS_YR_79'
    #                 , 'FREEZE_THAW_YR_41', 'INTENSE_PRECIP_DAYS_YR_81'
    #                 , 'FREEZE_THAW_YR_43', 'INTENSE_PRECIP_DAYS_YR_83'
    #                 , 'FREEZE_THAW_YR_45', 'INTENSE_PRECIP_DAYS_YR_85'
    #                 , 'FREEZE_THAW_YR_47', 'INTENSE_PRECIP_DAYS_YR_87'
    #                 , 'FREEZE_THAW_YR_49', 'INTENSE_PRECIP_DAYS_YR_89'
    #                 , 'FREEZE_THAW_YR_51', 'INTENSE_PRECIP_DAYS_YR_91'
    #                 , 'FREEZE_THAW_YR_53', 'INTENSE_PRECIP_DAYS_YR_93'
    #                 , 'FREEZE_THAW_YR_55', 'INTENSE_PRECIP_DAYS_YR_95'
    #                 , 'FREEZE_THAW_YR_57', 'INTENSE_PRECIP_DAYS_YR_97'
    #                 , 'FREEZE_THAW_YR_59', 'INTENSE_PRECIP_DAYS_YR_99']]

    dataset = read_csv('final_version_structure.csv', header=0, index_col=0)
    using = dataset[[ 'MRI', 'MRI0', 'DAYS_BETWEEN', 'LAT', 'LONGT', 'ELEV', 'AGE_TO_MRI_MEASURE', 'TRUCK_TRAFFIC', 'KESAL_TRAFFIC'
        , '1_layer_exists', '1_layer_thickness', '1_type_AC', '1_type_EF', '1_type_GB', '1_type_GS', '1_type_PC',
                      '1_type_SS', '1_type_TB', '1_type_TS', '2_layer_exists', '2_layer_thickness', '2_type_AC',
                      '2_type_EF', '2_type_GB',
                      '2_type_GS', '2_type_PC', '2_type_SS', '2_type_TB', '2_type_TS', '3_layer_exists',
                      '3_layer_thickness', '3_type_AC', '3_type_EF', '3_type_GB', '3_type_GS', '3_type_PC', '3_type_SS',
                      '3_type_TB', '3_type_TS', '4_layer_exists', '4_layer_thickness', '4_type_AC', '4_type_EF',
                      '4_type_GB', '4_type_GS', '4_type_PC', '4_type_SS',
                      '4_type_TB', '4_type_TS', '5_layer_exists', '5_layer_thickness', '5_type_AC', '5_type_EF',
                      '5_type_GB', '5_type_GS', '5_type_PC', '5_type_SS', '5_type_TB', '5_type_TS', '6_layer_exists',
                      '6_layer_thickness', '6_type_AC',
                      '6_type_EF', '6_type_GB', '6_type_GS', '6_type_PC', '6_type_SS', '6_type_TB', '6_type_TS',
                      '7_layer_exists', '7_layer_thickness', '7_type_AC', '7_type_EF', '7_type_GB', '7_type_GS',
                      '7_type_PC', '7_type_SS', '7_type_TB', '7_type_TS', '8_layer_exists', '8_layer_thickness',
                      '8_type_AC', '8_type_EF', '8_type_GB', '8_type_GS', '8_type_PC',
                      '8_type_SS', '8_type_TB', '8_type_TS', '9_layer_exists', '9_layer_thickness', '9_type_AC',
                      '9_type_EF', '9_type_GB', '9_type_GS', '9_type_PC', '9_type_SS', '9_type_TB', '9_type_TS',
                      '10_layer_exists', '10_layer_thickness', '10_type_AC', '10_type_EF', '10_type_GB', '10_type_GS',
                      '10_type_PC', '10_type_SS', '10_type_TB', '10_type_TS', '11_layer_exists', '11_layer_thickness',
                      '11_type_AC', '11_type_EF', '11_type_GB', '11_type_GS', '11_type_PC', '11_type_SS', '11_type_TB',
                      '11_type_TS', '12_layer_exists', '12_layer_thickness', '12_type_AC', '12_type_EF', '12_type_GB',
                      '2_type_GS', '12_type_PC', '12_type_SS', '12_type_TB', '12_type_TS', '13_layer_exists',
                      '13_layer_thickness', '13_type_AC', '13_type_EF', '13_type_GB', '13_type_GS', '13_type_PC',
                      '13_type_SS', '13_type_TB', '13_type_TS', '14_layer_exists', '14_layer_thickness', '14_type_AC',
                      '14_type_EF', '14_type_GB', '14_type_GS', '14_type_PC', '14_type_SS', '14_type_TB', '14_type_TS'
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
                    , 'MEAN_TEMP_AVG_LAST_10','FREEZE_INDEX_LAST_10','FREEZE_THAW_LAST_10','MAX_TEMP_AVG_LAST_10','MIN_TEMP_AVG_LAST_10','MAX_TEMP_LAST_10','MIN_TEMP_LAST_10','DAYS_ABOVE_32_LAST_10','DAYS_BELOW_0_LAST_10','TOTAL_PRECIP_LAST_10','TOTAL_SNOWFALL_LAST_10','INTENSE_PRECIP_DAYS_LAST_10','WET_DAYS_LAST_10']]
    # missing_val_count_by_column = (using.isnull().sum())
    # print(missing_val_count_by_column)
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])
    values_3 = using.values
    values_3 = values_3
    values_3 = shuffle(values_3)
    # ensure all data is float
    values_3 = values_3.astype('float32')
    # # normalize features
    # scaler_3 = MinMaxScaler(feature_range=(-1, 1))
    # scaled_3 = scaler_3.fit_transform(values_3)
    # values_3 = scaled_3
    # x_1 = np.array(values_3[:, 149:]).reshape((values_3.shape[0], 10, 13))
    # y = np.array(values_3[:, 0])
    #
    # # PCA降维
    # pca = PCA(n_components=5, whiten=True, random_state=42)
    # nn_input = np.concatenate([np.array(values_3[:, 1:9]), pca.fit_transform(np.array(values_3[:, 9:149]))], axis=1)




    ##############################################
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
    print(list_features)
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
    print(analysis_features)
    print(len(analysis_features))

    # PCA降维
    pca = PCA(n_components=5, whiten=True, random_state=42)
    # y, nn input: 8+5+26,(气候前有40列)
    values_3 = np.concatenate([np.array(values_3[:, 0:9]), pca.fit_transform(np.array(values_3[:, 9:149])), analysis_features, np.array(values_3[:, 149:])], axis=1)

    print(np.array(values_3).shape)
    # nn_input增加了26列（13个10年每个都浓缩为两个：均值、方差）
    # normalize features
    scaler_3 = MinMaxScaler(feature_range=(-1, 1))
    scaled_3 = scaler_3.fit_transform(values_3)
    values_3 = scaled_3
    # x_1 = np.array(values_3[:, 149:]).reshape((values_3.shape[0], 10, 13))
    # y = np.array(values_3[:, 0])
    x_1 = np.array(values_3[count, 40:]).reshape((1, 10, 13))
    nn_input = np.array(values_3[count, 1:NN_FEATURES + 1])
    y = np.array(values_3[count, 0])
    if n == 1:
        x_1 = np.array(values_3[count, 6:]).reshape((1, 20, 2))
        nn_input = np.array(values_3[count, 1:6]).reshape((1, 5))
        y = np.array(values_3[count, 0])
    return x_1, nn_input, y

















