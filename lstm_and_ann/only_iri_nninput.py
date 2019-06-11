from keras.layers import Input, Dense, merge, Masking
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt
from numpy import concatenate
import pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras import initializers
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder




class MyFlatten(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MyFlatten, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask==None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))


from attention_utils import get_activations, get_data_recurrent, get_data_recurrent_1

NN_FEATURES = 13  # + 26
INPUT_DIM = 13  # 2
TIME_STEPS = 10  # 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False
not_use_LSTM = False

def r2_pred(y_true, y_pred):
    return r2_score(y_true, y_pred)



def myloss(y_true, y_pred):
    loss = (y_true - y_pred)*(y_true - y_pred)
    return loss

# # 不做相关的attention，只有nn的attention参与其中
# def attention_3d_block(inputs_lstm, inputs_nn):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     # input_lstm_dim_1 = int(inputs_lstm.shape[1])
#     input_lstm_dim_2 = int(inputs_lstm.shape[2])
#     # a = Permute((2, 1))(inputs_nn)
#     a = Reshape((TIME_STEPS, input_lstm_dim_2), name='nn_reshape')(inputs_nn)  # ? * 640 -> ? * 20* 32
#     a = Permute((2, 1))(a)
#     # print("在这个地方的形状：", input_dim, TIME_STEPS)
#     # assert input_lstm_dim_1 == TIME_STEPS
#     a = Dense(TIME_STEPS, activation='sigmoid')(a)  # softmax ······
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     output_attention_mul = merge([inputs_lstm, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul


# 包含两个网络的相关性attention；
def attention_3d_block(inputs_lstm, inputs_nn):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # input_lstm_dim_1 = int(inputs_lstm.shape[1])
    input_lstm_dim_2 = int(inputs_lstm.shape[2])
    # a = Permute((2, 1))(inputs_nn)
    a = Reshape((TIME_STEPS, input_lstm_dim_2), name='nn_reshape')(inputs_nn)  # ? * 640 -> ? * 20* 32
    a = Permute((2, 1))(a)
    # print("在这个地方的形状：", input_dim, TIME_STEPS)
    # assert input_lstm_dim_1 == TIME_STEPS
    a = Dense(TIME_STEPS, activation='tanh')(a)  # tanh ······
    print(a.shape)
    nn_1 = Permute((2, 1), name='attention_vec')(a)
    print(nn_1.shape)
    lstm_1 = Dense(input_lstm_dim_2, activation='tanh')(inputs_lstm)  # tanh
    print(lstm_1.shape)
    addings = merge([nn_1, lstm_1], name='dense_adding', mode='sum')
    print(addings.shape)
    a_probs = Dense(input_lstm_dim_2, activation='sigmoid', use_bias=False)(addings)  # softmax
    print(a_probs.shape)
    output_attention_mul = merge([inputs_lstm, a_probs], name='attention_mul', mode='mul')
    print(output_attention_mul.shape)
    return output_attention_mul


# # parallel: nn + lstm; With attention model；没有做masking，LSTM是定长序列
# def model_attention_applied_after_lstm():
#     # K.clear_session()
#     inputs_time_series = Input(shape=(TIME_STEPS, INPUT_DIM,))
#     inputs_nn_features = Input(shape=(NN_FEATURES, ))  # 默认用全局变量设置为5列nn特征输入
#     print("shape of inputs_nn_features：", inputs_nn_features.shape)
#     nn_dense_1 = Dense(640, activation='relu')(inputs_nn_features)  # relu  640 tanh 320
#     print("shape of nn_dense_1: ", nn_dense_1.shape)
#     lstm_units = 64  # 32
#     lstm_out = LSTM(lstm_units, return_sequences=True)(inputs_time_series)
#     attention_mul = attention_3d_block(lstm_out, nn_dense_1)
#     print(attention_mul.shape)
#     attention_mul = Flatten()(attention_mul)
#     attention_mul_next = Dense(10, activation='tanh')(attention_mul)  # ······  10
#     nn_out_final = Dense(15, activation='tanh')(nn_dense_1)  # ······  15  relu  , activation='linear'
#     con = merge([attention_mul_next, nn_out_final], name='last_merge', mode='concat')
#     output = Dense(1)(con)
#     model = Model(input=[inputs_time_series, inputs_nn_features], output=output)
#     return model

# 前人论文中的ANN模型
def model_ANN():
    K.clear_session()
    inputs_nn_features = Input(shape=(NN_FEATURES,))  # 默认用全局变量设置为5列nn特征输入
    print("shape of inputs_nn_features：", inputs_nn_features.shape)
    nn_dense_1 = Dense(130, activation='relu')(inputs_nn_features)  # relu  640
    print("shape of nn_dense_1: ", nn_dense_1.shape)
    nn_dense_2 = Dense(100, activation='relu')(nn_dense_1)
    nn_dense_3 = Dense(100, activation='relu')(nn_dense_2)

    nn_flatten = MyFlatten()(nn_dense_3)

    output = Dense(1, activation='linear')(nn_flatten)
    model = Model(input=inputs_nn_features, output=output)
    return model


# parallel: nn + lstm; With attention model；做了masking，LSTM是变长序列
def model_attention_applied_after_lstm():
    K.clear_session()
    inputs_time_series = Input(shape=(TIME_STEPS, INPUT_DIM,))
    inputs_nn_features = Input(shape=(NN_FEATURES, ))  # 默认用全局变量设置为5列nn特征输入
    print("shape of inputs_nn_features：", inputs_nn_features.shape)
    nn_dense_1 = Dense(320, activation='relu')(inputs_nn_features)  # relu  640-
    print("shape of nn_dense_1: ", nn_dense_1.shape)
    masking = Masking(mask_value=-100, input_shape=(inputs_time_series.shape[1], inputs_time_series.shape[2]))(inputs_time_series)
    lstm_units = 32  # 32-
    lstm_out = LSTM(lstm_units, return_sequences=True)(masking)
    attention_mul = attention_3d_block(lstm_out, nn_dense_1)
    print(attention_mul.shape)
    # attention_mul = Flatten()(attention_mul)
    attention_mul = MyFlatten()(attention_mul)
    attention_mul_next = Dense(200, activation='relu')(attention_mul)  # ······  (200 relu)
    nn_out_final = Dense(100, activation='relu')(nn_dense_1)  # ······  (100 relu)
    con = merge([attention_mul_next, nn_out_final], name='last_merge', mode='concat')

    # lstm_out = MyFlatten()(lstm_out)
    # before_output_1 = Dense(50, activation='relu')(lstm_out)  # (30 relu) 可以做进一步降低尝试（小幅度）(20-80)
    output = Dense(1, activation='linear')(con)
    model = Model(input=[inputs_time_series, inputs_nn_features], output=output)


    # lstm_out = MyFlatten()(lstm_out)
    # con = merge([lstm_out, nn_out_final], name='last_merge', mode='concat')
    #
    #
    # before_output_1 = Dense(10, activation='relu')(con)  # (30 relu) 可以做进一步降低尝试（小幅度）(20-80)
    # # before_output = Dense(30, activation='relu')(before_output_1)  # 70 relu
    # # drop = Dropout(0.1)(before_output_1)
    # output = Dense(1, activation='linear')(before_output_1)
    # model = Model(input=[inputs_time_series, inputs_nn_features], output=output)
    return model

# # 差值tanh尝试
# def model_attention_applied_after_lstm():
#     K.clear_session()
#     inputs_time_series = Input(shape=(TIME_STEPS, INPUT_DIM,))
#     inputs_nn_features = Input(shape=(NN_FEATURES, ))
#     inputs_nn_mri = Input(shape=(1, ))
#     print("shape of inputs_nn_features：", inputs_nn_features.shape)
#     print("type of inputs_nn_features：", type(inputs_nn_features))
#     nn_dense_1 = Dense(320, activation='relu')(inputs_nn_features)  # relu  640
#     print("shape of nn_dense_1: ", nn_dense_1.shape)
#     masking = Masking(mask_value=-100, input_shape=(inputs_time_series.shape[1], inputs_time_series.shape[2]))(inputs_time_series)
#     lstm_units = 32  # 32
#     lstm_out = LSTM(lstm_units, return_sequences=True)(masking)
#     attention_mul = attention_3d_block(lstm_out, nn_dense_1)
#     print(attention_mul.shape)
#     # attention_mul = Flatten()(attention_mul)
#     attention_mul = MyFlatten()(attention_mul)
#     attention_mul_next = Dense(4, activation='relu')(attention_mul)  # ······  8；10；4 relu
#     nn_out_final = Dense(500, activation='relu')(nn_dense_1)  # ······  8；10；200  relu
#     con = merge([attention_mul_next, nn_out_final], name='last_merge', mode='concat')
#     before_output = Dense(100, activation='tanh')(con)
#     # drop = Dropout(0.1)(before_output)
#     for_delta = Dense(1)(con)  # ````````
#     # output = Dense(1, activation='softmax')(con)
#     output = merge([for_delta, inputs_nn_mri], name='delta_sum', mode='sum')
#     model = Model(input=[inputs_time_series, inputs_nn_features, inputs_nn_mri], output=output)
#     return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':

    N = 300000
    # N = 300 -> too few = no training
    inputs_1, inputs_2, outputs, primary_data, scaler = get_data_recurrent(N, TIME_STEPS, INPUT_DIM, NN_FEATURES)
    print(inputs_2[:, 0].shape)
    print(inputs_2.shape)
    print(inputs_1.shape)

    input_3 = inputs_2[:, 0].reshape(np.array(inputs_2.shape[0]), 1)
    print(input_3.shape)
    print("******")

    for i in range(inputs_2.shape[1] - 1):
        input_3 = np.concatenate([np.array(input_3), np.array(inputs_2[:, 0]).reshape(np.array(inputs_2.shape[0]), 1)], axis=1)

    inputs_2 = input_3
    print(input_3.shape)
    print("******")
    print(input_3)

    inputs_4 = []

    for i in range(inputs_1.shape[1]):
        np.concatenate([np.array(inputs_1[:, i]), np.array(inputs_2[:, 0]).reshape(np.array(inputs_2.shape[0]), 1)],
                                 axis=1)
        print(np.array(inputs_1[:, i]).shape)
        print((np.array(inputs_2[:, 0]).reshape(np.array(inputs_2.shape[0]), 1)).shape)
        print((np.concatenate([np.array(inputs_1[:, i]), np.array(inputs_2[:, 0]).reshape(np.array(inputs_2.shape[0]), 1)],
                                 axis=1)).shape)
        # print(np.concatenate([np.array(inputs_1[:, i]), np.array(inputs_2[:, 0]).reshape(np.array(inputs_2.shape[0]), 1)],
        #                          axis=1))


    print("@@@@@@@@@@@@@@@@@@")
    print(np.array(inputs_4).shape)
    # inputs_4 = np.array(inputs_4).swapaxes(1, 0)
    print(np.array(inputs_4).shape)
    print("@@@@@@@@@@@@@@@@@@")

    if not_use_LSTM:
        m = model_ANN()
    else:
        m = model_attention_applied_after_lstm()  # 目前正在用的







    adam = Adam(lr=1.e-4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    m.compile(optimizer='adam', loss='mse', metrics=['mse'])  # rmsprop

    # m.compile(optimizer='sgd', loss=myloss, metrics=['mae'])

    last_for_train = 500  # 500
    print(m.summary())
    # 16 - 8 : 0.8
    # 12 - 8
    # 80 - 512
    # 100 - 512 - 0.05
    # 用LSTM的版本-80-512
    # 指数衰减 - 65 - 512 - 0.1
    # input_2是nninput
    history = m.fit([inputs_1[:-last_for_train], inputs_2[:-last_for_train]], outputs[:-last_for_train], epochs=1, batch_size=512, validation_split=0.1, callbacks=[reduce_lr])
    y_predictions = m.predict([inputs_1[-last_for_train:], inputs_2[-last_for_train:]])

    # history = m.fit(inputs_4[:-last_for_train], outputs[:-last_for_train], epochs=1, batch_size=512, validation_split=0.1, callbacks=[reduce_lr])
    # y_predictions = m.predict(inputs_4[-last_for_train:])


    # # 16 - 8 : 0.8  差值训练版本
    # m.fit([inputs_1[:-500], inputs_2[:-500], inputs_2[:-500, 0]], outputs[:-500], epochs=16, batch_size=8, validation_split=0.2)
    # y_predictions = m.predict([inputs_1[-500:], inputs_2[-500:], inputs_2[:-500, 0]])


    # convert predictions
    print(y_predictions.shape)
    print(np.array(primary_data[-last_for_train:]).shape)
    y_predictions = concatenate((y_predictions, primary_data[-last_for_train:]), axis=1)
    y_predictions = scaler.inverse_transform(y_predictions)
    y_predictions = y_predictions[:, 0]

    # convert real value
    print(np.array(outputs[-last_for_train:]).shape)
    print(np.array(primary_data[-last_for_train:]).shape)
    y_real = concatenate((np.array(outputs[-last_for_train:]).reshape(len(outputs[-last_for_train:]), 1), primary_data[-last_for_train:]), axis=1)
    y_real = scaler.inverse_transform(y_real)
    y_real = y_real[:, 0]

    problems = []
    for i in range(len(y_real)):
        if abs(y_real[i] - y_predictions[i]) >= 2:
            problems.append(i)
    print(problems)



    rmse = sqrt(mean_squared_error(y_real, y_predictions))
    print('Test RMSE: %.3f' % rmse)
    print("length of test data: ", len(y_real))
    r_2 = r2_score(y_real, y_predictions)
    print('Test R^2: %.3f' % r_2)













    # plt.plot()
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    #
    # #
    # x_1 = [x/100 for x in range(1, 500)]
    # y_1 = [x/100 for x in range(1, 500)]
    #
    # plt.figure()
    # plt.plot(y_real, y_predictions, "o")
    # plt.plot(x_1, y_1, color='red', label='y=x')
    # plt.xlabel("Real value(MRI)")
    # plt.ylabel("Predicted value(MRI)")
    # plt.show()


    # estimator = KerasClassifier(build_fn=m, nb_epoch=70, batch_size=512)
    #
    # ################################交叉验证
    # results = cross_val_score(estimator, X, dummy_y, cv=10)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    #     cvscores.append(scores[1])

    ######################################

    # inputs_2 = np.array(inputs_2).reshape((np.array(inputs_2).shape[0], 1, np.array(inputs_2).shape[1]))
    # print(np.array(inputs_1).shape)
    # print(np.array(inputs_2).shape)
    # X = np.concatenate([inputs_1, inputs_2], axis = 1)
    # print(X.shape)
    y = np.array(outputs).reshape((len(outputs), 1))
    print(y.shape)
    print('i am still working~')
    estimator = KerasClassifier(build_fn=m, nb_epoch=70, batch_size=512)
    seed = 42
    np.random.seed(seed)
    # kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    # results = cross_val_score(estimator, [inputs_1, inputs_2], outputs, cv=kfold)
    #
    # # fix random seed for reproducibility
    # seed = 7
    # np.random.seed(seed)
    # # load pima indians dataset
    # dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter="\t")
    # # split into input (X) and output (Y) variables
    # X = dataset[:, 0:8]
    # Y = dataset[:, 8]
    # # define 10-fold cross validation test harness

    X = np.array(inputs_1)[:2400, :, :]

    y = np.array(outputs)[:2400]
    print(np.array(X).shape)
    print(np.array(y).shape)

    X = np.array([
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74]
    ])

    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])

    X = np.array([[x] for x in range(0, 2400)])

    y = np.array([x for x in range(0, 2400)])



    print(np.array(X).shape)
    print(np.array(y).shape)
    floder = KFold(n_splits=5, random_state=0, shuffle=False)

    # for train, test in floder.split(X, y):
    #     print('Train: %s | test: %s' % (train, test))
    #     print(" ")



    cvscores = []
    r_2_list = []
    rmse_list = []
    for (train, test) in floder.split(X, y):
        #最好效果
        #history = m.fit([inputs_1[train], inputs_2[train]], outputs[train], epochs=65, batch_size=512, verbose=0, validation_split=0.1, callbacks=[reduce_lr])

        history = m.fit([inputs_1[train], inputs_2[train]], outputs[train], epochs=20, batch_size=512, verbose=0, validation_split=0.1, callbacks=[reduce_lr])
        y_predictions = m.predict([inputs_1[test], inputs_2[test]])


        # evaluate the model
        # scores = m.evaluate([inputs_1[train], inputs_2[train]], outputs[train], verbose=0)
        # print("%s: %.2f%%" % (m.metrics_names[1], scores[1]*1000))
        # cvscores.append(scores[1])



        # history = m.fit(inputs_4[:-last_for_train], outputs[:-last_for_train], epochs=1, batch_size=512,
        #                 validation_split=0.1, callbacks=[reduce_lr])
        # y_predictions = m.predict(inputs_4[-last_for_train:])

        # # 16 - 8 : 0.8  差值训练版本
        # m.fit([inputs_1[:-500], inputs_2[:-500], inputs_2[:-500, 0]], outputs[:-500], epochs=16, batch_size=8, validation_split=0.2)
        # y_predictions = m.predict([inputs_1[-500:], inputs_2[-500:], inputs_2[:-500, 0]])

        # convert predictions
        print(y_predictions.shape)
        print(np.array(primary_data[test]).shape)
        y_predictions = concatenate((y_predictions, primary_data[test]), axis=1)
        y_predictions = scaler.inverse_transform(y_predictions)
        y_predictions = y_predictions[:, 0]

        # convert real value
        print(np.array(outputs[test]).shape)
        print(np.array(primary_data[test]).shape)
        y_real = concatenate((np.array(outputs[test]).reshape(len(outputs[test]), 1),
                              primary_data[test]), axis=1)
        y_real = scaler.inverse_transform(y_real)
        y_real = y_real[:, 0]


        rmse = sqrt(mean_squared_error(y_real, y_predictions))
        print('Test RMSE: %.3f' % rmse)
        print("length of test data: ", len(y_real))
        r_2 = r2_score(y_real, y_predictions)
        print('Test R^2: %.3f' % r_2)
        rmse_list.append(rmse)
        # print('Test RMSE: %.3f' % rmse)
        r_2_list.append(r_2)

        # plt.plot()
        # plt.plot(history.history['loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        #
    print('************************')
    print("R^2 = ", '[', format(r_2_list[0], '.3f'), format(r_2_list[1], '.3f'), format(r_2_list[2], '.3f'),
          format(r_2_list[3], '.3f'), format(r_2_list[4], '.3f'), ']')
    print("mean: ", format(np.array(r_2_list).mean(), '.3f'))
    print("std: ", format(np.array(r_2_list).std(ddof=1), '.3f'))
    print("RMSE = ", '[', format(rmse_list[0], '.3f'), format(rmse_list[1], '.3f'), format(rmse_list[2], '.3f'),
          format(rmse_list[3], '.3f'), format(rmse_list[4], '.3f'), ']')
    print("mean: ", format(np.array(rmse_list).mean(), '.3f'))
    print("std: ", format(np.array(rmse_list).std(ddof=1), '.3f'))


    # x_1 = [x / 100 for x in range(1, 500)]
    # y_1 = [x / 100 for x in range(1, 500)]
    #
    # plt.figure()
    # plt.plot(y_real, y_predictions, "o")
    # plt.plot(x_1, y_1, color='red', label='y=x')
    # plt.xlabel("Real value(MRI)")
    # plt.ylabel("Predicted value(MRI)")
    # plt.show()








    # attention_vectors = []
    # for i in range(1):
    #     testing_inputs_1, testing_inputs_2, testing_outputs = get_data_recurrent_1(1, i+1000, TIME_STEPS, INPUT_DIM, NN_FEATURES)
    #     attention_vector = np.mean(get_activations(m,
    #                                                [testing_inputs_1, testing_inputs_2],
    #                                                print_shape_only=True,
    #                                                layer_name='attention_vec')[0], axis=2).squeeze()
    #     print('attention =', attention_vector)
    #     assert (np.sum(attention_vector) - 1.0) < 1e-5
    #     attention_vectors.append(attention_vector)
    #
    # attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # # plot part.
    #
    #
    # pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
    #                                                                      title='Attention Mechanism as '
    #                                                                            'a function of input'
    #                                                                            ' dimensions.')
    # plt.show()
















