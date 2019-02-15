import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Flatten, Dense, LSTM
from keras.optimizers import Adam
from keras.layers.merge import concatenate

def LSTM_NN_Model():
    input_layer = Input(shape=(4,2))
    b1_out = LSTM(64, return_sequences=False)(input_layer)

    b2_out = Dense(32, activation="relu", kernel_regularizer="l2")(input_layer)
    b2_out = Flatten()(b2_out)

    concatenated = concatenate([b1_out, b2_out])
    out = Dense(4, activation="relu", kernel_regularizer="l2")(concatenated)
    out = Dense(4, activation="relu", kernel_regularizer="l2")(out)
    # out = Dense(1, activation="linear", kernel_constraint=non_neg(), name='output_layer')(out)
    out = Dense(1, activation="linear", name='output_layer')(out)

    model = Model([input_layer], out)
    model.compile(loss=["mse"], optimizer=Adam(0.0001), metrics=["mae"])

    return model

def getModels():
    models = {
        "LSTM-NN": LSTM_NN_Model()
    }
    return models