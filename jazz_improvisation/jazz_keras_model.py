import tensorflow as tf
import numpy as np
import data_utils as du
import rnn_functions as lstm
from music_utils import one_hot
import utils
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

X_train, Y_train, n_values, indices_values = du.load_music_utils()

n_a = 64
n_values = 78

reshapor = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation = 'softmax')

def djmodel(T_x, n_a, n_values):
    X = Input(shape = (T_x, n_values))
    a0 = Input(shape = (n_a, ), name = 'a0')
    c0 = Input(shape = (n_a, ), name = 'c0')
    a = a0
    c = c0

    outputs = []
    for t in range(T_x):
        x = Lambda(lambda x : X[:, t, :])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(inputs = x, initial_state = [a, c])
        out = densor(a)
        outputs.append(out)

    model = Model(inputs = [X, a0, c0], outputs = outputs)
    
    return model

model = djmodel(T_x = 30, n_a = 64, n_values = 78)

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit([X_train, a0, c0], list(Y_train), epochs = 100)

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, T_y = 100):
    x0 = Input(shape = (1, n_values))
    a0 = Input(shape = (n_a, ), name = 'a0')
    c0 = Input(shape = (n_a, ), name = 'c0')
    a = a0
    c = c0
    x = x0
    
    outputs = []
    for t in range(T_y):
        a, _, c = LSTM_cell(inputs = x, initial_state = [a, c])
        out = densor(a)
        outputs.append(out)
        x = Lambda(one_hot)(out)
    
    inference_model = Model(inputs = [x0, a0, c0], outputs = outputs)
    
    return inference_model

inference_model = music_inference_model(LSTM_cell, densor)
inference_model.summary()

x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, 2)
    results = to_categorical(indices, num_classes = x_initializer.shape[2])
    
    return results, indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

out_stream = du.generate_music(inference_model)