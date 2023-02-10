import numpy as np
import data_utils as du
import rnn_functions as lstm
import utils

X, Y, n_values, indices_values = du.load_music_utils()
X = np.transpose(X, axes = [2, 0, 1])
Y = np.transpose(Y, axes = [2, 0, 1])

T_x = X.shape[2]
n_values = X.shape[0]
n_a = 64

def djmodel(X_train, Y_train, a0, parameters, learning_rate, v, s, t = 1):
    
    a, y_pred, c, caches = lstm.lstm_forward(X, a0, parameters)
    cost = utils.compute_cost(y_pred, Y_train)
    da, dz = utils.backpropagation(X_train, Y_train, parameters, caches)
    gradients = lstm.lstm_backward(dz, da, caches)
    parameters, v, s = utils.adam(parameters, gradients, learning_rate, v, s, t)
    
    return a[:, :, X_train.shape[2] - 1], cost, parameters, v, s

def model(X_train, Y_train, num_epochs, n_a, learning_rate_0, decay):
    n_values = X.shape[0]
    m = X.shape[1]
    
    parameters = utils.initialize_parameters(n_values, n_values, n_a)
    v, s = utils.initialize_adam(parameters)
    a = np.zeros((n_a, m))
    
    for epoch in range(num_epochs):
        learning_rate = learning_rate_0 * ( 1. / (1. + decay * epoch))
        a, cost, parameters, v, s = djmodel(X_train, Y_train, a, parameters, learning_rate, v, s, t = 1)
        print('Iteration %d, Cost: %f' % (epoch, cost) + '\n')
    
    return parameters

parameters = model(X, Y, num_epochs = 100, n_a = n_a, learning_rate_0 = 0.01, decay = 0.01)

def sample(T_x, n_values, n_a, parameters, T_y = 100, temperature = 1.0):    
    x = np.zeros((n_values, 1))
    a0 = np.zeros((n_a, 1))
    c0 = np.zeros((n_a, 1))
    a = a0
    c = c0
    indices = []
    X = []
    for t in range(T_y):
        a, c, y_pred, _ = lstm.lstm_cell_forward(x, a, c, parameters)
        y_pred = np.squeeze(y_pred)
        y_pred = np.log(y_pred) / temperature
        exp_y = np.exp(y_pred)
        y_pred = exp_y / np.sum(exp_y)
        probas = np.random.multinomial(1, y_pred, 1)
        idx = np.random.choice(list(range(n_values)), p = probas.ravel())
        indices.append(idx)
        
        x = np.zeros((n_values, 1))
        x[idx, 0] = 1
        X.append(x)
   
    X = np.array(X)
    X = X.squeeze()
    indices = np.array(indices)
    indices = np.expand_dims(indices, axis = 1)  
    return X, indices

X_pred, indices = sample(T_x, n_values, n_a, parameters, T_y = 100, temperature = 1.0)  
    
out_stream = du.generate_music(parameters)