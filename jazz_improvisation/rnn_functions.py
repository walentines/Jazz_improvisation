import numpy as np
from utils import *

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']
    
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    
    concat = np.concatenate((a_prev, xt), axis = 0)
    
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)    
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
    
    yt_pred = softmax(np.dot(Wy, a_next) + by)
    
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    
    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    caches = []

    Wy = parameters['Wy']
    
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    a_next = a0
    c_next = np.zeros((n_a, m))
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt
        caches.append(cache)
        
    caches = (caches, x, y)
    
    return a, y, c, caches

def lstm_cell_backward(dz, da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    dWy = np.dot(dz, a_next.T)
    dby = np.sum(dz, axis = 1, keepdims = True)
    
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next) * (1 - cct * 2)
    dit = (dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next) * ft * (1 - ft)
    
    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis = 0).T)
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis = 0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis = 0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis = 0).T)
    dbf = np.sum(dft, axis = 1, keepdims = True)
    dbi = np.sum(dit, axis = 1, keepdims = True)
    dbc = np.sum(dcct, axis = 1, keepdims = True)
    dbo = np.sum(dot, axis = 1, keepdims = True)

    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.tanh(c_next) ** 2) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)
    
    gradients = {'dxt': dxt,
                 'da_prev': da_prev,
                 'dc_prev': dc_prev,
                 'dWf': dWf,
                 'dbf': dbf,
                 'dWi': dWi,
                 'dbi': dbi,
                 'dWc': dWc,
                 'dbc': dbc,
                 'dWo': dWo,
                 'dbo': dbo,
                 'dWy': dWy,
                 'dby': dby}
    
    return gradients

def lstm_backward(dz, da, caches):
    (caches, x, y) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    n_y = parameters['Wy'].shape[0]
    
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dWy = np.zeros((n_y, n_a))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    dby = np.zeros((n_y, 1))
    
    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(dz[:, :, t], da[:, :, t] + da_prevt, dc_prevt, caches[t])
        da_prevt = gradients['da_prev']
        dc_prevt = gradients['dc_prev']
        dx[:, :, t] = gradients['dxt']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']
        dWy += gradients['dWy']
        dby += gradients['dby']
    
    dWf /= m
    dWi /= m
    dWc /= m
    dWo /= m
    dbf /= m
    dbi /= m
    dbc /= m
    dbo /= m
    dWy /= m
    dby /= m
        
    da0 = da_prevt
    
    gradients = {'dx': dx,
                 'da0': da0,
                 'dWf': dWf,
                 'dbf': dbf,
                 'dWi': dWi,
                 'dbi': dbi,
                 'dWc': dWc,
                 'dbc': dbc,
                 'dWo': dWo,
                 'dbo': dbo,
                 'dWy': dWy,
                 'dby': dby}
    
    return gradients