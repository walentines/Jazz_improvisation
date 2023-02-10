import numpy as np

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis = 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(n_x, n_y, n_a):    
    Wf = np.random.randn(n_a, n_a + n_x) * 0.01
    bf = np.zeros((n_a, 1))
    Wi = np.random.randn(n_a, n_a + n_x) * 0.01
    bi = np.zeros((n_a, 1))
    Wc = np.random.randn(n_a, n_a + n_x) * 0.01
    bc = np.zeros((n_a, 1))
    Wo = np.random.randn(n_a, n_a + n_x) * 0.01
    bo = np.zeros((n_a, 1))
    Wy = np.random.randn(n_y, n_a) * 0.01
    by = np.zeros((n_y, 1))
    
    parameters = {'Wf': Wf, 'bf': bf, 'Wi': Wi, 'bi': bi, 'Wc': Wc, 'bc': bc, 'Wo': Wo, 'bo': bo, 'Wy': Wy, 'by': by}
    
    return parameters

def initialize_adam(parameters):    
    v = {}
    s = {}
    
    v['dWf'] = np.zeros((parameters['Wf'].shape[0], parameters['Wf'].shape[1]))
    v['dbf'] = np.zeros((parameters['bf'].shape[0], parameters['bf'].shape[1]))
    v['dWi'] = np.zeros((parameters['Wi'].shape[0], parameters['Wi'].shape[1]))
    v['dbi'] = np.zeros((parameters['bi'].shape[0], parameters['bi'].shape[1]))
    v['dWc'] = np.zeros((parameters['Wc'].shape[0], parameters['Wc'].shape[1]))
    v['dbc'] = np.zeros((parameters['bc'].shape[0], parameters['bc'].shape[1]))
    v['dWo'] = np.zeros((parameters['Wo'].shape[0], parameters['Wo'].shape[1]))
    v['dbo'] = np.zeros((parameters['bo'].shape[0], parameters['bo'].shape[1]))
    v['dWy'] = np.zeros((parameters['Wy'].shape[0], parameters['Wy'].shape[1]))
    v['dby'] = np.zeros((parameters['by'].shape[0], parameters['by'].shape[1]))
    
    s['dWf'] = np.zeros((parameters['Wf'].shape[0], parameters['Wf'].shape[1]))
    s['dbf'] = np.zeros((parameters['bf'].shape[0], parameters['bf'].shape[1]))
    s['dWi'] = np.zeros((parameters['Wi'].shape[0], parameters['Wi'].shape[1]))
    s['dbi'] = np.zeros((parameters['bi'].shape[0], parameters['bi'].shape[1]))
    s['dWc'] = np.zeros((parameters['Wc'].shape[0], parameters['Wc'].shape[1]))
    s['dbc'] = np.zeros((parameters['bc'].shape[0], parameters['bc'].shape[1]))
    s['dWo'] = np.zeros((parameters['Wo'].shape[0], parameters['Wo'].shape[1]))
    s['dbo'] = np.zeros((parameters['bo'].shape[0], parameters['bo'].shape[1]))
    s['dWy'] = np.zeros((parameters['Wy'].shape[0], parameters['Wy'].shape[1]))
    s['dby'] = np.zeros((parameters['by'].shape[0], parameters['by'].shape[1]))
    
    return v, s     

def compute_cost(y_pred, y_train):
    m = y_train.shape[0]
    cost = - 1 / m * np.sum(y_train * np.log(y_pred))
    cost = float(cost)
    
    return cost

def backpropagation(X, Y, parameters, caches):
    T_x = X.shape[2]
    n_values = X.shape[0]
    m = X.shape[1]
    n_a = parameters['Wy'].shape[1]
    
    (caches, x, y_pred) = caches

    da = np.zeros((n_a, m, T_x))
    dz = np.zeros((n_values, m, T_x))
    for t in reversed(range(T_x)):
        dz[:, :, t] = y_pred[:, :, t] - Y[:, :, t]
        da[:, :, t] = np.dot(parameters['Wy'].T, dz[:, :, t])
    
    return da, dz

def gradient_descent(parameters, gradients, learning_rate):
    parameters['Wf'] = parameters['Wf'] - np.dot(learning_rate, gradients['dWf'])
    parameters['bf'] = parameters['bf'] - np.dot(learning_rate, gradients['dbf'])
    parameters['Wi'] = parameters['Wi'] - np.dot(learning_rate, gradients['dWi'])
    parameters['bi'] = parameters['bi'] - np.dot(learning_rate, gradients['dbi'])
    parameters['Wc'] = parameters['Wc'] - np.dot(learning_rate, gradients['dWc'])
    parameters['bc'] = parameters['bc'] - np.dot(learning_rate, gradients['dbc'])
    parameters['Wo'] = parameters['Wo'] - np.dot(learning_rate, gradients['dWo'])
    parameters['bo'] = parameters['bo'] - np.dot(learning_rate, gradients['dbo'])
    parameters['Wy'] = parameters['Wy'] - np.dot(learning_rate, gradients['dWy'])
    parameters['by'] = parameters['by'] - np.dot(learning_rate, gradients['dby'])
    
    return parameters
    
def adam(parameters, gradients, learning_rate, v, s, t, beta1 = 0.9, beta2 = 0.999, epsilon = 0.001):
    v_corrected = {}
    s_corrected = {}
    
    v['dWf'] = beta1 * v['dWf'] + (1 - beta1) * gradients['dWf']
    v['dbf'] = beta1 * v['dbf'] + (1 - beta1) * gradients['dbf']
    v['dWi'] = beta1 * v['dWi'] + (1 - beta1) * gradients['dWi']
    v['dbi'] = beta1 * v['dbi'] + (1 - beta1) * gradients['dbi']
    v['dWc'] = beta1 * v['dWc'] + (1 - beta1) * gradients['dWc']
    v['dbc'] = beta1 * v['dbc'] + (1 - beta1) * gradients['dbc']
    v['dWo'] = beta1 * v['dWo'] + (1 - beta1) * gradients['dWo']
    v['dbo'] = beta1 * v['dbo'] + (1 - beta1) * gradients['dbo']
    v['dWy'] = beta1 * v['dWy'] + (1 - beta1) * gradients['dWy']
    v['dby'] = beta1 * v['dby'] + (1 - beta1) * gradients['dby']
    
    v_corrected['dWf'] = v['dWf'] / (1 - beta1 ** t)
    v_corrected['dbf'] = v['dbf'] / (1 - beta1 ** t)
    v_corrected['dWi'] = v['dWi'] / (1 - beta1 ** t)
    v_corrected['dbi'] = v['dbi'] / (1 - beta1 ** t)
    v_corrected['dWc'] = v['dWc'] / (1 - beta1 ** t)
    v_corrected['dbc'] = v['dbc'] / (1 - beta1 ** t)
    v_corrected['dWo'] = v['dWo'] / (1 - beta1 ** t)
    v_corrected['dbo'] = v['dbo'] / (1 - beta1 ** t)
    v_corrected['dWy'] = v['dWy'] / (1 - beta1 ** t)
    v_corrected['dby'] = v['dby'] / (1 - beta1 ** t)
    
    s['dWf'] = beta2 * s['dWf'] + (1 - beta2) * (gradients['dWf'] ** 2)
    s['dbf'] = beta2 * s['dbf'] + (1 - beta2) * (gradients['dbf'] ** 2)
    s['dWi'] = beta2 * s['dWi'] + (1 - beta2) * (gradients['dWi'] ** 2)
    s['dbi'] = beta2 * s['dbi'] + (1 - beta2) * (gradients['dbi'] ** 2)
    s['dWc'] = beta2 * s['dWc'] + (1 - beta2) * (gradients['dWc'] ** 2)
    s['dbc'] = beta2 * s['dbc'] + (1 - beta2) * (gradients['dbc'] ** 2)
    s['dWo'] = beta2 * s['dWo'] + (1 - beta2) * (gradients['dWo'] ** 2)
    s['dbo'] = beta2 * s['dbo'] + (1 - beta2) * (gradients['dbo'] ** 2)
    s['dWy'] = beta2 * s['dWy'] + (1 - beta2) * (gradients['dWy'] ** 2)
    s['dby'] = beta2 * s['dby'] + (1 - beta2) * (gradients['dby'] ** 2)
    
    s_corrected['dWf'] = s['dWf'] / (1 - beta2 ** t)
    s_corrected['dbf'] = s['dbf'] / (1 - beta2 ** t)
    s_corrected['dWi'] = s['dWi'] / (1 - beta2 ** t)
    s_corrected['dbi'] = s['dbi'] / (1 - beta2 ** t)
    s_corrected['dWc'] = s['dWc'] / (1 - beta2 ** t)
    s_corrected['dbc'] = s['dbc'] / (1 - beta2 ** t)
    s_corrected['dWo'] = s['dWo'] / (1 - beta2 ** t)
    s_corrected['dbo'] = s['dbo'] / (1 - beta2 ** t)
    s_corrected['dWy'] = s['dWy'] / (1 - beta2 ** t)
    s_corrected['dby'] = s['dby'] / (1 - beta2 ** t)

    parameters['Wf'] = parameters['Wf'] - np.dot(learning_rate, v_corrected['dWf'] / (np.sqrt(s_corrected['dWf']) + epsilon))
    parameters['bf'] = parameters['bf'] - np.dot(learning_rate, v_corrected['dbf'] / (np.sqrt(s_corrected['dbf']) + epsilon))
    parameters['Wi'] = parameters['Wi'] - np.dot(learning_rate, v_corrected['dWi'] / (np.sqrt(s_corrected['dWi']) + epsilon))
    parameters['bi'] = parameters['bi'] - np.dot(learning_rate, v_corrected['dbi'] / (np.sqrt(s_corrected['dbi']) + epsilon))
    parameters['Wc'] = parameters['Wc'] - np.dot(learning_rate, v_corrected['dWc'] / (np.sqrt(s_corrected['dWc']) + epsilon))
    parameters['bc'] = parameters['bc'] - np.dot(learning_rate, v_corrected['dbc'] / (np.sqrt(s_corrected['dbc']) + epsilon))
    parameters['Wo'] = parameters['Wo'] - np.dot(learning_rate, v_corrected['dWo'] / (np.sqrt(s_corrected['dWo']) + epsilon))
    parameters['bo'] = parameters['bo'] - np.dot(learning_rate, v_corrected['dbo'] / (np.sqrt(s_corrected['dbo']) + epsilon))
    parameters['Wy'] = parameters['Wy'] - np.dot(learning_rate, v_corrected['dWy'] / (np.sqrt(s_corrected['dWy']) + epsilon))
    parameters['by'] = parameters['by'] - np.dot(learning_rate, v_corrected['dby'] / (np.sqrt(s_corrected['dby']) + epsilon))
    
    return parameters, v, s

def clip(gradients, maxValue):
    dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby = gradients['dWf'], gradients['dbf'], gradients['dWi'], gradients['dbi'], gradients['dWc'], gradients['dbc'], gradients['dWo'], gradients['dbo'], gradients['dWy'], gradients['dby']
    
    for gradient in [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient)
        
    gradients = {'dWf': dWf, 'dbf': dbf, 'dWi': dWi, 'dbi': dbi, 'dWc': dWc, 'dbc': dbc, 'dWo': dWo, 'dbo': dbo, 'dWy': dWy, 'dby': dby}
    
    return gradients
