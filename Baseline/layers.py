import numpy as np

#%% Affine Layer
def affine_forward(x, w, b):
    # x: (N, D)
    # w: (D, M)
    # b: (M, )
    out = x.dot(w) + b.reshape(1, -1)
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache

    dwx = dout
    dx = dwx.dot(w.T)
    dw = x.T.dot(dwx)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db


#%% ReLU Layer
def relu_forward(x):
    out = x * (x > 0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


#%% Softmax Loss
def softmax_loss(x, y):
    # x: (N, C)
    # y: (N, )
    N,_ = x.shape
    
    exp = np.exp(x)
    sumScores = np.sum(exp, axis=1)
    scores = exp / sumScores.reshape(-1, 1)

    loss = - np.log(scores[np.arange(N), y])
    loss = np.sum(loss) / N

    dx = scores.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    return loss, dx


#%% Batchnorm Layer
def batchnorm_forward(x, gamma, beta, bn_param):
    # x: (N, D)
    # gamma: (D, )
    # beta: (D, )
    # param: mode, esp, momentum, running_mean(D, ), running_var(D, )
    N, D = x.shape
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    
    cache = None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        bn_param["running_mean"] = momentum * running_mean + (1 - momentum) * sample_mean
        bn_param["running_var"] = momentum * running_var + (1 - momentum) * sample_var
        x_numer = x - sample_mean
        x_denom = 1 / np.sqrt(sample_var + eps)
        x_norm = x_numer * x_denom
        out = gamma * x_norm + beta
        cache = (x_numer, x_denom, x_norm, gamma)
    elif mode == "test":
        out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta
    else:
        raise ValueError('Invalid forward batchnorm mode')
    
    return out, cache

def batchnorm_backward(dout, cache):
    N, D = dout.shape
    x_numer, x_denom, x_norm, gamma = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    dx_norm = gamma * dout
    dx_numer = dx_norm * x_denom
    dx_denom = np.sum(dx_norm * x_numer, axis=0)
    dx = dx_numer.copy()
    dx_var = -0.5 * np.power(x_denom, 3) * dx_denom
    dx += 2 / N * x_numer * dx_var
    dmean = -1 * np.sum(dx_numer, axis=0)
    dx += 1 / N * dmean
    
    return dx, dgamma, dbeta


#%% Dropout Layer
def dropout_forward(x, dropout_param):
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    if mode == "train":
        mask = np.random.rand(*x.shape) < p
        out = x * mask / p
    else:
        out = x
    
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    
    if mode == "train":
        dx = mask * dout / dropout_param["p"]
    else:
        dx = dout
    
    return dx