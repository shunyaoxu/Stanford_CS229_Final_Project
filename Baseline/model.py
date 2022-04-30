import numpy as np
from layers import *
from optim import *


class Model2x2(object):
    # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    # Learnable parameters are stored in the self.params.
    def __init__(
        self,
        hidden_dims,
        input_dim=120000,
        num_classes=24,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None):
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        hiddenDim = hidden_dims.copy()
        hiddenDim.insert(0, input_dim)
        hiddenDim.append(num_classes)
        for i in range(1, self.num_layers+1):
          self.params['W'+str(i)] = np.random.randn(hiddenDim[i-1], hiddenDim[i]) * weight_scale
          self.params['b'+str(i)] = np.zeros(hiddenDim[i])
          if i != self.num_layers and self.normalization:
            self.params['gamma'+str(i)] = np.ones(hiddenDim[i])
            self.params['beta'+str(i)] = np.zeros(hiddenDim[i])

        # initialize dropout
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # initialize batchnorm
        self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        # Inputs -- X: (N, D), y: (N, )
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        
        # Forward Propagation
        scores = None
        affine_cache = {}
        relu_cache = {}
        bn_cache = {}
        drop_cache = {}
        output = X.copy()

        for i in range(1, self.num_layers):
          output, cache = affine_forward(output, self.params['W'+str(i)], self.params['b'+str(i)])
          affine_cache[i] = cache

          if self.normalization == 'batchnorm':
            output, cache = batchnorm_forward(output, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1])
            bn_cache[i] = cache
          elif self.normalization == 'layernorm':
            output, cache = layernorm_forward(output, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1])
            bn_cache[i] = cache

          output, cache = relu_forward(output)
          relu_cache[i] = cache

          if self.use_dropout:
            output, cache = dropout_forward(output, self.dropout_param)
            drop_cache[i] = cache

        scores, cache = affine_forward(output, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        
        # If test mode return early.
        if mode == "test":
            return scores
        
        # Backward Propagation
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)
        dout, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dout, cache)
        loss += 0.5 * self.reg * np.sum(np.power(self.params['W'+str(self.num_layers)], 2))
        grads['W'+str(self.num_layers)] += self.reg * self.params['W'+str(self.num_layers)]

        for i in range(self.num_layers-1, 0, -1):
          if self.use_dropout:
            dout = dropout_backward(dout, drop_cache[i])
          dout = relu_backward(dout, relu_cache[i])
          if self.normalization == 'batchnorm':
            dout, grads['gamma'+str(i)], grads['beta'+str(i)] = batchnorm_backward(dout, bn_cache[i])
          elif self.normalization == 'layernorm':
            dout, grads['gamma'+str(i)], grads['beta'+str(i)] = layernorm_backward(dout, bn_cache[i])
          dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dout, affine_cache[i])
          loss += 0.5 * self.reg * np.sum(np.power(self.params['W'+str(i)], 2))
          grads['W'+str(i)] += self.reg * self.params['W'+str(i)]

        return loss, grads