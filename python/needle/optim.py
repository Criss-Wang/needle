"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        # BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            p_id = id(p)
            grad = (1. - self.momentum) * \
                (p.grad.data + self.weight_decay * p.data)
            grad += self.momentum * self.u.get(p_id, 0.)
            grad = ndl.Tensor(grad, dtype=p.data.dtype)
            p.data -= self.lr * grad
            self.u[p_id] = grad
        # END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
          if self.weight_decay > 0:
            grad = param.grad.data + self.weight_decay * param.data
          else:
            grad = param.grad.data

          self.m[param] = (
            self.beta1 * self.m.get(param, 0.) + ( 1 - self.beta1 ) * grad
          )
          self.v[param] = (
            self.beta2 * self.v.get(param, 0.) + ( 1 - self.beta2 ) * ( grad ** 2 )
          )

          m_param_hat = self.m[param] / ( 1 - self.beta1 ** self.t )
          v_param_hat = self.v[param] / ( 1 - self.beta2 ** self.t )

          param.data = param.data - self.lr * m_param_hat / (v_param_hat**0.5 + self.eps)