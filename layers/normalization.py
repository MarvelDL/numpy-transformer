import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):

        self.d_model = d_model
        self.eps = eps

        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

    def forward(self, x):

        mean = np.mean(x, axis= -1, keepdims=True)
        var  = np.var(x, axis =-1, keepdims=True)


        x_norm = (x - mean)/np.sqrt(var + self.eps)

        return self.gamma * x_norm + self.beta
    

class AddNorm:
    def __init__(self, d_model, eps = 1e-6):

        self.layer_norm = LayerNorm(d_model, eps)

    def forward(self, x, sublayer_output):

        residual =  x + sublayer_output

        return self.layer_norm.forward(residual)

