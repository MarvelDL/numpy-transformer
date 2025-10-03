import numpy as np

class FeedForwardNetwork:
    def __init__(self, d_model, d_ff):

        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = np.random.randn(d_model, d_ff)*0.01
        self.b1 = np.zeros((d_ff,))


        self.W2 = np.random.randn(d_ff, d_model)*0.01
        self.b2 = np.zeros((d_model,))

    
    def forward(self, x):

        #First Layer
        hidden = x @ self.W1 + self.b1

        #ReLU
        hidden = np.maximum(0, hidden)

        #Second linear Layer
        output = hidden @ self.W2 + self.b2

        return output


        
