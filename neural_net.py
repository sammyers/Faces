import numpy as np
import scipy.optimize

#!/usr/bin/env python2

class NeuralNet(object):
    """docstring for NeuralNet"""
    def __init__(self, input_size=2, hidden_size=3, output_size=1, Lambda=0.0001):
        super(NeuralNet, self).__init__()
        self.input_layer_size = input_size
        self.output_layer_size = output_size
        self.hidden_layer_size = hidden_size

        self.Lambda = Lambda
    
        self.W1 = np.random.randn(
            self.input_layer_size,
            self.hidden_layer_size
        )

        self.W2 = np.random.randn(
            self.hidden_layer_size,
            self.output_layer_size
        )

    def forward(self, X):
        # propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)

        y_hat = self.sigmoid(self.z3)

        return y_hat

    def sigmoid(self, z):
        # normalize to (0,1)
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def cost_function(self, X, y):
        self.y_hat = self.forward(X)
        J = 0.5 * sum((y - self.y_hat) ** 2)# / X.shape[0] + (self.Lambda / 2) * (sum(self.W1 ** 2) + sum(self.W2 ** 2))

        # Add regularization term
        # J += 

        return J

    def cost_function_prime(self, X, y):
        # compute derivative with respect to W1 and W2

        self.y_hat = self.forward(X)

        delta3 = np.multiply(-(y - self.y_hat), self.sigmoid_prime(self.z3))

        # Add gradient of regularization term
        dJdW2 = np.dot(self.a2.T, delta3)# + self.Lambda * self.W2

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)

        # Add gradient of regularization term
        dJdW1 = np.dot(X.T, delta2)# + self.Lambda * self.W1

        return dJdW1, dJdW2

    def get_params(self):
        # get W1 and W2 rolled into a vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))

        return params

    def set_params(self, params):
        # set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end = self.hidden_layer_size * self.input_layer_size
        self.W1 = np.reshape(
            params[W1_start:W1_end],
            (self.input_layer_size, self.hidden_layer_size)
        )

        W2_end = W1_end + self.hidden_layer_size * self.output_layer_size
        self.W2 = np.reshape(
            params[W1_end:W2_end],
            (self.hidden_layer_size, self.output_layer_size)
        )

    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def compute_numerical_gradent(N, X, y):
        params_initial = N.get_params()
        num_grad = np.zeros(params_initial.shape)

        perturb = np.zeros(params_initial.shape)

        e = 1e-4

        for p in range(len(params_initial)):
            # set perturation vector

            perturb[p] = e
            N.set_params(params_initial + perturb)
            loss2 = N.cost_function(X, y)

            N.set_params(params_initial - perturb)
            loss1 = N.cost_function(X, y)

            # compute numerical gradient
            num_grad[p] = (loss2 - loss1) / (2 * e)

            # reset the value we changed back to zero
            perturb[p] = 0

        # reset params to original values
        N.set_params(params_initial)

        return num_grad

class NetTrainer(object):
    """docstring for NetTrainer"""
    def __init__(self, N):
        super(NetTrainer, self).__init__()
        self.N = N

    def cost_function_wrapper(self, params, X, y):
        self.N.set_params(params)

        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X, y)

        return cost, grad

    def callback_func(self, params):
        self.N.set_params(params)
        self.J.append(self.N.cost_function(self.X, self.y))

    def train(self, X, y):
        # make internal vars for callbacks

        self.X = X
        self.y = y

        self.J = []

        params0 = self.N.get_params()

        options = {'maxiter': 2000, 'disp': True}
        _res = scipy.optimize.minimize(
            self.cost_function_wrapper,
            params0,
            jac = True,
            method = 'BFGS',
            args = (X, y),
            options = options,
            callback = self.callback_func
        )

        self.N.set_params(_res.x)

        self.optimization_results = _res


if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt

    bos = datasets.load_boston()

    nn = NeuralNet(input_size=bos.data.shape[1]-1, output_size=1)

    trainer = NetTrainer(nn)

    bos_inputs = bos.data[:-100, :-1]
    input_norm_max = np.amax(bos_inputs, axis=0)
    bos_inputs = bos_inputs / input_norm_max

    bos_outputs = bos.data[:-100, -1].reshape((406,1))
    output_norm_max = np.amax(bos_outputs, axis=0)
    bos_outputs = bos_outputs / output_norm_max

    trainer.train(bos_inputs, bos_outputs)
    # trainer.train(X, y)

    # print(trainer.J)

    plt.plot(trainer.J)
    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


    print(nn.forward(bos.data[-100:, :-1])*output_norm_max)




def test_some_stuff():
    X = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
    y = np.array(([75], [82], [93], [70]), dtype=float)

    X = X / np.amax(X, axis=0)
    y = y / 100

    #Test network for various combinations of sleep/study:
    hoursSleep = np.linspace(0, 10, 100)
    hoursStudy = np.linspace(0, 5, 100)

    #Normalize data (same way training data way normalized)
    hoursSleepNorm = hoursSleep/10.
    hoursStudyNorm = hoursStudy/5.

    #Create 2-d versions of input for plotting
    a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

    #Join into a single input matrix:
    allInputs = np.zeros((a.size, 2))
    allInputs[:, 0] = a.ravel()
    allInputs[:, 1] = b.ravel()

    allOutputs = nn.forward(allInputs)

    #Contour Plot:
    yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
    xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

    CS = plt.contour(xx,yy,100*allOutputs.reshape(100, 100))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('Hours Sleep')
    plt.ylabel('Hours Study')
    plt.show()

    ###

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100))

    ax.set_xlabel('Hours Sleep')
    ax.set_ylabel('Hours Study')
    ax.set_zlabel('Test Score')
    plt.show()