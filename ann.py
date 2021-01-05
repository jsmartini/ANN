import numpy as np

def create_net(input, shape):
    net = []
    for layer in shape:
        l = []
        for _ in range(layer):
            l.append(np.random.uniform(-1, 1, input))
        l = np.array(l)
        net.append(l)
        input = layer
    return net


from abc import ABC, abstractmethod

class function(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def f(self, z):
        pass

    @abstractmethod
    def df(self, z):
        pass

    def __str__(self):
        return self.name

class logistic(function):

    def __init__(self):
        super().__init__("logistic")

    def f(self, z):
        return 1/(1+np.exp(-z))

    def df(self, z):
        return (-np.exp(-z))/((1+np.exp(-z))**2)

class MSE(function):

    def __init__(self):
        super().__init__("Mean Square Error")

    def f(self, y, yp):
        return 0.5 * np.abs(y - yp)**2

    def df(self, y, yp):
        return np.abs(y - yp)**2


class network:

    def __init__(self, input, hidden, activiation, cost):

        self.net = create_net(input=input, shape = hidden)
        self.a = activiation()
        self.c = cost()

    def feedforward(self, x):
        a = np.full(shape=(self.net[1].shape[0], len(x)), fill_value=x)
        #print(a.shape)
        #print(self.net[0].shape)
#        return np.dot(a, self.net[0].T)

        for i in range(len(self.net)):
            z = []
            for j in range(len(self.net[i])):
                z.append(np.dot(x, self.net[i][j]))
            x = np.vectorize(self.a.f)(z)
            print(x.shape)
        return x



    def backprop(self, x, y):
        debug = True
        A= []
        dA = []
        Z = [x]
        gradient_w = []
        for i, _ in enumerate(self.net):

            z = []
            for j in range(len(self.net[i])):

                z.append(np.dot(x, self.net[i][j]))
            z = np.array(z)
            Z.append(z)
            A.append(np.vectorize(self.a.f)(z))
            dA.append(np.vectorize(self.a.df)(z))
            del z
            x = A[i]
        dA.insert(0, np.vectorize(self.a.df)(x))
        y_e = x
        output_error = np.multiply(self.c.f(y_e, y), dA[-1])
        A.insert(0,np.vectorize(self.a.f)(x))
        i = len(self.net) - 1

        while i >= 0:
            print(i)
            print(output_error.shape)
            print("n")
            print(self.net[i].T.shape)
            print(dA[i].shape)
            gradient_w.append(output_error)
            output_error = np.multiply(
                np.matmul(self.net[i].T, output_error),
                dA[i]
            )
            i -= 1

        gradient_w.append(output_error)
        return list(reversed(gradient_w))



if __name__ == "__main__":
    input_sz = 784
    output_sz = 10
    training_x = np.random.random(size=(100, input_sz))
    training_y = np.random.random(size=(100, output_sz))
    #print(training_x.shape)
    #print(training_y.shape)
    net = network(
        input=input_sz,
        hidden= [100, 30, 10],
        activiation=logistic,
        cost=MSE)
    #print(net.feedforward(training_x[0]))
    w = net.backprop(training_x[0], training_y[0])
    #for l in w:
        #print(l.shape)
