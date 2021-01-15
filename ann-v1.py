
import numpy as np
import logging
import json
from progress.bar import Bar

# sigmoid activation
sigmoid = {
    "f": lambda z: np.exp(-z) / (1 + np.exp(-z)),
    "df": lambda z: -np.exp(-z) / (1 + np.exp(-z)) ** 2,
    "df_quick": lambda a: a * (1 - a)  # easy
}
# MSE cost
cost = {
    "f": lambda y, yp: 0.5 * (y - yp) ** 2,
    "df": lambda y, yp: y - yp
}

class nn:

    """
        set-backs: only use one activation for all layers
    """

    lr = 0.01
    training_cost = []
    testing_cost = []

    def __init__(self, input, output, hidden :list):

        self.net_w = list()
        self.net_b = list()
        hidden.append(output)
        next = input
        # generates an array of arrays to for the weights
        for hsize in hidden:
            self.net_b.append(np.random.random())
            self.net_w.append(np.array([np.random.random(next) for _ in range(hsize)]))
            next = hsize
        print()

    def get_network(self):
        network = dict()
        for i in enumerate(self.net_w):
            network[f"layer {i[0x0]}"] = {
                "hidden units": i[0x1].shape[0x0],
                "weight length/unit": i[0x1].shape[0x1]
            }
        return json.dumps(
            network, indent=4
        )

    def feed(self, x, f = sigmoid["f"], df = sigmoid["df_quick"], backprop = False):
        A, dA = [], []
        for layer_w, layer_b in zip(self.net_w, self.net_b):
            z = np.dot(layer_w, x) + layer_b
            a = f(z)
            if backprop:
                A.append(a)
                dA.append(df(a))
            x = a
        if backprop: return x, A, dA
        return x

    def backprop(self, x, y):
        y_out, A, dA = self.feed(x, backprop=True)
        e0 = cost["df"](y_out, y)
        self.training_cost.append(cost["f"](y_out, y))
        error = [e0]
        nabla_w = [y_out * e0]  # output layer weights
        nabla_b = [e0]  # output layer bias
        for i in reversed(range(1, len(self.net_w))):
            temp1 = np.matmul(self.net_w[i].T, error[-0x1])
            temp2 = np.multiply(temp1, dA[ i -0x1])
            error.append(temp2)
            nabla_w.insert(0x0, A[ i -0x1 ] *error[-0x1])
            nabla_b.insert(0x0, error[-0x1])
        return nabla_w, nabla_b

    def update(self, nw, nb):
        nb = list(nb)
        nw = list(nw)
        for i in range(len(self.net_w)):
            self.net_w[i] = (self.net_w[i].T - self.lr *nw[i]).T
            self.net_b[i] = self.net_b[i] - self.lr *nb[i]

    def mini_batch_update(self, x_set, y_set, b_size):
        # chunks dataset into subsets
        chunker = lambda dset,  c: [dset[i: i +c] for i in range(0x0, len(dset), c)]
        # chunks (list of features, list of labels) for each chunk // divides the dataset in tuples lists of x,y pairs
        data_chunks = [(cx, cy) for cx, cy in zip(chunker(x_set, b_size), chunker(y_set, b_size))]
        iterations = len(data_chunks)
        bar = Bar("Processing", max=iterations)
        for i in range(iterations):
            bar.next()
            dataset = data_chunks[i]
            n = {}
            cnt = 0x0
            for x, y in zip(dataset[0x0], dataset[0x1]):
                n[cnt] = {}
                nabla_w, nabla_b = self.backprop(x, y)
                for j in range(len(nabla_w)):
                    n[cnt][j] = (nabla_w[j], nabla_b[j])
                cnt += 0x1
            network_keys = n[0x0].keys()
            # lk -> layer keys
            # dw -> gradient for w
            # k -> network iteration key // per example gradient
            nw = []  # nabla weights
            nb = []  # nabla bias
            # adds up all weights in same layers for the entire chunk of data and updates with avg dw
            for lk in network_keys:
                dw = n[0x0][lk][0x0]
                db = n[0x0][lk][0x1]
                for k in list(n.keys())[0x1:]:
                    dw += n[k][lk][0x0]
                    db += n[k][lk][0x1]
                nw.insert(0x0, dw /len(network_keys))
                nb.insert(0x0, db /len(network_keys))
            self.update(reversed(nw), reversed(nb))
            print \
                (f"training loss {np.array(self.training_cost).sum() / len(self.training_cost)} \t dataset chunk iteration {i}")
            self.training_cost = []
        bar.finish()

if __name__ == "__main__":
    """
    testing network functionality: making sure loss changes with each epoch -- indicative of changing weights and biases
    """
    np.random.seed(1)
    # logging.basicConfig(filemode="w", filename="sandbox_nn.py.log", level=logging.INFO)
    input_sz = 100
    output_sz = 10
    hidden = [300, 500, 40]
    net = nn(
        input=input_sz,
        output=output_sz,
        hidden=hidden
    )
    net.lr = 0.0001
    def gen_random_data(amt, input, output):
        x = [np.random.random(input) for _ in range(amt)]
        y = [np.random.random(output) for _ in range(amt)]
        return x, y

    print(net.get_network())
    def pause(msg):
        val = True
        while val:
            d = input("= " *(int(len(msg ) *1.5)) + f"\n{msg} \t(y/yes/n/no)\n "+ "= " *(int(len(msg ) *1.5) ) +"\n>>>")
            if d.upper() == "Y" or d.upper() == "yes".upper():
                val = False
            elif d.upper() == "N" or d.upper() == "no".upper():
                print("BYE!")
                exit(0)
            else:
                print("INVALID OPTION")
                continue

    pause("CONTINUE to mini batch random data test?")
    tx, ty = gen_random_data(1000, input_sz, output_sz)
    epochs = 100
    for i in range(epochs):
        print(f"Epoch {i}\t " + "= " *10)
        net.mini_batch_update(tx, ty, 64)
        pause(f"CONTINUE testing epoch { i +1} of {epochs}?")