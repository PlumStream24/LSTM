import numpy as np
import functools

#sigmoid
def sigmoid(sop):
    if type(sop) in [list, tuple]:
        sop = np.array(sop)
    return 1.0 / (1 + np.exp(-1 * sop))

#tanh activation
def tanh_activation(X):
    return np.tanh(X)

#softmax activation
def softmax(sop):
	e = np.exp(sop)
	return e / e.sum()

def relu(sop):
    if not (type(sop) in [list, tuple, np.ndarray]):
        if sop < 0:
            return 0
        else:
            return sop
    elif type(sop) in [list, tuple]:
        sop = np.array(sop)

    result = sop
    result[sop < 0] = 0

    return result

def linear(sop):
    return sop

class Input:
    def __init__(self, input_shape):

        # If the input sample has less than 2 dimensions, then an exception is raised.
        if len(input_shape) < 2:
            raise ValueError("The input class creates an input layer for data inputs with at least 2 dimensions but ({num_dim}) dimensions found.".format(num_dim=len(input_shape)))
        # If the input sample has exactly 2 dimensions, the third dimension is set to 1.
        elif len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 1)

        for dim_idx, dim in enumerate(input_shape):
            if dim <= 0:
                raise ValueError("The dimension size of the inputs cannot be <= 0. Please pass a valid value to the 'input_size' parameter.")

        self.input_shape = input_shape # Shape of the input sample.
        self.output_size = input_shape # Shape of the output from the current layer. For an input layer, it is the same as the shape of the input sample.


class DenseLayer:
    def __init__(self, n_neurons, prev_layer, activation='sigmoid'):
        self.n_neurons = n_neurons
        self.delta_weights = 0

        if (activation == 'sigmoid'):
            self.activation = sigmoid
        elif (activation == 'relu'):
            self.activation = relu
        elif (activation == 'softmax'):
            self.activation = softmax
        elif (activation == 'linear'):
            self.activation = linear

        self.prev_layer = prev_layer

        self.init_weights = np.random.uniform(low=-0.1, high=0.1, size=(self.prev_layer.output_size, self.n_neurons))
        self.trained_weights = self.init_weights.copy()

        self.input_size = self.prev_layer.output_size

        self.output_size = self.n_neurons

        self.output = None

    def dense(self, input):
        self.input = input
        sop = np.matmul(input, self.init_weights)
        self.output = self.activation(sop)

class Flatten:
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        self.input_size = self.prev_layer.output_size
        self.output_size = functools.reduce(lambda x, y: x*y, self.prev_layer.output_size)
        self.output = None
    
    def flatten(self, input):
        self.output_size = input.size
        self.output = np.ravel(input)

class Model:

    def __init__(self, epoch=10, learning_rate=0.01):
        self.network_layers = []
        self.epoch = epoch
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.network_layers.append(layer)
        
    def feed_forward(self, sample):
        last_output = sample

        for layer in self.network_layers:

            if type(layer) is DenseLayer:
                layer.dense(input=last_output)
            elif type(layer) is Flatten:
                layer.flatten(input=last_output)
            elif type(layer) is LSTMLayer:
                layer.forward_propagation(input=last_output)
            elif type(layer) is Input:
                pass
            else:
                raise TypeError("The layer of type {layer_type} is not supported yet.".format(layer_type=type(layer)))

            last_output = layer.output
            
        return last_output

    
    def summary(self):

        print("\n----------Network Architecture----------")
        for layer in self.network_layers:
            print(type(layer))
        print("----------------------------------------\n")


class LSTMLayer:
    def __init__(self, prev_layer, hidden_units):
        #initialize the parameters with 0 mean and 0.01 standard deviation
        mean = 0
        std = 1

        self.input_units = prev_layer.output_size[2]
        self.hidden_units = hidden_units
        
        #lstm cell weights
        self.forget_gate_weights = np.random.normal(mean,std,(self.input_units+hidden_units,hidden_units))
        self.input_gate_weights  = np.random.normal(mean,std,(self.input_units+hidden_units,hidden_units))
        self.output_gate_weights = np.random.normal(mean,std,(self.input_units+hidden_units,hidden_units))
        self.cell_gate_weights   = np.random.normal(mean,std,(self.input_units+hidden_units,hidden_units))
        
        self.output_size = (1, hidden_units)
        self.output = None
    

    def lstm_cell(self, data, prev_activation_matrix, prev_cell_matrix):
        #get parameters
        fgw = self.forget_gate_weights
        igw = self.input_gate_weights
        ogw = self.output_gate_weights
        cgw = self.cell_gate_weights
        
        #concat batch data and prev_activation matrix
        concat_data = np.concatenate((data,prev_activation_matrix),axis=1)

        #forget gate activations
        fa = np.matmul(concat_data,fgw)
        fa = sigmoid(fa)
        
        #input gate activations
        ia = np.matmul(concat_data,igw)
        ia = sigmoid(ia)
        
        #output gate activations
        oa = np.matmul(concat_data,ogw)
        oa = sigmoid(oa)
        
        #gate gate activations
        ca = np.matmul(concat_data,cgw)
        ca = tanh_activation(ca)
        
        #new cell memory matrix
        cell_memory_matrix = np.multiply(fa,prev_cell_matrix) + np.multiply(ia,ca)
        
        #current activation matrix
        activation_matrix = np.multiply(oa, tanh_activation(cell_memory_matrix))

        return cell_memory_matrix, activation_matrix

    def forward_propagation(self, input):
        timestep = input.shape[0]
        n_dim = input.shape[1]
        
        #initial activation_matrix(a0) and cell_matrix(c0)
        a0 = np.zeros([1, self.hidden_units],dtype=np.float32)
        c0 = np.zeros([1, self.hidden_units],dtype=np.float32)
        
        #unroll the names
        for i in range(len(input)):
            #get first first character batch
            data = input[i].reshape(1,n_dim)
            
            #lstm cell
            ct,at = self.lstm_cell(data, a0, c0)
            
            #update a0 and c0 to new 'at' and 'ct' for next lstm cell
            a0 = at
            c0 = ct
        
        self.output = at