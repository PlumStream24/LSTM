## Classes Definition for CNN
## M Iqbal Sigid - 13519152

import numpy
import functools

def relu(sop):

    if not (type(sop) in [list, tuple, numpy.ndarray]):
        if sop < 0:
            return 0
        else:
            return sop
    elif type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    result = sop
    result[sop < 0] = 0

    return result

def drelu(sop):
    if not (type(sop) in [list, tuple, numpy.ndarray]):
        if sop < 0:
            return 0
        else:
            return 1
    elif type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    result = sop
    result[sop < 0] = 0
    result[sop > 0] = 1

    return result

def sigmoid(sop):

    if type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    return 1.0 / (1 + numpy.exp(-1 * sop))

def dsigmoid(sop):
    s = sigmoid(sop)
    return s * (1 - s)

def softmax(sop):
	e = numpy.exp(sop)
	return e / e.sum()

def dsoftmax(sop):
    soft = softmax(sop)
    diag_soft = soft*(1- soft)
    return diag_soft

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

class ConvLayer:
    def __init__(self, prev_layer, n_filters, filter_size, padding, stride):
        self.activation = relu
        self.activation_dfn = drelu
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride

        self.prev_layer = prev_layer

        self.filter_bank_size = (self.n_filters,
                                 self.filter_size, 
                                 self.filter_size, 
                                 self.prev_layer.output_size[-1])

        self.init_weights = numpy.random.uniform(low=-0.1, high=0.1, size=self.filter_bank_size)
        self.trained_weights = self.init_weights.copy()

        self.input_size = self.prev_layer.output_size

        self.output_size = ((self.prev_layer.output_size[0] - self.filter_size + 2*self.padding) / self.stride + 1, 
                            (self.prev_layer.output_size[1] - self.filter_size + 2*self.padding) / self.stride + 1, 
                            self.n_filters)
        
        self.output = None

        self.delta_weights = numpy.zeros(((self.filter_bank_size[1], self.filter_bank_size[2], self.input_size[2], self.filter_bank_size[3])))
    
    def conv_(self, input, conv_filter):

        result = numpy.zeros((input.shape[0], input.shape[1], conv_filter.shape[0]))
        
        for r in numpy.uint16(numpy.arange(self.filter_bank_size[1]/2, 
                                input.shape[0]-self.filter_bank_size[1]/2+1,
                                    self.stride)):
            for c in numpy.uint16(numpy.arange(self.filter_bank_size[1]/2, 
                                    input.shape[1]-self.filter_bank_size[1]/2+1,
                                        self.stride)):
                
                curr_region = input[r-numpy.uint16(numpy.floor(self.filter_bank_size[1]/2)):r+numpy.uint16(numpy.ceil(self.filter_bank_size[1]/2)), 
                                        c-numpy.uint16(numpy.floor(self.filter_bank_size[1]/2)):c+numpy.uint16(numpy.ceil(self.filter_bank_size[1]/2)), :]
                
                # Element-wise multipliplication between the current region and the filter.
                for filter_idx in range(conv_filter.shape[0]):
                    curr_result = curr_region * conv_filter[filter_idx]
                    conv_sum = numpy.sum(curr_result)
    
                    result[r, c, filter_idx] = self.activation(conv_sum)

        # Clipping the outliers of the result matrix.
        final_result = result[numpy.uint16(self.filter_bank_size[1]/2):result.shape[0]-numpy.uint16(self.filter_bank_size[1]/2), 
                              numpy.uint16(self.filter_bank_size[1]/2):result.shape[1]-numpy.uint16(self.filter_bank_size[1]/2), :]
        return final_result
    
    def conv(self, input):

        if len(input.shape) != len(self.init_weights.shape) - 1:
            raise ValueError("Number of dimensions in the conv filter and the input do not match.")  
        if len(input.shape) > 2 or len(self.init_weights.shape) > 3:
            if input.shape[-1] != self.init_weights.shape[-1]:
                raise ValueError("Number of channels in both the input and the filter must match.")
        
        # Filter must be square and odd
        if self.init_weights.shape[1] != self.init_weights.shape[2]:
            raise ValueError('A filter must be a square matrix.')
        if self.init_weights.shape[1]%2==0:
            raise ValueError('A filter must have an odd size.')
        
        self.input = input
        padded_input = numpy.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        self.output = self.conv_(input, self.init_weights)
    

    def backpropagate(self, nx_layer):
        layer = self
        
        layer.delta = numpy.zeros((layer.input_size[0], layer.input_size[1], layer.input_size[2]))
        
        image = layer.input
    
        for f in range(layer.filter_bank_size[0]):
            kshape = (layer.filter_bank_size[1], layer.filter_bank_size[2], layer.input_size[2], layer.filter_bank_size[3])
            shape = layer.input_size
            stride = layer.stride
            rv = 0
            i = 0
            
            for r in range(kshape[0], shape[0]+1, stride):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1]+1, stride):
                    chunk = image[rv:r, cv:c]

                    layer.delta_weights[:, :, :, f] += chunk * nx_layer.delta[i, j, f]
                    
                    layer.delta[rv:r, cv:c, :] += nx_layer.delta[i, j, f] * layer.init_weights[:, :, :, f]
                    
                    j+=1
                    cv+=stride
                rv+=stride
                i+=1
            
            #layer.delta_biases[f] = numpy.sum(nx_layer.delta[:, :, f])
        layer.delta = layer.activation_dfn(layer.delta)




class PoolLayer:
    def __init__(self, prev_layer, pool_size, stride, pool_type="max"):
        self.pool_size = pool_size
        self.prev_layer = prev_layer
        self.stride = stride
        self.pool_type = pool_type

        self.input_size = self.prev_layer.output_size

        self.output_size = (numpy.uint16((self.prev_layer.output_size[0] - self.pool_size + 1)/stride + 1), 
                                numpy.uint16((self.prev_layer.output_size[1] - self.pool_size + 1)/stride + 1), 
                                self.prev_layer.output_size[-1])
        
        self.output = None
    
    def pool(self, input):
        self.input = input
        pool_out = numpy.zeros((numpy.uint16((input.shape[0]-self.pool_size+1)/self.stride+1),
                                numpy.uint16((input.shape[1]-self.pool_size+1)/self.stride+1),
                                input.shape[-1]))
        
        for ch in range(input.shape[-1]):
            r2 = 0
            for r in numpy.arange(0, input.shape[0] - self.pool_size+1, self.stride):
                c2 = 0
                for c in numpy.arange(0, input.shape[1] - self.pool_size+1, self.stride):
                    if self.pool_type == "max":
                        pool_out[r2, c2, ch] = numpy.max([input[r:r+self.pool_size,  c:c+self.pool_size, ch]])
                    elif self.pool_type == "mean":
                        pool_out[r2, c2, ch] = numpy.mean([input[r:r+self.pool_size,  c:c+self.pool_size, ch]])
                    c2 = c2 + 1
                r2 = r2 + 1
        
        self.output = pool_out
    
    def backpropagate(self, nx_layer):
        """
            Gradients are passed through index of largest value .
        """
        layer = self
        stride = layer.stride
        kshape = layer.pool_size
        image = layer.input
        shape = image.shape
        layer.delta = numpy.zeros(shape)
        
        cimg = []
        rstep = stride
        cstep = stride
        
        for f in range(shape[2]):
            i = 0
            rv = 0
            for r in range(kshape, shape[0]+1, rstep):
                cv = 0
                j = 0
                for c in range(kshape, shape[1]+1, cstep):
                    chunk = image[rv:r, cv:c, f]
                    dout = nx_layer.delta[i, j, f]
                    
                    if layer.pool_type == "max":
                        p = numpy.max(chunk)
                        index = numpy.argwhere(chunk == p)[0]
                        layer.delta[rv+index[0], cv+index[1], f] = dout
                    
                    if layer.pool_type == "mean":
                        p = numpy.mean(chunk)
                        layer.delta[rv:r, cv:c, f] = dout

                    j+=1
                    cv+=cstep
                rv+=rstep
                i+=1


class Flatten:
    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        self.input_size = self.prev_layer.output_size
        self.output_size = functools.reduce(lambda x, y: x*y, self.prev_layer.output_size)
        self.output = None
    
    def flatten(self, input):
        self.output_size = input.size
        self.output = numpy.ravel(input)
    
    def backpropagate(self, nx_layer):
        self.error = numpy.dot(nx_layer.init_weights, nx_layer.delta)
        self.delta = self.error * self.output
        self.delta = self.delta.reshape(self.input_size)


class DenseLayer:
    def __init__(self, n_neurons, prev_layer, activation='sigmoid'):
        self.n_neurons = n_neurons
        self.delta_weights = 0

        if (activation == 'sigmoid'):
            self.activation = sigmoid
            self.activation_dfn = dsigmoid
        elif (activation == 'relu'):
            self.activation = relu
            self.activation_dfn = drelu
        elif (activation == 'softmax'):
            self.activation = softmax
            self.activation_dfn = dsoftmax

        self.prev_layer = prev_layer

        self.init_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(self.prev_layer.output_size, self.n_neurons))
        self.trained_weights = self.init_weights.copy()

        self.input_size = self.prev_layer.output_size

        self.output_size = self.n_neurons

        self.output = None

    def dense(self, input):
        self.input = input
        sop = numpy.matmul(input, self.init_weights)
        self.output = self.activation(sop)

    def backpropagate(self, nx_layer):
        self.error = numpy.dot(nx_layer.init_weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.output)
        self.delta_weights += self.delta * numpy.atleast_2d(self.input).T
        #self.delta_biases += self.delta


class Model:

    def __init__(self, epoch=10, learning_rate=0.01):
        self.network_layers = []
        self.epoch = epoch
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.network_layers.append(layer)


    def train(self, X, Y, batch_size):
        network_predictions = []
        #network_error = 0

        len_batch = int(len(X)/batch_size)
        curr_ind = numpy.arange(0, len(X), dtype=numpy.int32)
        if len(curr_ind) % batch_size != 0:
            len_batch+=1
        batches = numpy.array_split(curr_ind, len_batch)

        for epoch in range(self.epoch):
            err = []
            for batch in batches:
                a = [] 
                curr_x, curr_y = X[batch], Y[batch]
                b = 0
                batch_loss = 0
                for x, y in zip(curr_x, curr_y):
                    output = self.feed_forward(x)

                    pred_label = numpy.where(numpy.max(output) == output)[0][0]
                    network_predictions.append(pred_label)
                    
                    loss, error = self.apply_loss(y, output)
                    batch_loss += loss
                    err.append(error)

                    update = False
                    if b == batch_size-1:
                        update = True
                        loss = batch_loss/batch_size
                    self.backpropagate(loss, update)
                    b+=1
                
    def predict(self, data_inputs):

        if (data_inputs.ndim != 4):
            raise ValueError("The data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=data_inputs.ndim))

        predictions = []
        for sample in data_inputs:
            probs = self.feed_sample(sample=sample)
            predicted_label = numpy.where(numpy.max(probs) == probs)[0][0]
            predictions.append(predicted_label)
        return predictions
    
    def feed_forward(self, sample):
        last_output = sample

        for layer in self.network_layers:

            if type(layer) is ConvLayer:
                layer.conv(input=last_output)
            elif type(layer) is DenseLayer:
                layer.dense(input=last_output)
            elif type(layer) is PoolLayer:
                layer.pool(input=last_output)
            elif type(layer) is Flatten:
                layer.flatten(input=last_output)
            elif type(layer) is Input:
                pass
            else:
                raise TypeError("The layer of type {layer_type} is not supported yet.".format(layer_type=type(layer)))

            last_output = layer.output
            
        return last_output
    
    def apply_loss(self, y, out):
        loss = y - out
        mse = numpy.mean(numpy.square(loss), axis=-1)
        return loss, mse
    
    def backpropagate(self, loss, update):
        
        # if it is output layer
        nx_layer = None
        for i in reversed(range(len(self.network_layers))):
            layer = self.network_layers[i]
            if layer == self.network_layers[-1]:
                if (type(layer).__name__ == "DenseLayer"):
                    layer.error = loss
                    layer.delta = layer.error * layer.activation_dfn(layer.output)
                    layer.delta_weights += layer.delta * numpy.atleast_2d(layer.input).T
                    #layer.delta_biases += layer.delta
            elif (type(layer).__name__ == "Input"):
                continue
            else:
                layer.backpropagate(nx_layer)
            
            nx_layer = layer
            
            if update:
                layer.delta_weights /= self.batch_size
                #layer.delta_biases /= self.batch_size

        if update: 
            self.update_weights()

            
    
    def update_weights(self):
        for layer in self.network_layers:
            if "trained_weights" in vars(layer).keys():
                layer.trained_weights = layer.init_weights - self.learning_rate * layer.delta_weights

    
    def summary(self):

        print("\n----------Network Architecture----------")
        for layer in self.network_layers:
            print(type(layer))
        print("----------------------------------------\n")