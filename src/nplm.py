import numpy
import theano
import theano.tensor as T
import cPickle
import time

class Layer(object):
    def __init__(self, inputsCount, outputsCount, defaultWeight=None, defaultBias=None, activationFunction=T.tanh, input=None):
        self.inputsCount = inputsCount
        self.outputsCount = outputsCount

        self.weight = self.initializeWeight(defaultWeight, inputsCount, outputsCount, activationFunction)
        self.bias = self.initializeBias(defaultBias, outputsCount, activationFunction)

        self.regularizationL1 = abs(self.weight).sum()
        self.regularizationL2 = (self.weight ** 2).sum()

        self.activationFunction = activationFunction

        self.parameters = [self.weight, self.bias]

        self.input = input if input is not None else T.matrix('input')

        self.weightedOutput = T.dot(self.input, self.weight) + self.bias
        self.output = self.weightedOutput if self.activationFunction is None else self.activationFunction(self.weightedOutput)

    def initializeWeight(self, defaultWeight, inputsCount, outputsCount, activationFunction):
        if defaultWeight is None:
            if activationFunction is None:
                weight = numpy.random.rand(inputsCount, outputsCount)
            else:
                seed = int(time.time() % 1 * 10000)
                randomNumberGenerator = numpy.random.RandomState(seed)
                bound = numpy.sqrt(6. / (inputsCount + outputsCount))

                weight = randomNumberGenerator.uniform(low=-bound, high=bound, size=(inputsCount, outputsCount))

                if activationFunction == T.nnet.sigmoid:
                    weight *= 4

            weight = numpy.asarray(weight, dtype=theano.config.floatX)
        else:
            weight = defaultWeight

        return theano.shared(value=weight, name='weight', borrow=True)

    def initializeBias(self, defaultBias, outputsCount, activationFunction):
        bias = numpy.zeros((outputsCount,), dtype=theano.config.floatX)
        return theano.shared(value=bias, name='bias', borrow=True)

    def dump(self):
        return {
            'inputsCount': self.inputsCount,
            'outputsCount': self.outputsCount,
            'weight': self.weight.get_value(),
            'bias': self.bias.get_value()
        }

    @staticmethod
    def load(configuration, activationFunction=T.tanh, input=None):
        inputsCount = configuration['inputsCount']
        outputsCount = configuration['outputsCount']
        weight = configuration['weight']
        bias = configuration['bias']

        return Layer(inputsCount, outputsCount, weight, bias, activationFunction, input)

class NeuralNetwork(object):
    def __init__(self, sizes=None, hiddenActivationFunction=T.tanh, outputActivationFunction=T.nnet.softmax, layers=None):
        self.inputsCount = sizes[0] if sizes is not None else layers[0]
        self.outputsCount = sizes[-1] if sizes is not None else layers[-1]

        self.input = T.matrix('input')
        self.output = T.ivector('output')

        self.layers = []

        #check if local variable may store a value other than class property with the same name
        input = self.input

        layerConfigs = []
        if sizes is not None:
            shapes = zip(sizes[:-1], sizes[1:])

            for inputsCount, outputsCount in shapes[:-1]:
                layerConfigs.append((inputsCount, outputsCount, None, None, hiddenActivationFunction))

            inputsCount, outputsCount = shapes[-1]
            layerConfigs.append((inputsCount, outputsCount, None, None, outputActivationFunction))
        elif layers is not None:
            for l in layers[:-1]:
                layerConfigs.append((l['inputsCount'], l['outputsCount'], l['weight'], l['bias'], hiddenActivationFunction))

            l = layers[-1]
            layerConfigs.append((l['inputsCount'], l['outputsCount'], l['weight'], l['bias'], outputActivationFunction))

        for inputsCount, outputsCount, weight, bias, activationFunction in layerConfigs:
            layer = Layer(inputsCount, outputsCount, weight, bias, activationFunction, input)
            self.layers.append(layer)

            input = layer.output

        self.activationFunction = self.layers[-1].activationFunction

        #check if sum works properly with theano.shared
        self.regularizationL1 = sum([l.regularizationL1 for l in self.layers])
        self.regularizationL2 = sum([l.regularizationL2 for l in self.layers])

        layersParameters = [l.parameters for l in self.layers]
        self.parameters = [parameter for layerParameters in layersParameters for parameter in layerParameters]

        #check if minus applied to T.log produce the same result as minus applied to T.mean
        self.costFunction = T.log(self.layers[-1].output)

    def getCost(self, output):
        examplesCount = output.shape[0]
        logLikelihoodIndeces = T.arange(examplesCount)
        logLikelihood = self.costFunction[logLikelihoodIndeces, output]

        return -T.mean(logLikelihood)

    def dump(self, file):
        layers = [layer.dump() for layer in self.layers]
        cPickle.dump(layers, file)

    @staticmethod
    def load(file):
        layers = cPickle.load(file)
        return NeuralNetwork(layers=layers)

class MultiLayerPerceptron(NeuralNetwork):
    def __init__(self, sizes=None, layers=None):
        super(MultiLayerPerceptron, self).__init__(sizes, T.tanh, T.nnet.softmax, layers)

        self.getClassMembership = T.argmax(self.layers[-1].output, axis=1)

        self.classificationFunction = theano.function(inputs=[self.input], outputs=self.getClassMembership)

    def zeroOneLoss(self, output):
        if output.ndim != self.getClassMembership.ndim:
            raise TypeError(
                'output {0} should have the same shape as self.getClassMembership {1}'.format(
                    output.type,
                    self.getClassMembership.type))

        if output.dtype.startswith('int'):
            return T.mean(T.neq(self.getClassMembership, output))
        else:
            raise NotImplementedError()

    def classify(self, input):
        return self.classificationFunction(input)

    @staticmethod
    def load(file):
        layers = cPickle.load(file)
        return MultiLayerPerceptron(layers=layers)
