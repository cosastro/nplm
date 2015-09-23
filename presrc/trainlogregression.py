import numpy
import theano
import theano.tensor as T
import gzip
import cPickle
import log
import time


def loadData(filePath, borrow=True, floatX=theano.config.floatX):
    with gzip.open(filePath, 'rb') as file:
        trainingData, validationData, testingData = cPickle.load(file)

    data = []
    for input, targetOutput in [trainingData, validationData, testingData]:
        sharedInput = numpy.asarray(input, dtype=floatX)
        sharedInput = theano.shared(sharedInput, borrow=borrow)

        sharedTargetOutput = numpy.asarray(targetOutput, dtype=floatX)
        sharedTargetOutput = theano.shared(sharedTargetOutput, borrow=borrow)
        sharedTargetOutput = T.cast(sharedTargetOutput, 'int32')

        data.append((sharedInput, sharedTargetOutput))

    return tuple(data)

def trainModel(trainingData, validationData, epochs, batchSize, learningRate, inputsCount, outputsCount):
    trainingInput, trainingTargetOutput = trainingData
    validationInput, validationTargetOutput = validationData

    floatX = theano.config.floatX

    defaultWeight = numpy.zeros((inputsCount, outputsCount), dtype=floatX)
    defaultBias = numpy.zeros((outputsCount,), dtype=floatX)

    weight = theano.shared(defaultWeight, name='weight', borrow=True)
    bias = theano.shared(defaultBias, name='bias', borrow=True)

    input = T.matrix('input', dtype=floatX)
    output = T.nnet.softmax(T.dot(input, weight) + bias)
    targetOutput = T.ivector('targetOutput')

    cost = -T.mean(T.log(output)[T.arange(targetOutput.shape[0]), targetOutput])
    performance = T.mean(T.neq(T.argmax(output, axis=1), targetOutput))
    batchIndex = T.lscalar('index')

    validateModel = theano.function(
        inputs=[batchIndex],
        outputs=performance,
        givens={
            input: validationInput[batchIndex * batchSize: (batchIndex + 1) * batchSize],
            targetOutput: validationTargetOutput[batchIndex * batchSize: (batchIndex + 1) * batchSize]
        }
    )

    parameters = [weight, bias]
    gradients = [T.grad(cost, wrt=parameter) for parameter in parameters]
    updates = [(parameter, parameter - learningRate * gradient) for parameter, gradient in zip(parameters, gradients)]

    trainModel = theano.function(
        inputs=[batchIndex],
        outputs=cost,
        updates=updates,
        givens={
            input: trainingInput[batchIndex * batchSize: (batchIndex + 1) * batchSize],
            targetOutput: trainingTargetOutput[batchIndex * batchSize: (batchIndex + 1) * batchSize]
        }
    )

    log.info('Training model...')

    trainingBatchesCount = trainingInput.get_value(borrow=True).shape[0] / batchSize
    validationBatchesCount = validationInput.get_value(borrow=True).shape[0] / batchSize

    for epoch in xrange(epochs):
        for trainBatchIndex in xrange(trainingBatchesCount):
            trainModel(trainBatchIndex)

        validationLosses = [validateModel(validationBatchIndex) for validationBatchIndex in xrange(validationBatchesCount)]
        validationLoss = numpy.mean(validationLosses) * 100

        message = 'Validation loss: {0:.3f}%'.format(validationLoss)
        log.progress(epoch + 1, epochs, message)

    log.newline()
    log.info('Model training complete.')


def main():
    filePath = '../data/MNIST/mnist.pkl.gz'
    trainingData, validationData, testingData = loadData(filePath)

    inputsCount = 784
    outputsCount = 10

    floatX = theano.config.floatX

    trainModel(trainingData, validationData, 20, 500, 0.13, inputsCount, outputsCount)


if __name__ == '__main__':
    main()
