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


def main():
    inputsCount = 28 * 28
    outputsCount = 10

    floatX = theano.config.floatX

    defaultWeight = numpy.zeros((inputsCount, outputsCount), dtype=floatX)
    weight = theano.shared(defaultWeight, name='weight', borrow=True)

    defaultBias = numpy.zeros((outputsCount,), dtype=floatX)
    bias = theano.shared(defaultBias, name='bias', borrow=True)

    input = T.matrix('input', dtype=floatX)

    output = T.nnet.softmax(T.dot(input, weight) + bias)

    targetOutput = T.ivector('targetOutput')

    cost = -T.mean(T.log(output)[T.arange(targetOutput.shape[0]), targetOutput])

    performance = T.mean(T.neq(T.argmax(output, axis=1), targetOutput))

    filePath = '../data/MNIST/mnist.pkl.gz'
    trainingData, validationData, testingData = loadData(filePath)

    trainingInput, trainingTargetOutput = trainingData
    validationInput, validationTargetOutput = validationData
    testingInput, testingTargetOutput = testingData

    batchSize = 500

    trainingBatchesCount = trainingInput.get_value(borrow=True).shape[0] / batchSize
    validationBatchesCount = validationInput.get_value(borrow=True).shape[0] / batchSize
    testingBatchesCount = testingInput.get_value(borrow=True).shape[0] / batchSize

    batchIndex = T.lscalar('index')

    testModel = theano.function(
        inputs=[batchIndex],
        outputs=performance,
        givens={
            input: testingInput[batchIndex * batchSize: (batchIndex + 1) * batchSize],
            targetOutput: trainingTargetOutput[batchIndex * batchSize: (batchIndex + 1) * batchSize]
        }
    )

    validateModel = theano.function(
        inputs=[batchIndex],
        outputs=performance,
        givens={
            input: validationInput[batchIndex * batchSize: (batchIndex + 1) * batchSize],
            targetOutput: validationTargetOutput[batchIndex * batchSize: (batchIndex + 1) * batchSize]
        }
    )

    learningRate = 0.13
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

    epochs = 100
    for epoch in xrange(epochs):
        for trainBatchIndex in xrange(trainingBatchesCount):
            trainModel(trainBatchIndex)

        validationLosses = [validateModel(validationBatchIndex) for validationBatchIndex in xrange(validationBatchesCount)]
        validationLoss = numpy.mean(validationLosses) * 100

        message = 'Validation loss: {0:.3f}%'.format(validationLoss)
        log.progress(epoch + 1, epochs, message)

    log.newline()
    log.info('Model training complete.')


if __name__ == '__main__':
    main()
