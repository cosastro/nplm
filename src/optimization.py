import numpy
import theano
import theano.tensor as T
import timeit

class TrainingAlgorithm():
    def __init__(self, classifier, learningRate=0.01, batchSize=500, coefficientL1=0.00, coefficientL2=0.0001):
        self.classifier = classifier
        self.learningRate = learningRate
        self.coefficientL1 = coefficientL1
        self.coefficientL2 = coefficientL2
        self.batchSize = batchSize

        floatX = theano.config.floatX
        inputPlaceholder = numpy.empty((1,1), dtype=floatX)
        outputPlaceholder = numpy.empty((1,), dtype=floatX)

        self.trainInput = theano.shared(inputPlaceholder, borrow=True)
        self.trainOutput = theano.shared(outputPlaceholder, borrow=True)

        self.validationInput = theano.shared(inputPlaceholder, borrow=True)
        self.validationOutput = theano.shared(outputPlaceholder, borrow=True)

        castedTrainOutput = T.cast(self.trainOutput, 'int32')
        castedValidationOutput = T.cast(self.validationOutput, 'int32')

        batchIndex = T.lscalar()
        cost = classifier.getCost(classifier.output) #\
               # + self.coefficientL1 * classifier.regularizationL1 \
               # + self.coefficientL2 * classifier.regularizationL2

        parameterGradients = [T.grad(cost, parameter) for parameter in classifier.parameters]

        updates = [
            (parameter, parameter - learningRate * parameterGradient)
            for parameter, parameterGradient in zip(classifier.parameters, parameterGradients)
        ]

        self.trainClassifier = theano.function(
            inputs=[batchIndex],
            outputs=cost,
            updates=updates,
            givens={
                classifier.input: self.trainInput[batchIndex * self.batchSize: (batchIndex + 1) * self.batchSize],
                classifier.output: castedTrainOutput[batchIndex * self.batchSize: (batchIndex + 1) * self.batchSize]
            }
        )

        self.validateClassifier = theano.function(
            inputs=[batchIndex],
            outputs=classifier.zeroOneLoss(classifier.output),
            givens={
                classifier.input: self.validationInput[batchIndex * self.batchSize: (batchIndex + 1) * self.batchSize],
                classifier.output: castedValidationOutput[batchIndex * self.batchSize: (batchIndex + 1) * self.batchSize]
            }
        )

    def train(self, trainingData, validationData,
              epochs=1000, patience = 5000, patienceIncrease = 2, improvementThreshold = 0.995):
        trainInput, trainOutput = trainingData
        validationInput, validationOutput = validationData

        asarray = lambda x: numpy.asarray(x, dtype=theano.config.floatX)

        trainInput = asarray(trainInput)
        trainOutput = asarray(trainOutput)
        validationInput = asarray(validationInput)
        validationOutput = asarray(validationOutput)

        # self.classifier.inputShape.set_value((500, 1, 28, 28))
        self.trainInput.set_value(trainInput)
        self.trainOutput.set_value(trainOutput)
        self.validationInput.set_value(validationInput)
        self.validationOutput.set_value(validationOutput)

        trainingBatchesCount = trainInput.shape[0] / self.batchSize
        validationBatchesCount = validationInput.shape[0] / self.batchSize

        print '... training the model'

        validationFrequency = min(trainingBatchesCount, patience / 2)
        bestValidationLoss = numpy.inf
        doneLooping = False
        epoch = 0

        while (epoch < epochs) and (not doneLooping):
            epoch = epoch + 1
            for trainingBatchIndex in xrange(trainingBatchesCount):

                startTime = timeit.default_timer()
                self.trainClassifier(trainingBatchIndex)
                endTime = timeit.default_timer()

                elapsed = (endTime - startTime) * 100

                iteration = (epoch - 1) * trainingBatchesCount + trainingBatchIndex

                if (iteration + 1) % validationFrequency == 0:
                    validationLosses = [self.validateClassifier(i) for i in xrange(validationBatchesCount)]
                    validationLoss = numpy.mean(validationLosses)

                    print 'epoch: {0}, elapsed: {2}s, validation error: {1} %'.format(epoch, validationLoss * 100., elapsed)

                    if validationLoss < bestValidationLoss:
                        if validationLoss < bestValidationLoss * improvementThreshold:
                            patience = max(patience, iteration * patienceIncrease)

                        bestValidationLoss = validationLoss

                if patience <= iteration:
                    doneLooping = True
                    break

        print 'Optimization complete with best validation score of {0} %'.format(bestValidationLoss * 100.)
