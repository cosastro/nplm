import gzip
from os.path import exists as fileExists
from nplm import *
from optimization import *

if __name__ == '__main__':
    dataFilePath = '../data/MNIST/mnist.pkl.gz'
    parametersFilePath = '../data/Model/model_parameters.bin'

    file = gzip.open(dataFilePath, 'rb')
    trainingData, validationData, testData = cPickle.load(file)
    file.close()

    testInput, testOutput = testData
    testInput, testOutput = testInput[:30], testOutput[:30]

    print 'Expected values: {0}'.format(testOutput)

    classifier = MultiLayerPerceptron([784, 500, 10])

    output = classifier.classify(testInput)
    print 'Not trained:     {0}'.format(output)

    classifierFile = 'models/mlp.pkl'
    if fileExists(classifierFile):
        with open(classifierFile) as file:
            classifier = MultiLayerPerceptron.load(file)
    else:
        classifier = MultiLayerPerceptron([784, 500, 10])

        trainingAlgorithm = TrainingAlgorithm(classifier)
        trainingAlgorithm.train(trainingData, validationData, 100)

        with open(classifierFile, 'w') as file:
            classifier.dump(file)

    output = classifier.classify(testInput)

    print 'Trained:         {0}'.format(output)
