from numpy.random import randn as random
import numpy
import time
import os

import theano

import theano.tensor as T

import parameters
import vectors
import log
import validation


class ProbabilisticLanguageModel():
    def __init__(self, fileVocabularySize, wordVocabularySize, contextSize, embeddingSize):
        floatX = theano.config.floatX
        empty = lambda *shape: numpy.empty(shape, dtype='int32')

        defaultWordEmbeddings = random(wordVocabularySize, embeddingSize).astype(dtype=floatX)
        defaultWeight = random(contextSize * embeddingSize, wordVocabularySize).astype(dtype=floatX)
        defaultBias = random(wordVocabularySize).astype(dtype=floatX)

        self.wordEmbeddings = theano.shared(defaultWordEmbeddings, name='wordEmbeddings', borrow=True)
        self.weight = theano.shared(defaultWeight, name='weight', borrow=True)
        self.bias = theano.shared(defaultBias, name='bias', borrow=True)

        parameters = [self.wordEmbeddings, self.weight, self.bias]

        contextIndices = T.imatrix('contextIndices')

        context = self.wordEmbeddings[contextIndices]
        context = context.reshape((contextIndices.shape[0], contextSize * embeddingSize))

        probabilities = T.nnet.softmax(T.dot(context, self.weight) + self.bias)
        targetProbability = T.ivector('targetProbability')

        l1Coefficient = T.scalar('l1Coefficient', dtype=floatX)
        l2Coefficient = T.scalar('l2Coefficient', dtype=floatX)

        l1Regularization = l1Coefficient * sum([abs(p).sum() for p in parameters])
        l2Regularization = l2Coefficient * sum([(p ** 2).sum() for p in parameters])

        cost = -T.mean(T.log(probabilities)[T.arange(targetProbability.shape[0]), targetProbability]) \
               + l1Regularization \
               + l2Regularization


        learningRate = T.scalar('learningRate', dtype=floatX)

        gradients = [T.grad(cost, wrt=p) for p in parameters]
        updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradients)]

        miniBatchIndex = T.lscalar('miniBatchIndex')
        miniBatchSize = T.iscalar('miniBatchSize')

        self.input = theano.shared(empty(1,1), borrow=True)
        self.targetOutput = theano.shared(empty(1), borrow=True)

        self.trainModel = theano.function(
            inputs=[miniBatchIndex, miniBatchSize, learningRate, l1Coefficient, l2Coefficient],
            outputs=cost,
            updates=updates,
            givens={
                contextIndices: self.input[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize],
                targetProbability: self.targetOutput[miniBatchIndex * miniBatchSize: (miniBatchIndex + 1) * miniBatchSize]
            }
        )

        embeddingIndices = T.ivector('embeddingIndices')
        wordEmbeddingSample = self.wordEmbeddings[embeddingIndices]

        self.getWordEmbeddings = theano.function(
            inputs=[embeddingIndices],
            outputs=wordEmbeddingSample
        )


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else self.wordEmbeddings.shape().get_value()[0]
            step = item.step if item.step is not None else 1

            embeddingIndices = [i for i in xrange(start, stop, step)]

            return self.getWordEmbeddings(embeddingIndices)

        return self.getWordEmbeddings([item])[0]


    def train(self, trainingInput, trainingTargetOutput, miniBatchSize, learningRate, l1Coefficient, l2Coefficient):
        asarray = lambda x: numpy.asarray(x, dtype='int32')

        trainingInput = asarray(trainingInput)
        trainingTargetOutput = asarray(trainingTargetOutput)

        self.input.set_value(trainingInput)
        self.targetOutput.set_value(trainingTargetOutput)

        trainInputSize = trainingInput.shape[0]
        trainingBatchesCount = trainInputSize / miniBatchSize + int(trainInputSize % miniBatchSize > 0)

        for trainingBatchIndex in xrange(0, trainingBatchesCount):
            self.trainModel(trainingBatchIndex, miniBatchSize, learningRate, l1Coefficient, l2Coefficient)


    def dump(self, parametersPath, embeddingsPath):
        embeddings = self.wordEmbeddings.get_value()
        parameters.dumpEmbeddings(embeddings, embeddingsPath)


def trainModel(fileVocabulary, wordVocabulary, contextProvider, model, superBatchSize, miniBatchSize, parametersPath, embeddingsPath, learningRate, l1Coefficient, l2Coefficient, epochs, metricsPath):
    if os.path.exists(metricsPath):
        os.remove(metricsPath)

    superBatchesCount = contextProvider.contextsCount / superBatchSize + 1
    startTime = time.time()
    previousTotal = 0

    for epoch in xrange(0, epochs):
        for superBatchIndex in xrange(0, superBatchesCount):
            contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

            fileIndices, wordIndices, targetWordIndices = contextSuperBatch[:,1], contextSuperBatch[:,1:-1], contextSuperBatch[:,-1]

            model.train(wordIndices, targetWordIndices, miniBatchSize, learningRate, l1Coefficient, l2Coefficient)

            metrics = validation.validate(wordVocabulary, model)
            # customMetrics = {
            #     'simAB': similarity('A', 'B', wordVocabulary, model),
            #     'simBC': similarity('B', 'C', wordVocabulary, model)
            # }
            #validation.dump(metricsPath, epoch, superBatchIndex, *metrics, **customMetrics)
            validation.dump(metricsPath, epoch, superBatchIndex, *metrics)

            if previousTotal < sum(metrics):
                model.dump(parametersPath, embeddingsPath)

            currentTime = time.time()
            elapsed = currentTime - startTime
            secondsPerEpoch = elapsed / (epoch + 1)

            rg, sim353, simLex999, syntRel, sat = metrics
            # log.progress('Training model: {0:.3f}%. Elapsed: {1}. Epoch: {2}. ({3:.3f} sec/epoch), RG: {4}. Sim353: {5}. SimLex999: {6}. SyntRel: {7}. SAT: {8}. A/B: {9:.3f}. B/C: {10:.3f}',
            #              epoch + 1, epochs, log.delta(elapsed), epoch, secondsPerEpoch,
            #              rg, sim353, simLex999, syntRel, sat,
            #              customMetrics['simAB'],
            #              customMetrics['simBC'])
            log.progress('Training model: {0:.3f}%. Elapsed: {1}. Epoch: {2}. ({3:.3f} sec/epoch), RG: {4:.3f}. Sim353: {5:.3f}. SimLex999: {6:.3f}.',
                         epoch + 1, epochs, log.delta(elapsed), epoch, secondsPerEpoch,
                         rg, sim353, simLex999)

    log.lineBreak()

    return model


def similarity(left, right, wordVocabulary, embeddings):
    leftIndex, leftFrequency = wordVocabulary[left]
    rightIndex, rightFrequency = wordVocabulary[right]

    leftEmbedding = embeddings[leftIndex]
    rightEmbedding = embeddings[rightIndex]

    return vectors.cosineSimilarity(leftEmbedding, rightEmbedding)


if __name__ == '__main__':
    fileVocabularyPath = '../data/Wikipedia/Processed/file_vocabulary.bin.gz'
    wordVocabularyPath = '../data/Wikipedia/Processed/word_vocabulary.bin.gz'
    contextsPath = '../data/Wikipedia/Processed/contexts.bin.gz'

    fileVocabulary = parameters.loadFileVocabulary(fileVocabularyPath)
    wordVocabulary = parameters.loadWordVocabulary(wordVocabularyPath, False)
    fileVocabularySize = parameters.getFileVocabularySize(fileVocabularyPath)
    wordVocabularySize = parameters.getWordVocabularySize(wordVocabularyPath)
    contextProvider = parameters.IndexContextProvider(contextsPath)
    contextSize = contextProvider.contextSize

    model = ProbabilisticLanguageModel(fileVocabularySize, wordVocabularySize, contextSize - 2, 10)

    model = trainModel(
        fileVocabulary,
        wordVocabulary,
        contextProvider,
        model,
        superBatchSize = 1000000,
        miniBatchSize = 1000,
        parametersPath = '../data/Wikipedia/Processed/parameters.bin',
        embeddingsPath = '../data/Wikipedia/Processed/embeddings.bin',
        learningRate = 0.13,
        l1Coefficient = 0.006,
        l2Coefficient = 0.001,
        epochs = 100,
        metricsPath = '../data/Wikipedia/Processed/metrics_l1_l2.csv')