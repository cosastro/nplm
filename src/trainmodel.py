from numpy.random import randn as random
import numpy
import time
import os

import theano

import theano.tensor as T

import parameters
import vectors
import log
import metrics


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

        cost = -T.mean(T.log(probabilities)[T.arange(targetProbability.shape[0]), targetProbability])

        learningRate = T.scalar('learningRate', dtype=floatX)

        gradients = [T.grad(cost, wrt=p) for p in parameters]
        updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradients)]

        miniBatchIndex = T.lscalar('miniBatchIndex')
        miniBatchSize = T.iscalar('miniBatchSize')

        self.input = theano.shared(empty(1,1), borrow=True)
        self.targetOutput = theano.shared(empty(1), borrow=True)

        self.trainModel = theano.function(
            inputs=[miniBatchIndex, miniBatchSize, learningRate],
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
            embeddingIndices = [i for i in xrange(item.start, item.stop, item.step)]

            return self.getWordEmbeddings(embeddingIndices)

        return self.getWordEmbeddings([item])[0]


    def train(self, trainingInput, trainingTargetOutput, miniBatchSize, learningRate):
        asarray = lambda x: numpy.asarray(x, dtype='int32')

        trainingInput = asarray(trainingInput)
        trainingTargetOutput = asarray(trainingTargetOutput)

        self.input.set_value(trainingInput)
        self.targetOutput.set_value(trainingTargetOutput)

        trainInputSize = trainingInput.shape[0]
        trainingBatchesCount = trainInputSize / miniBatchSize + int(trainInputSize % miniBatchSize > 0)

        for trainingBatchIndex in xrange(0, trainingBatchesCount):
            self.trainModel(trainingBatchIndex, miniBatchSize, learningRate)


    def dump(self, parametersPath, embeddingsPath):
        pass


def trainModel(fileVocabulary, wordVocabulary, contextProvider, model, superBatchSize, miniBatchSize, parametersPath, embeddingsPath, learningRate, epochs, metricsPath):
    if os.path.exists(metricsPath):
        os.remove(metricsPath)

    superBatchesCount = contextProvider.contextsCount / superBatchSize + 1
    startTime = time.time()
    previousTotal = 0

    for epoch in xrange(0, epochs):
        for superBatchIndex in xrange(0, superBatchesCount):
            contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

            fileIndices, wordIndices, targetWordIndices = contextSuperBatch[:,1], contextSuperBatch[:,1:-1], contextSuperBatch[:,-1]

            model.train(wordIndices, targetWordIndices, miniBatchSize, learningRate)

            rg, sim353, simLex999, syntRel, sat, total = metrics.validate(wordVocabulary, model)
            metrics.dump(epoch, superBatchIndex, rg, sim353, simLex999, syntRel, sat, total, metricsPath)

            if previousTotal < total:
                model.dump(parametersPath, embeddingsPath)

            currentTime = time.time()
            elapsed = currentTime - startTime

            log.progress('Training model: {0:.3f}%. Elapsed: {1}. Epoch: {2}. RG: {3}. Sim353: {4}. SimLex999: {5}. SyntRel: {6}. SAT: {7}. A/B: {8:.3f}. B/C: {9:.3f}',
                         epoch + 1, epochs, log.delta(elapsed), epoch,
                         rg, sim353, simLex999, syntRel, sat,
                         similarity('A', 'B', wordVocabulary, model),
                         similarity('B', 'C', wordVocabulary, model))

    log.lineBreak()

    return model


def similarity(left, right, wordVocabulary, embeddings):
    leftIndex, leftFrequency = wordVocabulary[left]
    rightIndex, rightFrequency = wordVocabulary[right]

    leftEmbedding = embeddings[leftIndex]
    rightEmbedding = embeddings[rightIndex]

    return vectors.cosineSimilarity(leftEmbedding, rightEmbedding)


if __name__ == '__main__':
    fileVocabularyPath = '../data/Fake/Processed/file_vocabulary.bin.gz'
    wordVocabularyPath = '../data/Fake/Processed/word_vocabulary.bin.gz'
    contextsPath = '../data/Fake/Processed/contexts.bin.gz'

    fileVocabulary = parameters.loadFileVocabulary(fileVocabularyPath)
    wordVocabulary = parameters.loadWordVocabulary(wordVocabularyPath)
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
        superBatchSize = 30,
        miniBatchSize = 10,
        parametersPath = '../data/Fake/Processed/parameters.bin',
        embeddingsPath = '../data/Fake/Processed/embeddings.bin',
        learningRate = 0.13,
        epochs = 1000,
        metricsPath = '../data/Fake/Processed/metrics.csv')