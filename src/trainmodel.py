from numpy.random import randn as random

import time
import theano
import theano.tensor as T

import parameters
import vectors
import log
import metrics


class ProbabilisticLanguageModel():
    def __init__(self, fileVocabularySize, wordVocabularySize, contextSize, embeddingSize):
        floatX = theano.config.floatX

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

        learningRate = T.fscalar('learningRate')

        gradients = [T.grad(cost, wrt=p) for p in parameters]
        updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradients)]

        self.trainModel = theano.function(
            inputs=[contextIndices, targetProbability, learningRate],
            outputs=cost,
            updates=updates
        )


    def train(self, trainingInput, trainingTargetOutput, learningRate):
        self.trainModel(trainingInput, trainingTargetOutput, learningRate)


    def getWordEmbeddings(self):
        return self.wordEmbeddings.get_value()


    def dump(self, parametersPath, embeddingsPath):
        pass


def trainModel(fileVocabulary, wordVocabulary, contextProvider, model, superBatchSize, miniBatchSize, parametersPath, embeddingsPath, learningRate, epochs, metricsPath):
    superBatchesCount = contextProvider.contextsCount / superBatchSize + 1

    startTime = time.time()
    miniBatchesTotal = 0

    for superBatchIndex in xrange(0, superBatchesCount):
        contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

        miniBatchStartTime = time.time()
        miniBatchesCount = len(contextSuperBatch) / miniBatchSize + 1

        for miniBatchIndex in xrange(0, miniBatchesCount):
            contextMiniBatch = contextSuperBatch[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize]

            if len(contextMiniBatch) == 0:
                continue

            fileIndices, wordIndices, targetWordIndices = contextMiniBatch[:,1], contextMiniBatch[:,1:-1], contextMiniBatch[:,-1]

            miniBatchesTotal += 1
            previousTotal = 0

            for epoch in xrange(0, epochs):
                model.train(wordIndices, targetWordIndices, learningRate)
                wordEmbeddigs = model.getWordEmbeddings()

                rg, sim353, simLex999, syntRel, sat, total = metrics.validate(wordVocabulary, wordEmbeddigs)

                metrics.dump(superBatchIndex, miniBatchIndex, epoch, rg, sim353, simLex999, syntRel, sat, total, metricsPath)

                if previousTotal < total:
                    model.dump(parametersPath, embeddingsPath)

                currentTime = time.time()
                elapsed = currentTime - startTime
                secondsPerSuperBatch = elapsed / (superBatchIndex + 1)
                secondsPerMiniBatch = elapsed / miniBatchesTotal

                log.progress('Training model: {0:.3f}%. Elapsed: {1}. ({2:.3f} sec/super batch). ({3:.3f} sec/mini batch).',
                             miniBatchIndex + float(epoch)/epochs + superBatchesCount * superBatchIndex,
                             superBatchesCount * miniBatchesCount,
                             log.delta(elapsed),
                             secondsPerSuperBatch,
                             secondsPerMiniBatch)

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
    embeddingSize = 10
    learningRate = 0.13
    epochs = 1000

    model = ProbabilisticLanguageModel(fileVocabularySize, wordVocabularySize, contextSize - 2, embeddingSize)

    wordEmbeddings = model.getWordEmbeddings()
    log.info('A & B: {0}', similarity('A', 'B', wordVocabulary, wordEmbeddings))
    log.info('B & C: {0}', similarity('B', 'C', wordVocabulary, wordEmbeddings))
    log.info('C & D: {0}', similarity('C', 'D', wordVocabulary, wordEmbeddings))

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

    wordEmbeddings = model.getWordEmbeddings()
    log.info('A & B: {0}', similarity('A', 'B', wordVocabulary, wordEmbeddings))
    log.info('B & C: {0}', similarity('B', 'C', wordVocabulary, wordEmbeddings))
    log.info('C & D: {0}', similarity('C', 'D', wordVocabulary, wordEmbeddings))