from numpy.random import randn as random

import theano
import theano.tensor as T

import parameters
import vectors
import log


class ProbabilisticLanguageModel():
    def __init__(self, fileVocabularySize, wordVocabularySize, contextSize, embeddingSize):
        floatX = theano.config.floatX

        defaultWordEmbeddings = random(wordVocabularySize, embeddingSize).astype(dtype=floatX)
        self.embeddings = theano.shared(defaultWordEmbeddings, name='wordEmbeddings', borrow=True)

        defaultWeight = random(contextSize * embeddingSize, wordVocabularySize).astype(dtype=floatX)
        self.weight = theano.shared(defaultWeight, name='weight', borrow=True)

        defaultBias = random(wordVocabularySize).astype(dtype=floatX)
        self.bias = theano.shared(defaultBias, name='bias', borrow=True)

        parameters = [self.embeddings, self.weight, self.bias]

        contextIndices = T.imatrix('contextIndices')

        context = self.embeddings[contextIndices]
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


    def train(self, trainingInput, trainingTargetOutput, learningRate, epochs):
        for epoch in xrange(epochs):
            self.trainModel(trainingInput, trainingTargetOutput, learningRate)

        trainedEmbeddings = self.embeddings.get_value()

        return trainedEmbeddings


def trainModel(model, contextProvider, superBatchSize, miniBatchSize, embeddingsPath):
    maxiBatchesCount = contextProvider.contextsCount / superBatchSize + 1

    for superBatchIndex in xrange(0, maxiBatchesCount):
        contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

        miniBatchesCount = len(contextSuperBatch) / miniBatchSize + 1
        for miniBatchIndex in xrange(0, miniBatchesCount):
            contextMiniBatch = contextSuperBatch[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize]

            if len(contextMiniBatch) == 0:
                continue

            fileIndices, wordIndices, targetWordIndices = contextMiniBatch[:,1], contextMiniBatch[:,1:-1], contextMiniBatch[:,-1]

            model.train(wordIndices, targetWordIndices, 0.13, 1000)

            log.progress('Training model: {0:.3f}%.',
                         miniBatchIndex + maxiBatchesCount * superBatchIndex,
                         maxiBatchesCount * miniBatchesCount)

    log.lineBreak()

    return model.embeddings.get_value()


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
    superBatchSize = 30
    miniBatchSize = 10
    embeddingsPath = '../data/Fake/Processed/embeddings.bin'

    fileVocabularySize = parameters.getFileVocabularySize(fileVocabularyPath)
    wordVocabularySize = parameters.getWordVocabularySize(wordVocabularyPath)
    contextProvider = parameters.IndexContextProvider(contextsPath)
    contextSize = contextProvider.contextSize
    embeddingSize = 10

    model = ProbabilisticLanguageModel(fileVocabularySize, wordVocabularySize, contextSize - 2, embeddingSize)
    embeddings = model.embeddings.get_value()

    wordVocabulary = parameters.loadWordVocabulary(wordVocabularyPath)

    log.info('A & B: {0}', similarity('A', 'B', wordVocabulary, embeddings))
    log.info('B & C: {0}', similarity('B', 'C', wordVocabulary, embeddings))
    log.info('C & D: {0}', similarity('C', 'D', wordVocabulary, embeddings))

    embeddings = trainModel(model, contextProvider, superBatchSize, miniBatchSize, embeddingsPath)

    log.info('A & B: {0}', similarity('A', 'B', wordVocabulary, embeddings))
    log.info('B & C: {0}', similarity('B', 'C', wordVocabulary, embeddings))
    log.info('C & D: {0}', similarity('C', 'D', wordVocabulary, embeddings))