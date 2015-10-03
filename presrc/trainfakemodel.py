from makeembeddings import *
from processgooglenews import *
from numpy.random import randn as random
import theano
import theano.tensor as T


class ProbabilisticLanguageModel():
    def __init__(self, fileVocabularySize, wordVocabularySize, contextSize, embeddingSize):
        floatX = theano.config.floatX

        defaultWordEmbeddings = random(wordVocabularySize, embeddingSize).astype(dtype=floatX)
        self.embeddings = theano.shared(defaultWordEmbeddings, name='wordEmbeddings', borrow=True)

        defaultWeight = random(contextSize * embeddingSize, wordVocabularySize).astype(dtype=floatX)
        weight = theano.shared(defaultWeight, name='weight', borrow=True)

        # defaultBias = random(wordVocabularySize).astype(dtype=floatX)
        # bias = theano.shared(defaultBias, name='bias', borrow=True)

        parameters = [self.embeddings, weight]

        contextIndices = T.imatrix('contextIndices')

        context = self.embeddings[contextIndices]
        context = context.reshape((contextIndices.shape[0], contextSize * embeddingSize))

        probabilities = T.nnet.softmax(T.dot(context, weight))
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


def train_old(vocabulary, trainingInput, trainingTargetOutput, defaultWordEmbeddings, learningRate=0.13, epochs=100):
    floatX = theano.config.floatX

    contextSize = trainingInput.shape[1]
    embeddingSize = defaultWordEmbeddings.shape[1]
    outputsCount = len(vocabulary)

    contextIndexes = T.imatrix('contextIndexes')
    wordEmbeddings = theano.shared(defaultWordEmbeddings, name='wordEmbeddings')

    context = wordEmbeddings[contextIndexes]
    context = context.reshape((contextIndexes.shape[0], contextSize * embeddingSize))

    defaultWeight = np.random.randn(contextSize*embeddingSize, outputsCount)
    defaultBias = np.random.randn(outputsCount)

    weight = theano.shared(defaultWeight, name='weight', borrow=True)
    bias = theano.shared(defaultBias, name='bias', borrow=True)

    output = T.nnet.softmax(T.dot(context, weight))
    targetOutput = T.ivector('targetOutput')

    cost = -T.mean(T.log(output)[T.arange(targetOutput.shape[0]), targetOutput])

    parameters = [wordEmbeddings, weight]
    gradients = [T.grad(cost, wrt=p) for p in parameters]
    updates = [(p, p - learningRate * g) for p, g in zip(parameters, gradients)]

    trainModel = theano.function(
        inputs=[contextIndexes, targetOutput],
        outputs=cost,
        updates=updates
    )

    for epoch in xrange(epochs):
        trainModel(trainingInput, trainingTargetOutput)
        log.progress(epoch + 1, epochs)

    log.newline()

    trainedWordEmbeddings = wordEmbeddings.get_value()

    return trainedWordEmbeddings


def train(vocabulary, trainingInput, trainingTargetOutput, defaultWordEmbeddings, learningRate=0.13, epochs=100):
    fileVocabularySize = 0
    wordVocabularySize = len(vocabulary)
    contextSize = 4
    embeddingSize = 10

    model = ProbabilisticLanguageModel(fileVocabularySize, wordVocabularySize, contextSize, embeddingSize)

    trainedEmbeddings = model.train(trainingInput, trainingTargetOutput, learningRate, epochs)

    return trainedEmbeddings


def sim(left, right, vocabulary, wordEmbeddings):
    leftVector = wordEmbeddings[vocabulary[left]]
    rightVector = wordEmbeddings[vocabulary[right]]

    return cosineSimilarity(leftVector, rightVector)


if __name__ == '__main__':
    pagesDirectoryPath = '../data/Fake'
    pageFilePath = pagesDirectoryPath + '/full.txt'

    A = lambda x, dtype=None: np.asarray(x, dtype=dtype)

    vocabulary, windows = processPage(pageFilePath, bufferSize=100, windowSize=5)
    windows = A(windows)
    input, targetOutput = A(windows[:,:-1], 'int32'), A(windows[:,-1], 'int32')

    fileVocabularySize = 0
    wordVocabularySize = len(vocabulary)
    contextSize = 4
    embeddingSize = 500

    model = ProbabilisticLanguageModel(fileVocabularySize, wordVocabularySize, contextSize, embeddingSize)

    wordEmbeddings = model.embeddings.get_value()

    pairs = [('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd'), ('E', 'e'), ('F', 'f')]

    print '-->Before training'
    for left, right in pairs:
        print '{0} & {1}: {2}'.format(left, right, sim(left, right, vocabulary, wordEmbeddings))

    trainedEmbeddings = model.train(input, targetOutput, learningRate=0.01, epochs=1000)

    print '-->After training'
    for left, right in pairs:
        print '{0} & {1}: {2}'.format(left, right, sim(left, right, vocabulary, trainedEmbeddings))