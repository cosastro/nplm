import parameters
import collections
import vectors
import log


def trainModel(fileVocabularyPath, wordVocabularyPath, contextsPath, superBatchSize, miniBatchSize, embeddingsPath):
    fileVocabularySize = parameters.getFileVocabularySize(fileVocabularyPath)
    wordVocabularySize = parameters.getWordVocabularySize(wordVocabularyPath)

    contextProvider = parameters.IndexContextProvider(contextsPath)

    maxiBatchesCount = contextProvider.contextsCount / superBatchSize + 1
    for superBatchIndex in xrange(0, maxiBatchesCount):
        contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

        miniBatchesCount = len(contextSuperBatch) / miniBatchSize + 1
        for miniBatchIndex in xrange(0, miniBatchesCount):
            contextMiniBatch = contextSuperBatch[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize]

            for context in contextMiniBatch:
                print context

    return None, None, None


def similarity(left, right, wordVocabulary, embeddings):
    leftEmbedding = embeddings[wordVocabulary[left]]
    rightEmbedding = embeddings[wordVocabulary[right]]

    return vectors.cosineSimilarity(leftEmbedding, rightEmbedding)


if __name__ == '__main__':
    fileVocabularyPath = '../data/Fake/Processed/file_vocabulary.bin.gz'
    wordVocabularyPath = '../data/Fake/Processed/word_vocabulary.bin.gz'
    contextsPath = '../data/Fake/Processed/contexts.bin.gz'
    superBatchSize = 30
    miniBatchSize = 10
    embeddingsPath = '../data/Fake/Processed/embeddings.bin'

    wight, bias, embeddings = trainModel(
        fileVocabularyPath,
        wordVocabularyPath,
        contextsPath,
        superBatchSize,
        miniBatchSize,
        embeddingsPath)

    wordVocabulary = parameters.loadWordVocabulary(wordVocabularyPath)

    log.info('A & B: {0}', similarity('A', 'B', wordVocabulary, embeddings))
    log.info('A & a: {0}', similarity('A', 'a', wordVocabulary, embeddings))