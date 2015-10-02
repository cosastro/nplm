import parameters
import collections


def trainModel(fileVocabularyPath, wordVocabularyPath, contextsPath, superBatchSize, miniBatchSize):
    fileVocabularySize = parameters.getFileVocabularySize(fileVocabularyPath)
    print 'File vocabulary size: {0}.'.format(fileVocabularySize)

    fileVocabulary = parameters.loadFileVocabulary(fileVocabularyPath)
    fileIndexVocabulary = [(value, key) for key, value in fileVocabulary.items()]
    fileIndexVocabulary = collections.OrderedDict(fileIndexVocabulary)

    fileVocabularySize = parameters.getWordVocabularySize(wordVocabularyPath)
    print 'Word vocabulary size: {0}.'.format(fileVocabularySize)

    wordVocabulary = parameters.loadWordVocabulary(wordVocabularyPath)
    wordIndexVocabulary = [(value[0], key) for key, value in wordVocabulary.items()]
    wordIndexVocabulary = collections.OrderedDict(wordIndexVocabulary)

    contextProvider = parameters.IndexContextProvider(contextsPath)

    print 'Contexts count: {0}'.format(contextProvider.contextsCount)
    print 'Context size: {0}'.format(contextProvider.contextSize)

    maxiBatchesCount = contextProvider.contextsCount / superBatchSize + 1
    for superBatchIndex in xrange(0, maxiBatchesCount):
        contextSuperBatch = contextProvider[superBatchIndex * superBatchSize:(superBatchIndex + 1) * superBatchSize]

        miniBatchesCount = len(contextSuperBatch) / miniBatchSize + 1
        print '{0}'.format(len(contextSuperBatch))

        for miniBatchIndex in xrange(0, miniBatchesCount):
            contextMiniBatch = contextSuperBatch[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize]

            if len(contextMiniBatch) > 0:
                print '\t{0}'.format(len(contextMiniBatch))

            for context in contextMiniBatch:
                fileIndex = context[0]
                wordIndices = context[1:]

                file = fileIndexVocabulary[fileIndex]
                words = map(lambda wordIndex: wordIndexVocabulary[wordIndex], wordIndices)

                print '\t\t{0} {1}'.format(file, words)


if __name__ == '__main__':
    fileVocabularyPath = '../data/Fake/Processed/file_vocabulary.bin.gz'
    wordVocabularyPath = '../data/Fake/Processed/word_vocabulary.bin.gz'
    contextsPath = '../data/Fake/Processed/contexts.bin.gz'
    superBatchSize = 30
    miniBatchSize = 10

    trainModel(fileVocabularyPath, wordVocabularyPath, contextsPath, superBatchSize, miniBatchSize)