import parameters
import collections


def trainModel():
    fileVocabularyPath = '../data/Fake/Processed/file_vocabulary.bin.gz'
    fileVocabulary = parameters.loadFileVocabulary(fileVocabularyPath)
    fileIndexVocabulary = [(value, key) for key, value in fileVocabulary.items()]
    fileIndexVocabulary = collections.OrderedDict(fileIndexVocabulary)

    wordVocabularyPath = '../data/Fake/Processed/word_vocabulary.bin.gz'
    wordVocabulary = parameters.loadWordVocabulary(wordVocabularyPath)
    wordIndexVocabulary = [(value[0], key) for key, value in wordVocabulary.items()]
    wordIndexVocabulary = collections.OrderedDict(wordIndexVocabulary)

    contextsPath = '../data/Fake/Processed/contexts.bin.gz'
    contextProvider = parameters.IndexContextProvider(contextsPath)

    print 'Contexts count: {0}'.format(contextProvider.contextsCount)
    print 'Context size: {0}'.format(contextProvider.contextSize)

    for context in contextProvider[:]:
        fileIndex = context[0]
        wordIndices = context[1:]

        file = fileIndexVocabulary[fileIndex]
        words = map(lambda wordIndex: wordIndexVocabulary[wordIndex], wordIndices)

        print '{0} {1}'.format(file, words)


if __name__ == '__main__':
    trainModel()