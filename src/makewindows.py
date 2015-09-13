import re
import glob
import collections
import gzip
import log
import os
import struct


class WordEmbedding():
    def __init__(self, index, frequency=1):
        self.index = index
        self.frequency = frequency


def processPages(pagesDirectoryPath, windowSize, contextSize):
    wikipediaFilesMask = pagesDirectoryPath + '/*/*.gz'
    pageFilePaths = glob.glob(wikipediaFilesMask)

    contexts = []
    vocabulary = collections.OrderedDict()
    fileIndex = 1
    filesCount = len(pageFilePaths)

    message = 'Found {0} files to process.'.format(filesCount)
    log.info(message)

    for pageFilePath in pageFilePaths:
        if pageFilePath.endswith('gz'):
            file = gzip.open(pageFilePath)
        else:
            file = open(pageFilePath)

        try:
            buffer = file.read(windowSize)
            tail = ''

            while buffer != '':
                buffer = tail + buffer
                buffer = re.split('\.', buffer)

                tail = buffer[-1]

                for sentence in buffer[:-1]:
                    words = re.split('\s+', sentence.strip())

                    for wordIndex in range(len(words) - contextSize + 1):
                        window = words[wordIndex: wordIndex + contextSize]

                        for word in window:
                            if word not in vocabulary:
                                vocabulary[word] = WordEmbedding(len(vocabulary))
                            else:
                                vocabulary[word].frequency += 1

                        window = map(lambda w: vocabulary[w].index, window)
                        contexts.append(window)

                words = re.split('\s+', tail.lstrip())

                buffer = file.read(windowSize)

                if len(words) > contextSize * 2 - 1 or buffer == '':
                    if buffer != '':
                        tail = ' '.join(words[-contextSize:])
                        words = words[:-contextSize]

                    for wordIndex in range(len(words) - contextSize + 1):
                        window = words[wordIndex: wordIndex + contextSize]

                        for word in window:
                            if word not in vocabulary:
                                vocabulary[word] = WordEmbedding(len(vocabulary))
                            else:
                                vocabulary[word].frequency += 1

                        window = map(lambda w: vocabulary[w].index, window)
                        contexts.append(window)

            message = 'Words: {0}. Contexts: {1}.'.format(len(vocabulary), len(contexts))
            log.progress(fileIndex, filesCount, message)

            fileIndex += 1
        finally:
            file.close()

    log.info('')

    return vocabulary, contexts


def dumpVocabulary(vocabulary, vocabularyFilePath):
    if os.path.exists(vocabularyFilePath):
        os.remove(vocabularyFilePath)

    itemsCount = len(vocabulary)
    itemIndex = 1

    message = 'Dumping vocabulary to {0}'.format(vocabularyFilePath)
    log.info(message)

    with gzip.open(vocabularyFilePath, 'w') as file:
        file.write(struct.pack('<i', itemsCount))

        for word, value in vocabulary.items():
            wordLength = len(word)
            wordLength = struct.pack('<i', wordLength)
            index = struct.pack('<i', value.index)
            frequency = struct.pack('<i', value.frequency)

            file.write(wordLength)
            file.write(word)
            file.write(index)
            file.write(frequency)

            file.flush()

            log.progress(itemIndex, itemsCount)
            itemIndex += 1

        log.info('')


def loadVocabulary(vocabularyFilePath):
    vocabulary = collections.OrderedDict()

    message = 'Reading vocabulary from {0}'.format(vocabularyFilePath)
    log.info(message)

    with gzip.open(vocabularyFilePath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('<i', itemsCount)[0]

        for itemIndex in range(0, itemsCount):
            wordLength = file.read(4)
            wordLength = struct.unpack('<i', wordLength)[0]

            word = file.read(wordLength)

            index = file.read(4)
            index = struct.unpack('<i', index)[0]

            frequency = file.read(4)
            frequency = struct.unpack('<i', frequency)[0]

            vocabulary[word] = WordEmbedding(index, frequency)

            log.progress(itemIndex + 1, itemsCount)

        log.info('')

    return vocabulary


def dumpContexts(contexts, contextsFilePath):
    if os.path.exists(contextsFilePath):
        os.remove(contextsFilePath)

    contextsCount = len(contexts)
    contextIndex = 1

    message = 'Dumping contexts to {0}'.format(contextsFilePath)
    log.info(message)

    with gzip.open(contextsFilePath, 'w') as file:
        file.write(struct.pack('<i', contextsCount))
        file.write(struct.pack('<i', len(contexts[0])))

        format = '{0}i'.format(len(contexts[0]))

        for context in contexts:
            ctx = struct.pack(format, *context)

            file.write(ctx)

            log.progress(contextIndex, contextsCount)
            contextIndex += 1


def loadContexts(contextsFilePath):
    message = 'Reading contexts to {0}'.format(contextsFilePath)
    log.info(message)

    contexts = []

    with gzip.open(contextsFilePath, 'rb') as file:
        contextsCount = file.read(4)
        contextsCount = struct.unpack('<i', contextsCount)[0]

        contextLength = file.read(4)
        contextLength = struct.unpack('<i', contextLength)[0]

        format = '{0}i'.format(contextLength)

        for contextIndex in range(0, contextsCount):
            context = file.read(contextLength * 4)
            ctx = struct.unpack(format, context)[0]

            contexts.append(ctx)

            log.progress(contextIndex, contextsCount)
            contextIndex += 1

        log.info('')

    return contexts


if __name__ == '__main__':
    #pagesDirectoryPath = '../data/Wikipedia-pages'
    #vocabulary, contexts = processPages(pagesDirectoryPath, windowSize=100, contextSize=5)

    vocabularyFilePath = '../data/Wikipedia-data/vocabulary.bin.gz'
    #dumpVocabulary(vocabulary, vocabularyFilePath)

    contextsFilePath = '../data/Wikipedia-data/context.bin.gz'
    #dumpContexts(contexts, contextsFilePath)

    #print 'Vocabulary size: {0}'.format(len(vocabulary))
    #print 'Contexts found: {0}'.format(len(contexts))

    vocabulary = loadVocabulary(vocabularyFilePath)
    contexts = loadContexts(contextsFilePath)

    print 'Vocabulary length: {0}'.format(len(vocabulary))
    print 'Contexts length: {0}'.format(len(contexts))