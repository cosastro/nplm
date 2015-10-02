import os
import gzip
import io
import log
import struct
import collections
import numpy


class IndexContextProvider():
    def __init__(self, contextsFilePath):
        self.contextsFilePath = contextsFilePath
        self.contextsFile = gzip.open(self.contextsFilePath, 'rb')
        self.contextsCount, self.contextSize = self.readContextsShape()


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop <= self.contextsCount else self.contextsCount
            step = item.step or 1

            return self.getContexts(start, stop, step)

        return self.getContexts(item, item + 1, 1)


    def __del__(self):
        self.contextsFile.close()


    def readContextsShape(self):
        self.contextsFile.seek(0, io.SEEK_SET)

        contextsCount = self.contextsFile.read(4)
        contextSize = self.contextsFile.read(4)

        self.contextsCount = struct.unpack('i', contextsCount)[0]
        self.contextSize = struct.unpack('i', contextSize)[0]

        return self.contextsCount, self.contextSize


    def getContexts(self, start, stop, step):
        if step == 1:
            count = stop - start
            contextBufferSize = self.contextSize * 4
            contextsBufferSize = count * contextBufferSize
            startPosition = start * contextBufferSize + 8 # 8 for contextsCount + contextSize

            self.contextsFile.seek(startPosition, io.SEEK_SET)
            contextsBuffer = self.contextsFile.read(contextsBufferSize)

            contextFormat = '{0}i'.format(self.contextSize * count)
            contexts = struct.unpack(contextFormat, contextsBuffer)

            contexts = numpy.reshape(contexts, (count, self.contextSize))
        else:
            contexts = []
            for contextIndex in xrange(start, stop, step):
                context = self[contextIndex][0]
                contexts.append(context)

        return contexts


class EmbeddingsProvider():
    def __init__(self, embeddingsFilePath):
        self.embeddingsFilePath = embeddingsFilePath

    def getEmbedding(self):
        with open(self.embeddingsFilePath, 'r') as embeddingsFile:
            embedding = embeddingsFile.readall()
            return embedding


def dumpFileVocabulary(vocabulary, vocabularyFilePath):
    if os.path.exists(vocabularyFilePath):
        os.remove(vocabularyFilePath)

    itemsCount = len(vocabulary)
    itemIndex = 0

    with gzip.open(vocabularyFilePath, 'w') as file:
        file.write(struct.pack('i', itemsCount))

        for key, index in vocabulary.items():
            keyLength = len(key)
            keyLength = struct.pack('i', keyLength)
            index = struct.pack('i', index)

            file.write(keyLength)
            file.write(key)
            file.write(index)

            itemIndex += 1
            log.progress('Dumping file vocabulary: {0:.3f}%.', itemIndex, itemsCount)

        file.flush()

        log.lineBreak()


def loadFileVocabulary(vocabularyFilePath):
    vocabulary = collections.OrderedDict()

    with gzip.open(vocabularyFilePath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        for itemIndex in range(0, itemsCount):
            wordLength = file.read(4)
            wordLength = struct.unpack('i', wordLength)[0]

            word = file.read(wordLength)

            index = file.read(4)
            index = struct.unpack('i', index)[0]

            vocabulary[word] = index

            log.progress('Loading file vocabulary: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.info('')

    return vocabulary


def getFileVocabularySize(fileVocabularyPath):
    with gzip.open(fileVocabularyPath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        return itemsCount


def dumpWordVocabulary(vocabulary, vocabularyFilePath):
    if os.path.exists(vocabularyFilePath):
        os.remove(vocabularyFilePath)

    itemsCount = len(vocabulary)
    itemIndex = 0

    with gzip.open(vocabularyFilePath, 'w') as file:
        file.write(struct.pack('i', itemsCount))

        for key, value in vocabulary.items():
            keyLength = len(key)
            keyLength = struct.pack('i', keyLength)
            index, frequency = value
            index = struct.pack('i', index)
            frequency = struct.pack('i', frequency)

            file.write(keyLength)
            file.write(key)
            file.write(index)
            file.write(frequency)

            itemIndex += 1
            log.progress('Dumping word vocabulary: {0:.3f}%.', itemIndex, itemsCount)

        file.flush()

        log.lineBreak()


def loadWordVocabulary(vocabularyFilePath):
    vocabulary = collections.OrderedDict()

    with gzip.open(vocabularyFilePath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        for itemIndex in range(0, itemsCount):
            wordLength = file.read(4)
            wordLength = struct.unpack('i', wordLength)[0]

            word = file.read(wordLength)

            index = file.read(4)
            index = struct.unpack('i', index)[0]

            frequency = file.read(4)
            frequency = struct.unpack('i', frequency)[0]

            vocabulary[word] = (index, frequency)

            log.progress('Loading word vocabulary: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.info('')

    return vocabulary


def getWordVocabularySize(wordVocabularyPath):
    with gzip.open(wordVocabularyPath, 'rb') as file:
        itemsCount = file.read(4)
        itemsCount = struct.unpack('i', itemsCount)[0]

        return itemsCount