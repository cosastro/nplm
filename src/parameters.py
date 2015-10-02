import os
import gzip
import io
import log
import struct
import collections


class IndexContextProvider():
    def __init__(self, contextsFilePath):
        self.contextsFilePath = contextsFilePath
        self.contextsFile = gzip.open(self.contextsFilePath, 'rb')
        self.contextsCount, self.contextSize = self.readContextsShape()
        self.bufferSize = self.contextSize * 4
        self.contextFormat = '{0}i'.format(self.contextSize)


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop if item.stop <= self.contextsCount else self.contextsCount
            step = item.step or 1
            return [self[i] for i in xrange(start, stop, step)]

        return self.getContext(item)


    def __del__(self):
        self.contextsFile.close()


    def readContextsShape(self):
        self.contextsFile.seek(0, io.SEEK_SET)

        contextsCount = self.contextsFile.read(4)
        contextSize = self.contextsFile.read(4)

        self.contextsCount = struct.unpack('i', contextsCount)[0]
        self.contextSize = struct.unpack('i', contextSize)[0]

        return self.contextsCount, self.contextSize


    def getContext(self, contextIndex):
        contextPosition = contextIndex * self.bufferSize + 8 # 8 for contextsCount + contextSize

        self.contextsFile.seek(contextPosition, io.SEEK_SET)

        buffer = self.contextsFile.read(self.bufferSize)
        context = struct.unpack(self.contextFormat, buffer)

        return context


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


def dumpEmbeddings(embeddings, embeddingsFilePath):
    with open(embeddingsFilePath, 'w') as embeddingsFile:
        embedding = embeddingsFile.readall()
        return embedding


def loadEmbeddings(embeddingsFilePath):
    with open(embeddingsFilePath) as embeddingsFile:
        embedding = embeddingsFile.readall()
        return embedding


def loadContexts(contextsFilePath):
    contexts = []

    with gzip.open(contextsFilePath, 'rb') as file:
        contextsCount = file.read(4)
        contextsCount = struct.unpack('i', contextsCount)[0]

        contextSize = file.read(4)
        contextSize = struct.unpack('i', contextSize)[0]

        contextSize += 1 # to include file index that preceeds context itself

        format = '{0}i'.format(contextSize)

        for contextIndex in range(0, contextsCount):
            context = file.read(contextSize * 4)
            context = struct.unpack(format, context)

            contexts.append(context)

            contextIndex += 1
            log.progress('Loading contexts: {0:.3f}%.', contextIndex, contextsCount)

        log.lineBreak()

    return contexts