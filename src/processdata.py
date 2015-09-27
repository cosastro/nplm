import os
import glob
import log
import time
import gzip
import re
import struct
import io
from datetime import timedelta


class ContextProvider:
    def __init__(self, textFilePath):
        if textFilePath.endswith('gz'):
            self.textFile = gzip.open(textFilePath)
        else:
            self.textFile = open(textFilePath, 'rb')


    def __del__(self):
        self.textFile.close()


    def next(self, contextSize, bufferSize=100):
        buffer = self.textFile.read(bufferSize)
        tail = ''

        while buffer != '':
            buffer = tail + buffer
            buffer = re.split('\.', buffer)

            tail = buffer[-1]

            for sentence in buffer[:-1]:
                words = re.split('\s+', sentence.strip())

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield window

            words = re.split('\s+', tail.lstrip())

            buffer = self.textFile.read(bufferSize)

            if len(words) > contextSize * 2 - 1 or buffer == '':
                if buffer != '':
                    tail = ' '.join(words[-contextSize:])
                    words = words[:-contextSize]

                for wordIndex in range(len(words) - contextSize + 1):
                    window = words[wordIndex: wordIndex + contextSize]

                    yield window


def dumpFileVocabulary(vocabulary, vocabularyFilePath):
    dumpVocabulary(vocabulary, vocabularyFilePath, 'Dumping file vocabulary: {0:.3f}%.')


def dumpWordVocabulary(vocabulary, vocabularyFilePath):
    dumpVocabulary(vocabulary, vocabularyFilePath, 'Dumping word vocabulary: {0:.3f}%.')


def dumpVocabulary(vocabulary, vocabularyFilePath, messageFormat):
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
            log.progress(messageFormat, itemIndex, itemsCount)

        file.flush()

        log.lineBreak()


def loadVocabulary(vocabularyFilePath):
    vocabulary = {}

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

            log.progress('Locading vocabulary: {0:.3f}%.', itemIndex + 1, itemsCount)

        log.info('')

    return vocabulary


def loadContexts(contextsFilePath):
    contexts = []

    with gzip.open(contextsFilePath, 'rb') as file:
        contextsCount = file.read(4)
        contextsCount = struct.unpack('i', contextsCount)[0]

        includeFileIndex = file.read(1)
        includeFileIndex = struct.unpack('?', includeFileIndex)[0]

        contextLength = file.read(4)
        contextLength = struct.unpack('i', contextLength)[0]

        if includeFileIndex:
            contextLength += 1

        format = '{0}i'.format(contextLength)

        for contextIndex in range(0, contextsCount):
            context = file.read(contextLength * 4)
            context = struct.unpack(format, context)

            contexts.append(context)

            contextIndex += 1
            log.progress('Locading contexts: {0:.3f}%.', contextIndex + 1, contextsCount)

        log.lineBreak()

    return contexts


def processData(inputDirectoryPath, fileVocabularyPath, wordVocabularyPath, contextsPath, includeFileIndex, contextSize):
    if os.path.exists(contextsPath):
        os.remove(contextsPath)

    fileVocabulary = {}
    wordVocabulary = {}

    tempContextsPath = contextsPath + '.tmp'

    if os.path.exists(tempContextsPath):
        os.remove(tempContextsPath)

    with open(tempContextsPath, 'wb+') as tempContextsFile:
        tempContextsFile.write(struct.pack('i', 0)) #this is a placeholder for contexts count
        tempContextsFile.write(struct.pack('?', includeFileIndex))
        tempContextsFile.write(struct.pack('i', contextSize))

        pathName = inputDirectoryPath + '/*/*.txt.gz'
        textFilePaths = glob.glob(pathName)
        textFileCount = len(textFilePaths)
        startTime = time.time()

        contextFormat = '{0}i'.format(contextSize + 1 if includeFileIndex else contextSize)
        contextsCount = 0

        for textFileIndex, textFilePath in enumerate(textFilePaths):
            fileVocabulary[textFilePath] = textFileIndex

            contextProvider = ContextProvider(textFilePath)
            for wordContext in contextProvider.next(contextSize):
                for word in wordContext:
                    if word not in wordVocabulary:
                        wordVocabulary[word] = len(wordVocabulary)

                indexContext = map(lambda w: wordVocabulary[w], wordContext)
                if includeFileIndex:
                    indexContext = [textFileIndex] + indexContext

                tempContextsFile.write(struct.pack(contextFormat, *indexContext))
                contextsCount += 1

            textFileName = os.path.basename(textFilePath)
            currentTime = time.time()
            elapsed = currentTime - startTime
            secondsPerFile = elapsed / (textFileIndex + 1)

            log.progress('Reading contexts: {0:.3f}%. Elapsed: {1} ({2:.3f} sec/file). Vocabulary: {3}.',
                         textFileIndex + 1,
                         textFileCount,
                         timedelta(seconds=elapsed),
                         secondsPerFile,
                         len(wordVocabulary))

        log.lineBreak()

        tempContextsFile.seek(0, io.SEEK_SET)
        tempContextsFile.write(struct.pack('i', contextsCount))

    bufferSize = 1048576
    tempFileStats = os.stat(tempContextsPath)
    tempFileSize = tempFileStats.st_size
    with open(tempContextsPath, 'rb') as tempContextsFile:
        with gzip.open(contextsPath, 'wb+') as contextsFile:
            buffer = tempContextsFile.read(bufferSize)
            while buffer != '':
                contextsFile.write(buffer)
                buffer = tempContextsFile.read(bufferSize)

                position = tempContextsFile.tell()
                log.progress('Compressing contexts file: {0:.3f}%.', position, tempFileSize)

    log.lineBreak()

    os.remove(tempContextsPath)

    dumpFileVocabulary(fileVocabulary, fileVocabularyPath)
    dumpWordVocabulary(wordVocabulary, wordVocabularyPath)


if __name__ == '__main__':
    inputDirectoryPath = '../data/Wikipedia_prepared'
    fileVocabularyPath = '../data/Wikipedia_processed/file_vocabulary.bin.gz'
    wordVocabularyPath = '../data/Wikipedia_processed/word_vocabulary.bin.gz'
    contextsPath = '../data/Wikipedia_processed/contexts.bin.gz'
    contextSize = 5

    processData(inputDirectoryPath, fileVocabularyPath, wordVocabularyPath, contextsPath, True, contextSize)