import os
import glob
import log
import time
import gzip
import re
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


def processData(inputDirectoryPath, fileVocabularyPath, wordVocabularyPath, contextsPath, contextSize):
    for filePath in [fileVocabularyPath, wordVocabularyPath, contextsPath]:
        if os.path.exists(filePath):
            os.remove(filePath)

    fileVocabulary = {}
    wordVocabulary = {}

    pathName = inputDirectoryPath + '/*/*.txt.gz'
    textFilePaths = glob.glob(pathName)
    textFileCount = len(textFilePaths)
    startTime = time.time()

    for textFileIndex, textFilePath in enumerate(textFilePaths):
        contextProvider = ContextProvider(textFilePath)
        for wordContext in contextProvider.next(contextSize):
            for word in wordContext:
                if word not in wordVocabulary:
                    wordVocabulary[word] = len(wordVocabulary)

            indexContext = map(lambda w: wordVocabulary[w], wordContext)

        textFileName = os.path.basename(textFilePath)
        currentTime = time.time()
        elapsed = currentTime - startTime
        secondsPerFile = elapsed / (textFileIndex + 1)

        log.progress('Reading contexts: {0:.3f}%. Elapsed: {1}. ({2:.3f} sec/file). Last file: {3}.',
                     textFileIndex + 1,
                     textFileCount,
                     timedelta(seconds=elapsed),
                     secondsPerFile,
                     textFileName)


if __name__ == '__main__':
    inputDirectoryPath = '../data/Wikipedia_prepared'
    fileVocabularyPath = '../data/Wikipedia_processed/file_vocabulary.bin'
    wordVocabularyPath = '../data/Wikipedia_processed/word_vocabulary.bin'
    contextsPath = '../data/Wikipedia_processed/contexts.bin'
    contextSize = 5

    processData(inputDirectoryPath, fileVocabularyPath, wordVocabularyPath, contextsPath, contextSize)