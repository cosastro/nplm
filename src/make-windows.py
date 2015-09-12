import re
import glob
import collections
import gzip
import log
import os


class WordEmbedding():
    def __init__(self, word, index, frequency=1):
        self.word = word
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

    for pageFilePath in pageFilePaths[:10]:
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
                                vocabulary[word] = WordEmbedding(word, len(vocabulary))
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
                                vocabulary[word] = WordEmbedding(word, len(vocabulary))
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

    with gzip.open(vocabularyFilePath, 'w') as file:
        for key, value in vocabulary.items():
            line = '{0}:{1},{2},{3}\n'.format(key, value.word, value.index, value.frequency)
            file.write(line)

def dumpContexts(contexts, contextsFilePath):
    if os.path.exists(contextsFilePath):
        os.remove(contextsFilePath)

    with gzip.open(contextsFilePath, 'w') as file:
        for context in contexts:
            line = '{0}\n'.format(context)
            file.write(line)


if __name__ == '__main__':
    pagesDirectoryPath = '../data/Wikipedia-pages'
    vocabulary, contexts = processPages(pagesDirectoryPath, windowSize=100, contextSize=5)

    vocabularyFilePath = '../data/Wikipedia-data/vocabulary.txt.gz'
    dumpVocabulary(vocabulary, vocabularyFilePath)

    contextsFilePath = '../data/Wikipedia-data/context.txt.gz'
    dumpContexts(contexts, contextsFilePath)

    print 'Vocabulary size: {0}'.format(len(vocabulary))
    print 'Contexts found: {0}'.format(len(contexts))