import re
import glob
import collections
import gzip
import log


class WordEmbedding():
    def __init__(self, word, index, frequency=1):
        self.word = word
        self.index = index
        self.frequency = frequency


def processPages():
    pagesDirectoryPath = '../data/Wikipedia-pages'

    wikipediaFilesMask = pagesDirectoryPath + '/*/*.gz'
    pageFilePaths = glob.glob(wikipediaFilesMask)

    windowSize = 100
    contextSize = 5
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


def dumpVocabulary(vocabulary):
    pass

def dumpContexts(contexts):
    pass


if __name__ == '__main__':
    vocabulary, contexts = processPages()

    dumpVocabulary(vocabulary)

    dumpContexts(contexts)

    print 'Vocabulary size: {0}'.format(len(vocabulary))
    print 'Contexts found: {0}'.format(len(contexts))