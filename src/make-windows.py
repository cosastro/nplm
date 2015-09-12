import re
import glob
import collections
import gzip
import log


def processPages():
    pagesDirectoryPath = '../data/Wikipedia-pages'

    wikipediaFilesMask = pagesDirectoryPath + '/*/*.gz'
    pageFilePaths = glob.glob(wikipediaFilesMask)

    windowSize = 4
    contextSize = 3
    contexts = []
    vocabulary = {}
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

                        contexts.append(window)

                        if window[0] not in vocabulary:
                            vocabulary[window[0]] = 0

                        vocabulary[window[0]] += 1

                words = re.split('\s+', tail.lstrip())

                buffer = file.read(windowSize)

                if len(words) > windowSize * 2 - 1 or buffer == '':
                    if buffer != '':
                        tail = ' '.join(words[-windowSize:])
                        words = words[:-windowSize]

                    for wordIndex in range(len(words) - contextSize + 1):
                        window = words[wordIndex: wordIndex + contextSize]

                        contexts.append(window)

                        if window[0] not in vocabulary:
                            vocabulary[window[0]] = 0

                        vocabulary[window[0]] += 1

            message = 'Words: {0}. Contexts: {1}.'.format(len(vocabulary), len(contexts))
            log.progress(fileIndex, filesCount, message)

            fileIndex += 1
        finally:
            file.close()

    vocabulary = collections.OrderedDict(sorted(vocabulary.items(), key=lambda x: x[1]))
    log.info('')

    return vocabulary, contexts


if __name__ == '__main__':
    vocabulary, contexts = processPages()

    print 'Vocabulary size: {0}'.format(len(vocabulary))
    print 'Contexts found: {0}'.format(len(contexts))