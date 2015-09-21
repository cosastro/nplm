from makeembeddings import *


if __name__ == '__main__':
    pagesDirectoryPath = '../data/Fake'

    vocabulary, windows = processDirectory(pagesDirectoryPath, bufferSize=100, windowSize=5)
    wordEmbeddings = makeEmbeddings(vocabulary, embeddingSize=5)

    log.info(vocabulary)
    log.newline()

    for window in windows:
        log.info(window)

    log.newline()
    log.info(wordEmbeddings)

