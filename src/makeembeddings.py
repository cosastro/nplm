import numpy as np
from makewindows import *


def makeEmbeddings(vocabulary, embeddingSize):
    wordsCount = len(vocabulary)
    embeddings = np.random.randn(wordsCount, embeddingSize)

    return embeddings


def dumpEmbeddings(embeddings, embeddingsFilePath):
    if os.path.exists(embeddingsFilePath):
        os.remove(embeddingsFilePath)

    message = 'Dumping word embeddings into {0}...'.format(embeddingsFilePath)
    log.info(message)

    wordsCount, embeddingSize = embeddings.shape

    with gzip.open(embeddingsFilePath, 'w') as file:
        file.write(struct.pack('<i', wordsCount))
        file.write(struct.pack('<i', embeddingSize))

        format = '{0}f'.format(embeddingSize)
        for wordIndex in range(0, wordsCount):
            wordEmbedding = embeddings[wordIndex]
            wordEmbedding = struct.pack(format, *wordEmbedding)
            file.write(wordEmbedding)

            log.progress(wordIndex + 1, wordsCount)

    log.info('')

def loadEmbeddigns(embeddingsFilePath):
    message = 'Reading word embeddings from {0}...'.format(embeddingsFilePath)
    log.info(message)

    with gzip.open(embeddingsFilePath, 'rb') as file:
        wordsCount = file.read(4)
        wordsCount = struct.unpack('<i', wordsCount)[0]

        embeddingSize = file.read(4)
        embeddingSize = struct.unpack('<i', embeddingSize)[0]

        embeddings = np.empty((wordsCount, embeddingSize))

        format = '{0}f'.format(embeddingSize)
        for wordIndex in range(0, wordsCount):
            wordEmbedding = file.read(embeddingSize * 4)
            wordEmbedding = struct.unpack(format, wordEmbedding)[0]

            log.progress(wordIndex + 1, wordsCount)

    log.info('')

    return embeddings


if __name__ == '__main__':
    vocabularyFilePath = '../data/Wikipedia-data/vocabulary.bin.gz'
    vocabulary = loadVocabulary(vocabularyFilePath)

    embeddingsFilePath = '../data/Wikipedia-data/embeddings.bin.gz'
    embeddings = makeEmbeddings(vocabulary, 300)

    dumpEmbeddings(embeddings, embeddingsFilePath)