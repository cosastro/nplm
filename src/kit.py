import log
import struct


def loadWord2VecEmbeddings(filePath):
    with open(filePath, 'rb') as file:
        firstLine = file.readline()
        embeddingsCount, embeddingSize = tuple(firstLine.split(' '))
        embeddingsCount, embeddingSize = int(embeddingsCount), int(embeddingSize)
        embeddingFormat = '{0}f'.format(embeddingSize)
        wordIndexMap = {}
        embeddings = []

        log.info('Vocabulary size: {0}. Embedding size: {1}.', embeddingsCount, embeddingSize)

        embeddingIndex = 0
        while True:
            word = ''
            while True:
                char = file.read(1)

                if not char:
                    log.lineBreak()
                    return wordIndexMap, embeddings

                if char == ' ':
                    word = word.strip()
                    break

                word += char

            embedding = struct.unpack(embeddingFormat, file.read(embeddingSize * 4))
            wordIndexMap[word] = len(wordIndexMap)
            embeddings.append(embedding)

            embeddingIndex += 1
            log.progress('Reading embeddings: {0:.3f}%.', embeddingIndex, embeddingsCount)