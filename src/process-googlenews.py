import struct
import log
import math

# filePath = '../data/GoogleNews-vectors/GoogleNews-vectors-negative300.bin'

filePath = '../data/Text8-vectors/vectors.bin'

def cosineSimilarity(vectorA, vectorB):
    numerator = sum([a * b for a, b in zip(vectorA, vectorB)])
    denumerator = math.sqrt(sum([a * a for a in vectorA]) * sum([b * b for b in vectorB]))

    denumerator = 0.000000000001 if denumerator == 0 else denumerator

    return numerator / denumerator

def loadWordVectors(filePath):
    with open(filePath, 'rb') as file:
        firstLine = file.readline()
        vocabularySize, vectorSize = tuple(firstLine.split(' '))
        vocabularySize, vectorSize = int(vocabularySize), int(vectorSize)
        vocabulary = {}

        message = 'Vocabulary size: {0}; vector size: {1}'.format(vocabularySize, vectorSize)
        log.info(message)

        wordCounter = 0.

        while True:
            word = ''
            while True:
                char = file.read(1)

                if not char:
                    return vocabulary

                if char == ' ':
                    word = word.strip()
                    break

                word += char

            wordFeatureVector = struct.unpack('{0}f'.format(vectorSize), file.read(4 * vectorSize))
            vocabulary[word] = wordFeatureVector

            wordCounter += 1
            log.progress(wordCounter, vocabularySize)

wordVectors = loadWordVectors(filePath)
word = 'god'

distances = [[alt, cosineSimilarity(wordVectors[word], wordVectors[alt])] for alt in wordVectors.keys()]

comparator = lambda a, b: cmp(a[1], b[1])
distances.sort(cmp=comparator, reverse=True)
closest = [d[0] for d in distances[:10]]

print '{0}'.format(word)

for d in distances[:10]:
    print '{0}\t\t\t\t{1}'.format(d[0], d[1])