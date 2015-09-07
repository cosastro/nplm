import struct

filePath = '../data/GoogleNews-vectors/GoogleNews-vectors-negative300.bin'

def loadWordVectors(filePath):
    with open(filePath, 'rb') as file:
        firstLine = file.readline()
        vocabularySize, vectorSize = tuple(firstLine.split(' '))
        vocabularySize, vectorSize = int(vocabularySize), int(vectorSize)
        vocabulary = {}

        print 'Vocabulary size: {0}; vector size: {1}'.format(vocabularySize, vectorSize)

        while True:
            word = ''
            while True:
                char = file.read(1)

                if not char:
                    return vocabulary

                if char == ' ':
                    break

                word += char

            wordFeatureVector = struct.unpack('{0}f'.format(vectorSize), file.read(4 * vectorSize))
            vocabulary[word] = wordFeatureVector

wordVectors = loadWordVectors(filePath)

print '{0} {1}'.format(wordVectors.items()[10])