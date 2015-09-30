import log
import adapters
import math
import scipy.stats
import numpy
import pandas
import re


def cosineSimilarity(vectorA, vectorB):
    numerator = sum([a * b for a, b in zip(vectorA, vectorB)])
    denumerator = math.sqrt(sum([a * a for a in vectorA]) * sum([b * b for b in vectorB]))

    denumerator = 0.000000000001 if denumerator == 0 else denumerator

    return numerator / denumerator


def evaluateRubensteinGoodenough(wordIndexMap, embeddings, filePath):
    with open(filePath) as file:
        lines = file.readlines()

    wordPairs = []
    targetScores = []
    for line in lines:
        word0, word1, score = tuple(line.strip().split('\t'))
        score = float(score)

        wordPairs.append((word0, word1))
        targetScores.append(score)

    scores = []
    for word0, word1 in wordPairs:
        word0Index = wordIndexMap[word0]
        word1Index = wordIndexMap[word1]
        word0Embedding = embeddings[word0Index]
        word1Embedding = embeddings[word1Index]

        score = cosineSimilarity(word0Embedding, word1Embedding)
        scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    metric = numpy.mean([pearson, spearman])

    return metric


def evaluateWordSimilarity353(wordIndexMap, embeddings, filePath):
    data = pandas.read_csv(filePath)

    wordPairs = []
    targetScores = []
    for word0, word1, score in zip(data['Word1'], data['Word2'], data['Score']):
        if word0 in wordIndexMap and word1 in wordIndexMap:
            wordPairs.append((word0, word1))
            targetScores.append(score)

    scores = []
    for word0, word1 in wordPairs:
        word0Index = wordIndexMap[word0]
        word1Index = wordIndexMap[word1]
        word0Embedding = embeddings[word0Index]
        word1Embedding = embeddings[word1Index]

        score = cosineSimilarity(word0Embedding, word1Embedding)
        scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    metric = numpy.mean([pearson, spearman])

    return metric


def evaluateSimLex999(wordIndexMap, embeddings, filePath):
    data = pandas.read_csv(filePath, sep='\t')

    wordPairs = []
    targetScores = []
    scores = []

    for word0, word1, targetScore in zip(data['word1'], data['word2'], data['SimLex999']):
        if word0 in wordIndexMap and word1 in wordIndexMap:
            wordPairs.append((word0, word1))

            targetScores.append(targetScore)

            word0Index = wordIndexMap[word0]
            word1Index = wordIndexMap[word1]
            word0Embedding = embeddings[word0Index]
            word1Embedding = embeddings[word1Index]

            score = cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    simLex999Metric = numpy.mean([pearson, spearman])

    return simLex999Metric


def evaluateSyntacticWordRelations(wordIndexMap, embeddings, filePath, maxWords=5):
    with open(filePath, 'r') as file:
        lines = file.readlines()
        words = [tuple(line.lower().split(' ')) for line in lines if not line.startswith(':')]
        words = [(word0.strip(), word1.strip(), word2.strip(), word3.strip()) for word0, word1, word2, word3 in words]

    scores = []
    for word0, word1, word2, word3 in words:
        if word0 not in wordIndexMap or word1 not in wordIndexMap or word2 not in wordIndexMap or word3 not in wordIndexMap:
            continue

        word0Index = wordIndexMap[word0]
        word1Index = wordIndexMap[word1]
        word2Index = wordIndexMap[word2]
        word3Index = wordIndexMap[word3]

        word0Embedding = embeddings[word0Index]
        word1Embedding = embeddings[word1Index]
        word2Embedding = embeddings[word2Index]
        word3Embedding = embeddings[word3Index]

        similarity01 = cosineSimilarity(word0Embedding, word1Embedding)
        similarity23 = cosineSimilarity(word2Embedding, word3Embedding)

        score = 1
        minSimilarityDelta = abs(similarity01 - similarity23)
        for embedding in embeddings[:maxWords]:
            similarity2N = cosineSimilarity(word2Embedding, embedding)
            similarityDelta = abs(similarity01 - similarity2N)

            score = not (similarityDelta < minSimilarityDelta)
            if not score:
                break

        scores.append(score)

    syntacticWordRelationsMetric = float(sum(scores)) / len(scores)

    return syntacticWordRelationsMetric


def evaluateSATQuestions(wordIndexMap, embeddings, filePath):
    maxLineLength = 50
    aCode = ord('a')

    scores = []
    with open(filePath) as file:
        line = file.readline()
        while line != '':
            if len(line) < maxLineLength:
                match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]\:[nvar]', line)
                if match:
                    stemWord0, stemWord1 = match.group('word0'), match.group('word1')
                    validSample = stemWord0 in wordIndexMap and stemWord1 in wordIndexMap

                    if validSample:
                        stemWord0Index = wordIndexMap[stemWord0]
                        stemWord1Index = wordIndexMap[stemWord1]
                        stemWord0Embedding, stemWord10Embedding = embeddings[stemWord0Index], embeddings[stemWord1Index]
                        stemSimilarity = cosineSimilarity(stemWord0Embedding, stemWord10Embedding)

                    choiceSimilarityDeltas = []
                    line = file.readline()
                    match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]\:[nvar]', line)
                    while match:
                        choiceWord0, choiceWord1 = match.group('word0'), match.group('word1')
                        validSample = validSample and choiceWord0 in wordIndexMap and choiceWord1 in wordIndexMap

                        if validSample:
                            choiceWord0Index = wordIndexMap[choiceWord0]
                            choiceWord1Index = wordIndexMap[choiceWord1]
                            choiceWord0Embedding, choiceWord1Embedding = embeddings[choiceWord0Index], embeddings[choiceWord1Index]
                            choiceSimilarity = cosineSimilarity(choiceWord0Embedding, choiceWord1Embedding)

                            choiceSimilarityDelta = abs(stemSimilarity - choiceSimilarity)
                            choiceSimilarityDeltas.append(choiceSimilarityDelta)

                        line = file.readline()
                        match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]\:[nvar]', line)

                    if validSample:
                        choice = numpy.argmin(choiceSimilarityDeltas)
                        correctChoiceIndex = ord(line.strip()) - aCode
                        scores.append(int(choice == correctChoiceIndex))

            line = file.readline()

    metric = float(sum(scores)) / len(scores)

    return metric


def main():
    embeddingsFilePath = '../data/Text8-vectors/vectors.bin'
    wordIndexMap, embeddings = adapters.loadWord2VecEmbeddings(embeddingsFilePath)

    rgFilePath = '../data/RG/EN-RG-65.txt'
    rgMetric = evaluateRubensteinGoodenough(wordIndexMap, embeddings, rgFilePath)
    log.info('Rubenstein-Goodenough: {0:.2f}/10. State of the art: 9.15/10', rgMetric * 10)

    wordSimilarity353FilePath = '../data/WordSimilarity-353/combined.csv'
    wordSim353Metric = evaluateWordSimilarity353(wordIndexMap, embeddings, wordSimilarity353FilePath)
    log.info('WordSimilarity-353: {0:.2f}/10. State of the art: 8.1/10', wordSim353Metric * 10)

    simLex999FilePath = '../data/SimLex-999/SimLex-999.txt'
    simLex999Metric = evaluateSimLex999(wordIndexMap, embeddings, simLex999FilePath)
    log.info('SimLex-999: {0:.2f}/10. State of the art: 6.42/10', simLex999Metric * 10)

    syntWordRelFilePath = '../data/Syntactic-Word-Relations/questions-words.txt'
    syntWordRelMetric = evaluateSyntacticWordRelations(wordIndexMap, embeddings, syntWordRelFilePath)
    log.info('Syntactic word relations: {0:.2f}/10. State of the art: 10/10', syntWordRelMetric * 10)

    satQuestionsFilePath = '../data/SAT-Questions/SAT-package-V3.txt'
    satQuestionsMetric = evaluateSATQuestions(wordIndexMap, embeddings, satQuestionsFilePath)
    log.info('SAT Questions: {0:.2f}/10. State of the art: 8.15/10', satQuestionsMetric * 10)

if __name__ == '__main__':
    main()