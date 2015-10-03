import log
import kit
import math
import scipy.stats
import numpy
import pandas
import re
import vectors


def rubensteinGoodenough(wordIndexMap, embeddings):
    rubensteinGoodenoughFilePath = 'res/RG/EN-RG-65.txt'

    with open(rubensteinGoodenoughFilePath) as file:
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

        score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
        scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    rubensteinGoodenoughMetric = numpy.mean([pearson, spearman])

    return rubensteinGoodenoughMetric


def wordSimilarity353(wordIndexMap, embeddings):
    wordSimilarity353FilePath = 'res/WordSimilarity-353/combined.csv'
    data = pandas.read_csv(wordSimilarity353FilePath)

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

        score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
        scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    metric = numpy.mean([pearson, spearman])

    return metric


def simLex999(wordIndexMap, embeddings):
    simLex999FilePath = 'res/SimLex-999/SimLex-999.txt'
    data = pandas.read_csv(simLex999FilePath, sep='\t')

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

            score = vectors.cosineSimilarity(word0Embedding, word1Embedding)
            scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    simLex999Metric = numpy.mean([pearson, spearman])

    return simLex999Metric


def syntacticWordRelations(wordIndexMap, embeddings, maxWords=10):
    syntWordRelFilePath = 'res/Syntactic-Word-Relations/questions-words.txt'

    with open(syntWordRelFilePath, 'r') as file:
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

        similarity01 = vectors.cosineSimilarity(word0Embedding, word1Embedding)
        similarity23 = vectors.cosineSimilarity(word2Embedding, word3Embedding)

        score = 1
        minSimilarityDelta = abs(similarity01 - similarity23)
        for embedding in embeddings[:maxWords]:
            similarity2N = vectors.cosineSimilarity(word2Embedding, embedding)
            similarityDelta = abs(similarity01 - similarity2N)

            score = not (similarityDelta < minSimilarityDelta)
            if not score:
                break

        scores.append(score)

    syntacticWordRelationsMetric = float(sum(scores)) / len(scores)

    return syntacticWordRelationsMetric


def satQuestions(wordIndexMap, embeddings):
    satQuestionsFilePath = 'res/SAT-Questions/SAT-package-V3.txt'

    maxLineLength = 50
    aCode = ord('a')

    scores = []
    with open(satQuestionsFilePath) as file:
        line = file.readline()
        while line != '':
            if len(line) < maxLineLength:
                match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                if match:
                    stemWord0, stemWord1 = match.group('word0'), match.group('word1')
                    validSample = stemWord0 in wordIndexMap and stemWord1 in wordIndexMap

                    if validSample:
                        stemWord0Index = wordIndexMap[stemWord0]
                        stemWord1Index = wordIndexMap[stemWord1]
                        stemWord0Embedding, stemWord10Embedding = embeddings[stemWord0Index], embeddings[stemWord1Index]
                        stemSimilarity = vectors.cosineSimilarity(stemWord0Embedding, stemWord10Embedding)

                    choiceSimilarityDeltas = []
                    line = file.readline()
                    match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)
                    while match:
                        choiceWord0, choiceWord1 = match.group('word0'), match.group('word1')
                        validSample = validSample and choiceWord0 in wordIndexMap and choiceWord1 in wordIndexMap

                        if validSample:
                            choiceWord0Index = wordIndexMap[choiceWord0]
                            choiceWord1Index = wordIndexMap[choiceWord1]
                            choiceWord0Embedding, choiceWord1Embedding = embeddings[choiceWord0Index], embeddings[choiceWord1Index]
                            choiceSimilarity = vectors.cosineSimilarity(choiceWord0Embedding, choiceWord1Embedding)

                            choiceSimilarityDelta = abs(stemSimilarity - choiceSimilarity)
                            choiceSimilarityDeltas.append(choiceSimilarityDelta)

                        line = file.readline()
                        match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]:[nvar]', line)

                    if validSample:
                        choice = numpy.argmin(choiceSimilarityDeltas)
                        correctChoiceIndex = ord(line.strip()) - aCode
                        scores.append(int(choice == correctChoiceIndex))

            line = file.readline()

    metric = float(sum(scores)) / len(scores)

    return metric


def validate(wordVocabulary, wordEmbeddigs):
    rg = 0
    sim353 = 0
    simLex999 = 0
    syntRel = 0
    sat = 0
    total = 0

    return rg, sim353, simLex999, syntRel, sat, total


def dump(epoch, superBatchIndex, rg, sim353, simLex999, syntRel, sat, total, metricsPath):
    return 0, 0, 0, 0, 0, 0


def main():
    embeddingsFilePath = '../data/Text8/Processed/vectors.bin'
    wordIndexMap, embeddings = kit.loadWord2VecEmbeddings(embeddingsFilePath)

    rgMetric = rubensteinGoodenough(wordIndexMap, embeddings)
    log.info('Rubenstein-Goodenough: {0:.2f}/10. State of the art: 9.15/10', rgMetric * 10)

    wordSim353Metric = wordSimilarity353(wordIndexMap, embeddings)
    log.info('WordSimilarity-353: {0:.2f}/10. State of the art: 8.1/10', wordSim353Metric * 10)

    simLex999Metric = simLex999(wordIndexMap, embeddings)
    log.info('SimLex-999: {0:.2f}/10. State of the art: 6.42/10', simLex999Metric * 10)

    syntWordRelMetric = syntacticWordRelations(wordIndexMap, embeddings)
    log.info('Syntactic word relations: {0:.2f}/10. State of the art: 10/10', syntWordRelMetric * 10)

    satQuestionsMetric = satQuestions(wordIndexMap, embeddings)
    log.info('SAT Questions: {0:.2f}/10. State of the art: 8.15/10', satQuestionsMetric * 10)


if __name__ == '__main__':
    main()