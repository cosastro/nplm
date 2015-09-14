from processgooglenews import *


def loadEmbeddings(embeddingsFilePath, stringBased):
    if stringBased:
        return loadWordVectors(embeddingsFilePath)

    return None


def evaluateRubensteinGoodenough(embeddings, filePath):
    return 0


def evaluateWordSimilarity353(embeddings, filePath):
    return 0


def evaluateSimLex999(embeddings, filePath):
    return 0


def evaluateSyntacticWordRelations(embeddings, filePath):
    return 0


def evaluateSATQuestions(embeddings, filePath):
    return 0


if __name__ == '__main__':
    embeddingsFilePath = '../data/Text8-vectors/vectors.bin'

    embeddings = loadEmbeddings(embeddingsFilePath, True)

    rubensteinGoodenoughFilePath = '../data/RG/EN-RG-65.txt'
    rubensteinGoodenoughMetric = evaluateRubensteinGoodenough(embeddings, rubensteinGoodenoughFilePath)

    wordSimilarity353FilePath = '../data/WordSimilarity-353/combined.csv'
    wordSimilarity353Metric = evaluateWordSimilarity353(embeddings, wordSimilarity353FilePath)

    simLex999FilePath = '../data/SimLex-999/SimLex-999.txt'
    simLex999Metric = evaluateSimLex999(embeddings, simLex999FilePath)

    syntacticWordRelationsFilePath = '../data/Syntactic-Word-Relations/questions-words.txt'
    syntacticWordRelationsMetric = evaluateSyntacticWordRelations(embeddings, syntacticWordRelationsFilePath)

    satQuestionsFilePath = '../data/SAT-Questions/SAT-package-V3.txt'
    satQuestionsMetric = evaluateSATQuestions(embeddings, satQuestionsFilePath)

    log.info('Rubenstein-Goodenough: {0}/10'.format(rubensteinGoodenoughMetric))
    log.info('WordSimilarity-353: {0}/10'.format(wordSimilarity353Metric))
    log.info('SimLex-999: {0}/10'.format(simLex999Metric))
    log.info('Syntactic word relations: {0}/10'.format(syntacticWordRelationsMetric))
    log.info('SAT Questions: {0}/10'.format(satQuestionsMetric))