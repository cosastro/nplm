import gzip
import re
import os
import shutil
import glob
import uuid
import sys
import log
import time


class WikipediaPage():
    restrictedHeaders = ['see also', 'footnotes', 'references', 'further reading', 'external links', 'books']

    def __init__(self, title, text):
        self.title = title
        self.text = text

        mayReferTo = '{0} may refer to'.format(title).lower()
        self.isArticle = len(text) > 2500 \
                         and ':' not in self.title \
                         and not text.startswith(mayReferTo) \
                         and not text.startswith('#redirect')

    @staticmethod
    def create(title, text):
        headings = [title] + re.findall('^=+\s*([^=]+)\s*=+$', text, flags=re.MULTILINE)
        paragraphs = re.split('^=+\s*[^=]+\s*=+$', text, flags=re.MULTILINE)

        text = ''

        for heading, paragraph in zip(headings, paragraphs):
            if heading.lower() not in WikipediaPage.restrictedHeaders:
                text += paragraph.lower()

        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
        text = re.sub('\([^\)]+\)', '', text)
        text = re.sub('(:[^\.]\.)', '', text)
        text = re.sub('[,":_\*]', ' ', text)
        text = re.sub('!', '.', text)
        text = re.sub('\?', '.', text)
        text = re.sub('\s(\.{4,})\s', ' ', text)
        text = re.sub('\s(\.{3})\s', '.', text)
        text = re.sub('\s(\.{2})\s', ' ', text)
        text = re.sub('[^a-z]+([0-9\-]+)[^a-z]+', ' NUMBER ', text)
        text = re.sub('\s([^a-zA-Z0-9\.\-\s]+)\s', ' SYMBOL ', text)
        text = re.sub('\s([bcdefghjklmnopqrstuvwxyz])\s', ' SYMBOL ', text)

        sentences = re.split('[(\n{2,})\.;]', text)
        sentences = [re.sub('[\s]+', ' ', sentence).strip() for sentence in sentences]
        sentences = [sentence for sentence in sentences
                     if len(sentence.split(' ')) > 5 \
                     and sentence.count('NUMBER') < 3]

        text = '. '.join(sentences)

        return WikipediaPage(title, text)
    
    def dump(self, filePath):
        with gzip.open(filePath, 'wb+') as file:
            file.write(self.title + '\n')
            file.write(self.text)


class WikipediaDumpFile():
    @staticmethod
    def load(filePath):
        with gzip.open(filePath, 'rb') as testFile:
            text = testFile.read()

        titles = [link.strip() for link in re.findall('^\[\[(?P<title>[^\]]+)\]\]\s?$', text, flags=re.MULTILINE)]
        texts = [text.strip() for text in re.split('^\[\[[^\]]+\]\]\s?$', text, flags=re.MULTILINE) if text != '']

        pages = [WikipediaPage.create(link, text) for link, text in zip(titles, texts)]
        pages = [page for page in pages if page.isArticle]

        return pages

outputDirectoryPath = '../data/Wikipedia-pages'

if os.path.exists(outputDirectoryPath):
    shutil.rmtree(outputDirectoryPath, ignore_errors=True)
    log.info('Old output directory has been removed')

os.mkdir(outputDirectoryPath)
os.chown(outputDirectoryPath, 1000, 1000)
log.info('Output directory has been created')

inputDirectoryPath = '../data/Wikipedia'
wikipediaFilesMask = inputDirectoryPath + '/*-wiki-en_*.txt.gz'

wikipediaFilePaths = glob.glob(wikipediaFilesMask)
wikipediaFilesCount = len(wikipediaFilePaths)
wikipediaFileIndex = 0
pagesCounter = 0
elapsed = 0.

for wikipediaFilePath in sorted(wikipediaFilePaths):
    startTime = time.time()
    wikipediaFileName = os.path.basename(wikipediaFilePath)
    wikipediaDumpName = wikipediaFileName.split('.')[0]

    wikipediaPages = WikipediaDumpFile.load(wikipediaFilePath)

    pagesCounter += len(wikipediaPages)

    wikipediaPagesDirectory = os.path.join(outputDirectoryPath, wikipediaDumpName)
    os.mkdir(wikipediaPagesDirectory)
    os.chown(wikipediaPagesDirectory, 1000, 1000)

    for wikipediaPage in wikipediaPages:
        # wikipediaPageName = '{0}.txt.gz'.format(uuid.uuid4())
        wikipediaPageName = re.sub('[^a-zA-Z0-9\s]', '', wikipediaPage.title).strip()
        wikipediaPageName = '{0}.txt.gz'.format(wikipediaPageName)
        wikipediaPagePath = os.path.join(wikipediaPagesDirectory, wikipediaPageName)

        wikipediaPage.dump(wikipediaPagePath)

    endTime = time.time()
    elapsed += (endTime - startTime)
    averageElapsed = elapsed / (wikipediaFileIndex + 1)

    message = 'Pages: {0}. File: {1} ({2}). Average: {3:.2f} seconds per file.'.format(pagesCounter, wikipediaDumpName, len(wikipediaPages), averageElapsed)
    log.progress(wikipediaFileIndex + 1, wikipediaFilesCount, message)
    wikipediaFileIndex += 1

log.newline()
log.info('Processing finished.')
log.info('Elapsed {0:.2f} seconds.'.format(elapsed))
log.info('Average {0:.2f} seconds per file.'.format(elapsed/wikipediaFileIndex))
log.info('Wikipedia dump files processed: {0}/{1}.'.format(wikipediaFileIndex, wikipediaFilesCount))
log.info('Wikipedia text pages created: {0}.'.format(pagesCounter, wikipediaFilesCount))