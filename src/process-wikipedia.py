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
    def create(title, text, filter=True):
        headings = [title] + re.findall('^=+\s*([^=]+)\s*=+$', text, flags=re.MULTILINE)
        paragraphs = re.split('^=+\s*[^=]+\s*=+$', text, flags=re.MULTILINE)

        if filter:
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
            text = re.sub('<[^>]+>', '', text)
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
    
    def dump(self, filePath, compress=True):
        if compress:
            filePath = filePath + '.gz'
            with gzip.open(filePath, 'wb+') as file:
                file.write(self.text)
        else:
            with open(filePath, 'wb+') as file:
                file.write(self.text)


class WikipediaDumpFile():
    @staticmethod
    def load(filePath, filter=True):
        with gzip.open(filePath, 'rb') as testFile:
            text = testFile.read()

        titles = [link.strip() for link in re.findall('^\[\[(?P<title>[^\]]+)\]\]\s?$', text, flags=re.MULTILINE)]
        texts = [text.strip() for text in re.split('^\[\[[^\]]+\]\]\s?$', text, flags=re.MULTILINE) if text != '']

        pages = [WikipediaPage.create(link, text, filter) for link, text in zip(titles, texts)]
        pages = [page for page in pages if page.isArticle]

        return pages


def processWikipediaDumps(filterText=True, compress=True):
    outputDirectoryPath = '../data/Wikipedia-pages'

    if os.path.exists(outputDirectoryPath):
        shutil.rmtree(outputDirectoryPath, ignore_errors=True)
        log.info('Old output directory has been removed')

    os.mkdir(outputDirectoryPath)
    os.chown(outputDirectoryPath, 1000, 1000)
    log.info('Output directory has been created')

    inputDirectoryPath = '../data/Wikipedia'
    wikipediaFilesMask = inputDirectoryPath + '/*-wiki-en_*.txt.gz'

    filePaths = glob.glob(wikipediaFilesMask)[:10]
    filesCount = len(filePaths)
    fileIndex = 0
    pagesCounter = 0
    elapsed = 0.

    for filePath in sorted(filePaths):
        startTime = time.time()
        fileName = os.path.basename(filePath)
        dumpName = fileName.split('.')[0]

        pages = WikipediaDumpFile.load(filePath, filterText)

        pagesCounter += len(pages)

        pagesDirectoryPath = os.path.join(outputDirectoryPath, dumpName)
        os.mkdir(pagesDirectoryPath)
        os.chown(pagesDirectoryPath, 1000, 1000)

        for page in pages:
            pageName = re.sub('[^a-zA-Z0-9\s\(\)]', '', page.title).strip()
            pageName = '{0}.txt'.format(pageName)
            pagePath = os.path.join(pagesDirectoryPath, pageName)

            page.dump(pagePath, compress)

        endTime = time.time()
        elapsed += (endTime - startTime)
        averageElapsed = elapsed / (fileIndex + 1)

        message = 'Pages: {0}. File: {1} ({2}). Average: {3:.2f} seconds per file.'.format(pagesCounter, dumpName, len(pages), averageElapsed)
        log.progress(fileIndex + 1, filesCount, message)
        fileIndex += 1

    log.newline()
    log.info('Processing finished.')
    log.info('Elapsed {0:.2f} seconds.'.format(elapsed))
    log.info('Average {0:.2f} seconds per file.'.format(elapsed/fileIndex))
    log.info('Wikipedia dump files processed: {0}/{1}.'.format(fileIndex, filesCount))
    log.info('Wikipedia text pages created: {0}.'.format(pagesCounter, filesCount))

if __name__ == '__main__':
    processWikipediaDumps(True, True)