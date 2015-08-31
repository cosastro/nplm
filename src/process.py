import gzip
import re


class WikipediaPage():
    def __init__(self, title, text):
        self.title = title
        self.text = text
        self.isArticle = ':' not in self.title

    @staticmethod
    def create(title, text):
        return WikipediaPage(title, text)


class WikipediaDumpFile():
    @staticmethod
    def load(filePath):
        with gzip.open(filePath, 'rb') as testFile:
            text = testFile.read()

        links = [link.strip() for link in re.findall('^\[\[(?P<link>[^\]]+)\]\]\s?$', text, flags=re.MULTILINE)]
        texts = [text.strip() for text in re.split('^\[\[[^\]]+\]\]\s?$', text, flags=re.MULTILINE) if text != '']

        pages = [WikipediaPage.create(link, text) for link, text in zip(links, texts)]
        pages = [page for page in pages if page.isArticle]

        return pages


#testFilePath = '../data/Wikipedia/20140615-wiki-en_000022.txt.gz'
testFilePath = '../data/Wikipedia/test.txt.gz'
pages = WikipediaDumpFile.load(testFilePath)

for page in pages:
    print '{0}: {1}'.format(page.title, len(page.text))