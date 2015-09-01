import gzip
import re
import uuid


class WikipediaPage():
    restrictedHeaders = ['see also', 'footnotes', 'references', 'further reading', 'external links', 'books']

    def __init__(self, title, text):
        self.title = title
        self.text = text
        self.isArticle = ':' not in self.title

    @staticmethod
    def create(title, text):
        headings = [title] + re.findall('^=+\s*([^=]+)\s*=+$', text, flags=re.MULTILINE)
        paragraphs = re.split('^=+\s*[^=]+\s*=+$', text, flags=re.MULTILINE)

        text = ''

        for heading, paragraph in zip(headings, paragraphs):
            if heading.lower() not in WikipediaPage.restrictedHeaders:
                text += paragraph.lower()

        text = re.sub('\([^\)]+\)', '', text)
        text = re.sub('(:[^\.]\.)', '', text)
        text = re.sub('[,":]', ' ', text)
        text = re.sub('\s(\.{4,})\s', ' ', text)
        text = re.sub('\s(\.{3})\s', '.', text)
        text = re.sub('\s(\.{2})\s', ' ', text)
        text = re.sub('[^a-z]+([0-9\-]+)[^a-z]+', ' NUMBER ', text)
        text = re.sub('\s([^a-zA-Z0-9_\.\-\s]+)\s', ' SYMBOL ', text)
        text = re.sub('\s([bcdefghjklmnopqrstuvwxyz])\s', ' SYMBOL ', text)

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


testFilePath = '../data/Wikipedia/20140615-wiki-en_000022.txt.gz'
#testFilePath = 'test.txt.gz'
pages = WikipediaDumpFile.load(testFilePath)

for page in pages:
    print '{0}: {1}'.format(page.title, len(page.text))
    
    pageFilePath = 'dumps/{0}.txt.gz'.format(uuid.uuid1())
    page.dump(pageFilePath)