import os
import shutil
import log
import glob
import time
from pathos import multiprocessing

class WikipediaPreparator():
    def __init__(self):
        self.startTime = None
        self.dumpsCount = 0
        self.flags = None


    def saveWikipediaPage(outputDirectoryPath, dumpIndex, dumpName, pageName, pageText, compress):
        pass


    def unpackWikipediaDump(self, dumpIndex, dumpPath, outputDirectoryPath, filterText, compress):
        self.startTime = time.time() if self.startTime is None else self.startTime
        dumpName = dumpPath
        pages = []
        currentTime = time.time()
        elapsed = currentTime - self.startTime
        secondsPerFile = elapsed

        log.progress('Unpacking Wikipedia dumps: {0:.3f}%. Last dump: {1} ({2} pages). Elapsed: {3:.3f} sec. ({4:.3f} sec/file)',
                     dumpIndex + 1,
                     self.dumpsCount,
                     dumpName,
                     len(pages),
                     elapsed,
                     secondsPerFile)

        time.sleep(0.1)


    def prepare(self, inputDirectoryPath, outputDirectoryPath, filterText=True, compress=True):
        if os.path.exists(outputDirectoryPath):
            shutil.rmtree(outputDirectoryPath, ignore_errors=True)
            log.info('Output directory {0} has been removed.', outputDirectoryPath)

        os.mkdir(outputDirectoryPath)
        os.chown(outputDirectoryPath, 1000, 1000)
        log.info('Output directory {0} has been created.', outputDirectoryPath)

        pathName = inputDirectoryPath + '/**/*wiki*.txt.gz'
        dumpPaths = glob.glob(pathName)
        self.dumpsCount = len(dumpPaths)
        log.info('Found {0} Wikipedia dumps.', self.dumpsCount)

        processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=processes)
        for dumpIndex, dumpPath in enumerate(dumpPaths):
            pool.apply_async(self.unpackWikipediaDump, args=(dumpIndex, dumpPath, outputDirectoryPath, filterText, compress))
        pool.close()
        pool.join()

        log.lineBreak()
        log.info('Processing complete.')

if __name__ == '__main__':
    inputDirectoryPath = '../data/Wikipedia'
    outputDirectoryPath = '../data/Wikipedia_prepared'

    wikipediaPreparator = WikipediaPreparator()
    wikipediaPreparator.prepare(inputDirectoryPath, outputDirectoryPath)