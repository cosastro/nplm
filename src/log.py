import sys


def info(message, insertNewLine=True):
    sys.stdout.write('\r' + message)
    if insertNewLine:
        newline()

    sys.stdout.flush()


def newline():
    sys.stdout.write('\n')
    sys.stdout.flush()

def progress(index, count, message=''):
    index = float(index)
    count = float(count)
    percentage = 100 * index / count

    message = 'Complete: {0:.3f}%. {1}'.format(percentage, message)
    info(message, False)