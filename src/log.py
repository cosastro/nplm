import sys


def info(message, *args):
    message = str(message)
    message = message.format(*args)
    message = '\r' + message

    sys.stdout.write(message)
    sys.stdout.flush()

    lineBreak()


def lineBreak():
    sys.stdout.write('\n')
    sys.stdout.flush()


def progress(message, index, count, *args):
    index = float(index)
    count = float(count)
    percentage = 100 * index / count
    args = [percentage] + list(args)

    message = message or 'Current progress: {0:.3f}%'
    message = message.format(*args)
    message = '\r' + message

    sys.stdout.write(message)
    sys.stdout.flush()