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


def progress(messageFormat, index, count, *args):
    index = float(index)
    count = float(count)
    percentage = 100 * index / count
    percentage = min(100, percentage)
    args = [percentage] + list(args)

    messageFormat = messageFormat.format(*args)
    messageFormat = '\r' + messageFormat

    sys.stdout.write(messageFormat)
    sys.stdout.flush()


def delta(seconds):
        seconds = int(seconds)
        if seconds == 0:
            return '0 seconds'

        periods = [('day', 60*60*24), ('hour', 60*60), ('minute', 60), ('second', 1)]

        strings=[]

        for periodName, periodSeconds in periods:
            if seconds >= periodSeconds:
                periodValue , seconds = divmod(seconds, periodSeconds)

                if periodValue == 1:
                    strings.append("%s %s" % (periodValue, periodName))
                else:
                    strings.append("%s %ss" % (periodValue, periodName))

        return ", ".join(strings)