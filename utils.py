import sys

def info(msg):
    '''Print informational messages to stderr so that structured output (stdout e.g. CSV) stays clean'''
    print(msg, file=sys.stderr)
