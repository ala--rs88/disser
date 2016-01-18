__author__ = 'IgorKarpov'

import time

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def enter(self):
        self.start = time.time()
        return self

    def exit(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs