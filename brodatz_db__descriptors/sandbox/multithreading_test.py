from multiprocessing import Process
from threading import Thread
import sys
import time
from multiprocessing import Pool


def func1(sleep, number_of_repeats, s, arr):

    line = s + arr[1](arr[0]) + '\n'
    for i in xrange(number_of_repeats):
        print line
        time.sleep(sleep)
    print 'FINISHED: ' + line


def prefix(s):
    v = [0]*len(s)
    for i in xrange(1,len(s)):
        k = v[i-1]
        while k > 0 and s[k] <> s[i]:
            k = v[k-1]
        if s[k] == s[i]:
            k = k + 1
        v[i] = k
    return v

def f(s):
    a = 0
    for i in xrange(100000000):
        a += 1
    print len(s)

if __name__=='__main__':
    # print prefix('abaaabbaabaabab')
    # p1 = Thread(target = func1, args=[0.5, 20, "A", ['1', lambda x: x+x]])
    # p2 = Thread(target = func1, args=[0.3, 10, "B", ['8', lambda x: x+x+x]])
    # p1.start()
    # p2.start()
    #
    # p1.join()
    # p2.join()



    pool = Pool(processes = 4)
    # pool.map(func1, [[0.5, 20, "A", ['1', lambda x: x+x]],
    #                  [0.3, 10, "B", ['8', lambda x: x+x+x]]])
    pool.map(f, [[0.5, 20, "A", ['1', lambda x: x+x]], 'b'])
    pool.close()
    pool.join()

    print 'TOTAL END'