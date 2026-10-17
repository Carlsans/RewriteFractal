
import time
import cython
# distutils: language = c++
from RewriterEngineCython import RewriterEngineCython
rw = RewriterEngineCython()
#rw.fib(10000000000000000000)
teststring = "AABBAABBAABB\nAABBAABBAABB\nABCCBAABCCBAABCCBA" * 1000000
print(teststring)
teststring = teststring
timebefore = time.time()
symmetryscore = rw.mesureSymetry(teststring)
print("symmetryscore",symmetryscore)
print(time.time() - timebefore)
# NOTE: no typing required for 'uchar' !
# timebefore = time.time()
# for uchar in teststring:
#     if uchar == u'A':
#         ...
# print(time.time() - timebefore)