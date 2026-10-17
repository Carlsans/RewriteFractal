import time
from datetime import datetime
from os.path import dirname, abspath

class Configuration():
    def __init__(self):
        #self.filepath = "/home/carl/PycharmProjects/rewriteFractals/images/"
        self.filepath = dirname(dirname(abspath(__file__)))+"/images/"
        self.debugmathfunction = False
    def generateTimestamp(self,prefix = "",suffix = "",extension = ".png"):
        scaledtimestamp = str(int(time.time() * 100))

        return self.filepath + prefix+scaledtimestamp+suffix+extension
