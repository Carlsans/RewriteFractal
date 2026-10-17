from src.Fractal import Fractal
from src.RewriterEngine import RewriterEngine
from src.RewriterRandomGenerator import RewriterRandomGenerator
from src.Sieve import Sieve


class Miner:
    def mineFractals(self,savehidef=True):
        sieve = Sieve()
        while (True):
            randomrewriter = RewriterRandomGenerator("ABCZ", "+-*/")
            startstring, rewriterules, constantsvalues = randomrewriter.randomstartstring(), randomrewriter.randomrules(), randomrewriter.setrandomconstants()
            #print("startstring =", startstring, "rewriterules =", rewriterules, "constantsvalues =", constantsvalues)
            fractal = Fractal()
            self.rewriter = RewriterEngine(startstring, rewriterules, constantsvalues, fractal, maxiter=100)
            self.rewriter.rewrite()
            issane = self.rewriter.sanitycheck()
            if issane:
                self.rewriter.calculateConstants()
                self.rewriter.runAllLines()
                note = sieve.getimageinformationsnote(fractal.bailout)
                print("note",note)
                if note > 0.1:
                    fractal.drawFractal(saveonly=True)
                if note > 0.2:
                    self.redrawfractal((2048,2048))

    def redrawfractal(self,resolution):
        self.rewriter.fractal = Fractal(resolution)
        self.rewriter.runAllLines()
        self.rewriter.fractal.drawFractal(saveonly=True)


miner = Miner()
miner.mineFractals()