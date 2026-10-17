import glob

import jsonpickle
from src.GeneticAlgorithm import GeneticAlgorithm

class FractalUpscaler:
    def generateUpscaledFromFolder(self):
        self.genalgo = GeneticAlgorithm()
        path = '/home/carl/PycharmProjects/RewriteFractal/images/tofullscale/' + "*.json"
        self.files = glob.glob(path)
        for file in self.files:
            print("Generating fractal for",file)
            with open(file, 'r') as fread:
                individual = jsonpickle.decode(fread.read())
                self.genalgo.draw_individual(individual,(4000,4000),notxtandjson=True)

upscaler = FractalUpscaler()
upscaler.generateUpscaledFromFolder()
    # def draw_individual(self, individual, resolution=(894 * 3, int(865 * 3))):
    #     self.fractal.__init__(self.sieve, resolution)
    #     rewriter = RewriterEngine(individual[0], individual[1], individual[2], self.fractal, maxiter=100)
    #     rewriter.rewrite()
    #     issane = rewriter.sanitycheck()
    #     if issane:
    #         rewriter.calculateConstants()
    #         rewriter.runAllLines()
    #         with open(self.conf.generateTimestamp(extension=".txt"), 'w') as fout:
    #             fout.write(rewriter.getLinesAsString())
    #         with open(self.conf.generateTimestamp(extension=".json"), 'w') as fout:
    #             jsondump = [individual[0], individual[1], individual[2]]
    #             # print(jsondump)
    #             # json.dump(list, jsondump)
    #             json_string = jsonpickle.encode(jsondump)
    #             fout.write(json_string)
    #         self.fractal.drawFractal(saveonly=True)