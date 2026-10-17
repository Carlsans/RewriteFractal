import random
from collections import OrderedDict
import Crypto.Random.random as rnd
import torch

from src.functions.Trigonometric import Trigonometric


class RewriterRandomGenerator():
    def __init__(self,vocabulary,operators):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocabulary = vocabulary
        self.operators = operators
        self.startstringlength = [3,15]
        self.randomstartstring()
        self.ruleslength = [1,5]
        self.rulescount = [2,35]
        self.randomconstantrange = [-1,1]
        self.randomtailorrange = [-1,1]
        self.randomcenterrange = [-10,10]
        self.randomfractalrange = [0.00000001,4]
        self.tailornumterm = 6
        self.polynomialnumterm = 6
        #self.setrandomconstants()
    def randomstartstring(self):
        #stringlength = random.randint(self.startstringlength[0],self.startstringlength[1])
        stringlength = rnd.randrange(self.startstringlength[0],self.startstringlength[1])
        startstring = ""
        for i in range(stringlength):
            if (i+1) % 2:
                startstring += rnd.choice(self.vocabulary)
            else:
                startstring += rnd.choice(self.vocabulary+self.operators)

        return startstring
    def randomrules(self):
        ruledict = OrderedDict()
        for i in range(rnd.randrange(self.rulescount[0],self.rulescount[1])):
            thisrulelenght = rnd.randrange(self.ruleslength[0],self.ruleslength[1])
            thisrulestrtoreplace = ""
            thisrulestrreplacement = ""
            for j in range(thisrulelenght):
                if (i + 1) % 2:
                    thisrulestrtoreplace += rnd.choice(self.vocabulary)
                else:
                    thisrulestrtoreplace += rnd.choice(self.vocabulary+self.operators)
            thisrulelenght = rnd.randrange(self.ruleslength[0],self.ruleslength[1])
            for j in range(thisrulelenght):
                if (i + 1) % 2:
                    thisrulestrreplacement += rnd.choice(self.vocabulary)
                else:
                    thisrulestrreplacement += rnd.choice(self.vocabulary+self.operators)
            ruledict[thisrulestrtoreplace] = thisrulestrreplacement
        return ruledict
    def setrandomconstants(self,hyperparameters):
        constantdict = OrderedDict()
        # Tuple representing fractal center
        constantdict['center'] = [random.uniform(self.randomcenterrange[0],self.randomcenterrange[1]),random.uniform(self.randomcenterrange[0],self.randomcenterrange[1])]
        # Range of the fractal
        constantdict['range'] = random.uniform(self.randomfractalrange[0],self.randomfractalrange[1])
        for i in self.vocabulary:
            if i in 'ZC':
                continue
            if i.isupper():
                constantdict[i] = random.uniform(self.randomconstantrange[0], self.randomconstantrange[1])+random.uniform(self.randomconstantrange[0], self.randomconstantrange[1])*1j
            else:
                fnchoice = random.choice(['randomtaylor','commontaylor','randompolynomial','randomrational','randompower','randomtrigonometric','randomexponential','randomlogarithmic'])
                if fnchoice == 'randomtaylor':
                    numberofconstants = rnd.randint(1,self.tailornumterm)
                    constantdict[i] =  {'fntype' : 'Taylorian', 'value' : torch.tensor([random.uniform(self.randomtailorrange[0], self.randomtailorrange[1])+random.uniform(self.randomtailorrange[0], self.randomtailorrange[1])*1j for i in range(numberofconstants)]).to(self.device)}

                elif fnchoice == 'commontaylor':
                    constantdict[i] = {'fntype' : 'Taylorian', 'value' : self.getrandomcommontaylorseries()}
                elif fnchoice == 'randompolynomial':
                    #print("hyperparameters.polynomialexposant",hyperparameters.polynomialexposant)
                    constantdict[i] = {'fntype' : 'Polynomial', 'value' : {'coefficients' : torch.randn((self.polynomialnumterm), dtype=torch.cdouble)*hyperparameters.polynomialcoeffmultiplier, 'exposants' : torch.randint(1, hyperparameters.polynomialexposant+1, (self.polynomialnumterm,)) }}
                elif fnchoice == 'randomrational':
                    #print("hyperparameters.polynomialexposant",hyperparameters.polynomialexposant)
                    constantdict[i] = {'fntype' : 'Rational', 'value' : {
                        'numeratorcoefficients' : torch.randn((self.polynomialnumterm), dtype=torch.cdouble)*hyperparameters.rationalnumeratorcoeffmultiplier,
                        'numeratorexposants' : torch.randint(1, hyperparameters.rationalnumeratorexposant+1, (self.polynomialnumterm,)),
                        'denominatorcoefficients': torch.randn((self.polynomialnumterm),
                                                             dtype=torch.cdouble) * hyperparameters.rationaldenominatorcoeffmultiplier,
                        'denominatorexposants': torch.randint(1, hyperparameters.rationaldenominatorexposant + 1,
                                                            (self.polynomialnumterm,))
                        }
                    }
                elif fnchoice == 'randompower':
                    constantdict[i] = {'fntype': 'Power', 'value': {'exponent' : (torch.rand(1, device=self.device)*2-1) * torch.randint(1, 11, (1,), device=self.device) }}
                elif fnchoice == 'randomtrigonometric':
                    trigo = Trigonometric()
                    constantdict[i] = {'fntype': 'Trigonometric',
                                       'value': {'trigooperation' :  random.choice(trigo.allowedoperations) }}
                elif fnchoice == 'randomexponential':
                    constantdict[i] = {'fntype': 'Exponential',
                                       'value': {'base': (torch.rand(1, device=self.device)*2-1) * torch.randint(1, 11, (1,), device=self.device)}}
                elif fnchoice == 'randomlogarithmic':
                    constantdict[i] = {'fntype': 'Logarithmic',
                                       'value': {
                                           'base': torch.rand(1, device=self.device) * torch.randint(1, 21, (1,), device=self.device)++ 0.00000001}}

        return constantdict
    def getrandomcommontaylorseries(self):
        serieslist = [torch.tensor([1],device=self.device), # exp
                      torch.tensor([0,-1,-1,-1,-1,-1], device=self.device),  # ln(1-x)
                      torch.tensor([0, 1,-1,-1,-1,-1], device=self.device),  # ln(1+x)
                      torch.tensor([0, 1, 0,-1, 0, 1], device=self.device)  # sin
                      # torch.tensor([], device=self.device),  # cos
                      # torch.tensor([], device=self.device),  # tan
                      # torch.tensor([], device=self.device),  #
                      # torch.tensor([], device=self.device),  #
                      # torch.tensor([], device=self.device),  #

                      ]
        return random.choice(serieslist)

