import torch
from src.Configuration import Configuration

from src.functions.Polynomial import Polynomial


class Rational:
    def __init__(self,numterm=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = Configuration()
        self.numterm = numterm
        self.setBaseRational()

    def setBaseRational(self):
       self.numerator = Polynomial(self.numterm // 2)
       self.denominator = Polynomial(self.numterm//2)

    def getY(self,x):
        if self.conf.debugmathfunction:
            print(self.__class__.__name__)
            if torch.is_tensor(x):
                startshape = x.shape
            else:
                startshape = x.__class__.__name__
        result = self.numerator.getY(x)/self.denominator.getY(x)
        if self.conf.debugmathfunction:
            print("ISNAN ? : ", torch.isnan(result).any())
            print("startshape", startshape)
            print("end shape : ", result.shape)
            if not (result.shape == startshape):
                print("Wrong shape")
        return result
    def setvalues(self, values):
        self.numerator.setvalues({'coefficients' : values['numeratorcoefficients'].to(device=self.device),
                                  'exposants' : values['numeratorexposants'].to(device=self.device)})
        self.denominator.setvalues({'coefficients': values['denominatorcoefficients'].to(device=self.device),
                                  'exposants': values['denominatorexposants'].to(device=self.device)})
# rat = Rational()
# print(rat.getY(5))
# print(rat.getY(torch.tensor([[2,3,4],[4,5,6]])))