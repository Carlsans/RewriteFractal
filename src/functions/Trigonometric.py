import random

import torch

from src.Configuration import Configuration


class Trigonometric:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = Configuration()
        self.allowedoperations = ['sin','cos','tan','cosec','sec','cotan']
        self.setTrigoOperation()

    def setTrigoOperation(self):
        self.trigooperation = random.choice(self.allowedoperations)

    def getY(self,x):
        if self.conf.debugmathfunction:
            print(self.__class__.__name__)
            if torch.is_tensor(x):
                startshape = x.shape
            else:
                startshape = x.__class__.__name__

        if not torch.is_tensor(x):
            x = torch.tensor([x],device=self.device)
        if self.trigooperation == 'sin':
            result = torch.sin(x)
        elif self.trigooperation == 'cos':
            result = torch.cos(x)
        elif self.trigooperation == 'tan':
            result = torch.tan(x)
        elif self.trigooperation == 'cosec':
            result = 1/torch.sin(x)
        elif self.trigooperation == 'sec':
            result = 1/torch.cos(x)
        elif self.trigooperation == 'cotan':
            result = 1/torch.tan(x)
        else:
            raise Exception("Trigo operation unknown : "+self.trigooperation)

        if self.conf.debugmathfunction:
            print("ISNAN ? : ", torch.isnan(result).any())
            print("startshape", startshape)
            print("end shape : ", result.shape)
            if not (result.shape == startshape):
                print("Wrong shape")
        return result

    def setvalues(self, value):
        self.trigooperation = value['trigooperation']
# trigo = Trigonometric()
# print(trigo.trigooperation)
# print(trigo.getY(5))
# print(trigo.getY(torch.tensor([[2,3,4],[4,5,6]])))