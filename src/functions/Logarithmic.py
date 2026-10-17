import torch
from src.Configuration import Configuration


class Logarithmic:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = Configuration()

        self.setBase()
    def setBase(self):
        self.base = torch.rand(1, device=self.device) * torch.randint(1, 21, (1,), device=self.device) + 0.00000001

    def getY(self,x):
        if self.conf.debugmathfunction:
            print(self.__class__.__name__)
            if torch.is_tensor(x):
                startshape = x.shape
            else:
                startshape = x.__class__.__name__

        if not torch.is_tensor(x):
            x = torch.tensor([x],device=self.device)
        result = torch.log(x)/torch.log(self.base)
        if self.conf.debugmathfunction:
            print("ISNAN ? : ", torch.isnan(result).any())
            print("startshape", startshape)
            print("end shape : ", result.shape)
            if not (result.shape == startshape):
                print("Wrong shape")
        return result
    def setvalues(self, values):
        self.base = values['base'].to(device=self.device)
        if self.base <=0 :
            raise Exception("Logarithmic base must be over 0, "+str(self.base))

# logar = Logarithmic()
# print(logar.base)
# print(logar.getY(5))
# print(logar.getY(torch.tensor([[2,3,4],[4,5,6]])))