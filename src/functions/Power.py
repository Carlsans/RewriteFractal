import torch
from src.Configuration import Configuration


class Power:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = Configuration()
        self.setExponent()
    def setExponent(self):
        self.exponent = (torch.rand(1, device=self.device)*2-1) * torch.randint(1, 11, (1,), device=self.device)

    def getY(self,x):
        if self.conf.debugmathfunction:
            print(self.__class__.__name__)
            if torch.is_tensor(x):
                startshape = x.shape
            else:
                startshape = x.__class__.__name__
        if not torch.is_tensor(x):
            x = torch.tensor([x],device=self.device)
        result = torch.pow(x,self.exponent)
        if self.conf.debugmathfunction:
            print("ISNAN ? : ", torch.isnan(result).any())
            print("startshape", startshape)
            print("end shape : ", result.shape)
            if not (result.shape == startshape):
                print("Wrong shape")
        return result
    def setvalues(self, values):
        self.exponent = values['exponent'].to(device=self.device)

# power = Power()
# print(power.exponent)
# print(power.getY(5))
# print(power.getY(torch.tensor([[2,3,4],[4,5,6]])))