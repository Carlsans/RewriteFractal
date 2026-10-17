
import torch
from src.Configuration import Configuration


class Exponential:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = Configuration()


        self.setBase()
    def setBase(self):
        self.base = (torch.rand(1, device=self.device)*2-1) * torch.randint(1, 11, (1,), device=self.device)

    def getY(self,x):
        if self.conf.debugmathfunction:
            print(self.__class__.__name__)
            if torch.is_tensor(x):
                startshape = x.shape
            else:
                startshape = x.__class__.__name__

        if not torch.is_tensor(x):
            x = torch.tensor([x],device=self.device)
        result = torch.pow(self.base,x)
        if self.conf.debugmathfunction:
            print("ISNAN ? : ", torch.isnan(result).any())
            print("startshape",startshape)
            print("end shape : ",result.shape)
            if not(result.shape == startshape):
                print("Wrong shape")

        return result
    def setvalues(self, values):
        self.base = values['base'].to(device=self.device)

# expon = Exponential()
# print(expon.base)
# print(expon.getY(5))
# print(expon.getY(torch.tensor([[2,3,4],[4,5,6]])))