import torch
from src.Configuration import Configuration


class Polynomial:
    def __init__(self,numterm=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = Configuration()

        self.numterm = numterm
        self.setBasePolynomials()
    def setBasePolynomials(self):
        self.basecoefficients = torch.randn((self.numterm), dtype=torch.cdouble,device=self.device)
        self.baseexposants = torch.randint(2, 10, (self.numterm,),device=self.device)
    def getY(self,x):
        if self.conf.debugmathfunction:
            print(self.__class__.__name__)
            if torch.is_tensor(x):
                startshape = x.shape
            else:
                startshape = x.__class__.__name__
        if not torch.is_tensor(x):
            x = torch.tensor([x],device=self.device)
        powers = torch.pow(x,self.baseexposants.unsqueeze(1).unsqueeze(2))
        resultunadded = powers * self.basecoefficients.unsqueeze(1).unsqueeze(2)
        #print("powers,resultunadded,torch.sum(resultunadded,0)",powers,resultunadded,torch.sum(resultunadded,0))
        result = torch.squeeze(torch.sum(resultunadded,0))
        #result = torch.unsqueeze(result,dim=0)
        #print("result",result)
        if self.conf.debugmathfunction:
            print("ISNAN ? : ", torch.isnan(result).any())
            print("startshape", startshape)
            print("end shape : ", result.shape)
            if not (result.shape == startshape):
                print("Wrong shape")
        return result

    def setvalues(self, values):
        self.basecoefficients = values['coefficients'].to(device=self.device)
        self.baseexposants = values['exposants'].to(device=self.device)

# poly = Polynomial()
# print(poly.basecoefficients)
# print(poly.baseexposants)
# poly.getY(5)
# poly.getY(torch.tensor([[2,3,4],[4,5,6]]))