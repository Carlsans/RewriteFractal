import numpy as np
import torch
from src.Configuration import Configuration


class TaylorSeries:
    def __init__(self,numterm=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = Configuration()
        self.numterm = numterm
        self.setBaseCoefficients()

    def setcoefficients(self,coefficients):
        # if isinstance(coefficients, list):
        #     coefficients = torch.tensor(coefficients)

        if not len(coefficients.shape) == 1:
            raise Exception("Coefficient must be a tensor with one dimension." + str(coefficients.shape))
        # Here, the number of coefficient is lower than the terms count.
        # We then repeat the existing terms so that we have the same number
        # of terms and coefficients.
        if coefficients.shape[0] < self.numterm:
            coeffindex=0
            newcoeffindex = 0
            newcoeff = torch.zeros((self.numterm),device=self.device)
            while(newcoeffindex < self.numterm ):
                #print(coeffindex)
                newcoeff[newcoeffindex] = coefficients[coeffindex]
                if coeffindex < coefficients.shape[0]- 1:
                    coeffindex += 1
                else:
                    coeffindex = 0
                newcoeffindex +=1
            self.coefficients = newcoeff
        else:
            self.coefficients = coefficients[0:self.numterm]
        #print(self.coefficients.to(self.device) , self.basecoefficients)
        self.coefficients = self.coefficients.to(self.device) * self.basecoefficients
        #print(self.coefficients)

    def setBaseCoefficients(self):
        #basecoeff = np.multiply(np.arange(1, self.numterm + 1))
        self.basecoefficients = torch.zeros((self.numterm),device=self.device)
        for i in range(self.numterm):
            values = torch.arange(1, i + 1).to(self.device)  # Create a tensor [1, 2, 3, ..., n]
            self.basecoefficients[i]  = 1/torch.prod(values)
        #print("self.basecoefficients",self.basecoefficients)

    def getY(self,x):
        if self.conf.debugmathfunction:
            print(self.__class__.__name__)
            if torch.is_tensor(x):
                startshape = x.shape
            else:
                startshape = x.__class__.__name__
        if not torch.is_tensor(x):
            x = torch.tensor([x],device=self.device)
        #print("x",x)
        powers = torch.stack([x ** i for i in range(self.numterm)], dim=-1)
        #print("powers",powers)
        #print("self.coefficients",self.coefficients)
        #print("sum",torch.sum(powers*self.coefficients,dim=-1))
        result = torch.sum(powers*self.coefficients,dim=-1)
        if self.conf.debugmathfunction:
            print("ISNAN ? : ", torch.isnan(result).any())
            print("startshape", startshape)
            print("end shape : ", result.shape)
            if not (result.shape == startshape):
                print("Wrong shape")
        return result
# series = TaylorSeries()
# series.setcoefficients(torch.tensor([1,-1,0,1,1,-1]))
# series.setBaseCoefficients()
# y = series.getY(torch.tensor([[2,3,4],[4,5,6]]))
# print(y)