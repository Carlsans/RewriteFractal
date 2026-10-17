import json
import time

import torch
#from torchvision.transforms import ToPILImage
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from src.ComplexMatrix import ComplexMatrix
from src.Configuration import Configuration
from src.Sieve import Sieve
from src.TimeoftheDayColormap import TimeoftheDayColormap


class Fractal:
    def __init__(self,sieve,center,range,resolution = (512,512) ,useexitbailoutvalues = False):
        self.conf = Configuration()
        self.tod = TimeoftheDayColormap()
        self.resolution= resolution
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bailout = torch.zeros(resolution, dtype=torch.int16, device=device,requires_grad=False)
        self.bailoutexitvalues = torch.zeros(resolution, dtype=torch.complex128, device=device,requires_grad=False)
        self.Z = torch.zeros(resolution, dtype=torch.complex128, device=device,requires_grad=False)
        self.C = torch.zeros(resolution,dtype=torch.complex128,device=device,requires_grad=False)
        rangereal = (center[0]-range,center[0]+range)
        rangeimag = (center[1] - range, center[1] + range)
        self.C += torch.linspace(rangereal[0],rangereal[1],resolution[1], device=device) * torch.ones(resolution[0],1, device=device)
        self.C += torch.linspace(rangeimag[0],rangeimag[1],resolution[0], device=device).reshape(-1,1) * torch.ones(1,resolution[1], device=device) *1j
        self.sieve = sieve
        self.complexmatrix = ComplexMatrix()
        self.useexitbailoutvalues = useexitbailoutvalues
    def bailouttest(self,iteration):
        if self.Z is None or not torch.is_tensor(self.Z):
            return False
        if self.Z.shape == torch.Size([1])or self.Z.shape == torch.Size([]):
            return False

        #print("self.Z",self.Z.shape)
        isbailingout = self.Z.real+self.Z.imag>4
        #print("iteration",iteration,"isbailingout",isbailingout,'self.bailout',self.bailout)
        self.bailout[isbailingout] = iteration
        #print("self.bailoutexitvalues",self.bailoutexitvalues,self.bailoutexitvalues.dtype,"self.Z",self.Z,self.Z.dtype,self.Z.shape)

        #nancompare = self.Z == ((float('nan') + float('nan')*1j)*torch.ones_like(self.Z))
        #print("all nan", nancompare)
        # if torch.all(nancompare):
        #     print("bbbbllllooouuuu")
        #     exit()
        #print("isbailingout",isbailingout)
        self.bailoutexitvalues[isbailingout] = self.Z[isbailingout]
        # Here, Z might only be a complex number
        if torch.is_tensor(self.Z):
            self.Z[isbailingout] = float('nan')+ float('nan')*1j
        return True
        #print(self.Z)
    def addnotbailedout(self):
        if self.Z == None:
            return
        nonbailedout = self.Z.real + self.Z.imag > 4
        if isinstance(nonbailedout,bool):
            return
        print("nonbailedout.shape",nonbailedout.shape)
        if not(nonbailedout.shape == self.bailoutexitvalues.shape and self.bailoutexitvalues.shape == self.Z.shape):
            return
        self.bailoutexitvalues[nonbailedout] = self.Z[nonbailedout]

    def invalidZtest(self):
        if not torch.is_tensor(self.Z):
            return True
        sumz = torch.sum(self.Z)
        #print("sumz",sumz)
        if sumz == 0:
            return True
        return False
    def zeroBailouttest(self):
        if not torch.is_tensor(self.bailout):
            return True
        maxbailout = torch.max(self.bailout)
        #print("sumbailout", sumbailout)
        if maxbailout == 0 or maxbailout == 1:
            return True
        return False
    def moreFrequentBailout(self):
        # Get unique elements and their counts
        unique_elements, counts = torch.unique(torch.flatten(self.bailout), return_counts=True)
        # Find the most frequent element
        most_frequent = unique_elements[torch.argmax(counts)]
        most_frequent_count = counts.max()
        return most_frequent_count/(self.bailout.shape[0]*self.bailout.shape[1])
    def printcomplexmatrix(self):
        r,i = self.complexmatrix.splitmatrix(self.bailoutexitvalues)
        print("r,i",r,i)

    def getNormalizedBailout(self):
        if self.zeroBailouttest():
            return None
        normalizedbailout = self.bailout / torch.max(self.bailout)
        return normalizedbailout
    def getNormalizedExitValues(self):
        return self.complexmatrix.normalizeComplexMatrix(self.bailoutexitvalues)


    def drawFractal(self,saveonly = False,saveblackandwhitetoo= True):
        if self.useexitbailoutvalues:
            normalizedbailout = self.getNormalizedExitValues()
        else:
            normalizedbailout = self.bailout / torch.max(self.bailout)
        #print(normalizedbailout)
        #colormap = cm.get_cmap('gist_earth')

        colormap = cm.get_cmap(self.tod.timeofthedaycolormap())
        fractalimage = colormap(normalizedbailout.cpu())
        #print(fractalimage.shape)
        sizebydim = self.sieve.getimageinformationsnote(fractalimage)
        #print("sizebydim",sizebydim)

        if saveonly:
            plt.imsave(self.conf.generateTimestamp(suffix="r"+str(self.resolution[0])+"x"+str(self.resolution[1])),fractalimage)
        else:
            plt.imshow(fractalimage)
            plt.show()
        if saveblackandwhitetoo:
            colormap = cm.get_cmap('binary')
            fractalimage = colormap(normalizedbailout.cpu())
            plt.imsave(self.conf.generateTimestamp(prefix="bin",suffix="r"+str(self.resolution[0])+"x"+str(self.resolution[1])),fractalimage)


#
# fract = Fractal()
# print(fract.C)