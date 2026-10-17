import numpy as np
import torch

from src.autoencoder.Configuration import Configuration as AEConfiguration
from src.autoencoder.ConvAutoEncoderwithClassifier import ConvAutoencoderwithClassifier
from src.autoencoder.ConvVAEwithClassifier import ConvVAEwithClassifier
from src.autoencoder.DatasetLoader import DatasetLoader
import torch.nn.functional as F

class Inference:
    def __init__(self):
        self.conf = AEConfiguration()
        self.device = self.conf.device
        self.modelspath = self.conf.modelspath
        self.modelname = self.conf.modelname

        self.datasetloader = DatasetLoader()
        self.class_to_idx = self.datasetloader.dataset.class_to_idx
        self.classes = self.datasetloader.dataset.classes
        net = torch.load(self.conf.modelspath+self.conf.modelname,
                         map_location=torch.device(self.device))
        self.model = ConvAutoencoderwithClassifier(self.datasetloader.dataset.classes, latent_dim=self.conf.modellatentdim).to(self.device)
        self.model.load_state_dict(net)
        self.model.eval()

    def getClassEntropy(self,imagetensor):
        reconstructed, latent, classes = self.model(imagetensor)
        self.model.zero_grad()
        torch.cuda.empty_cache()
        # Convert to probabilities
        probs = F.softmax(classes, dim=-1)

        # 1. Entropy of the probability distribution
        # Higher entropy = more uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        #print("entropy",entropy)
        #print("len(probs)",probs.shape)
        max_entropy = np.log(probs.shape[1])
        #print("max_entropy",max_entropy)
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    def getClassProbability(self,imagetensor,positiveclassnames,negativeclassnames,softmaxtemperature):

        positiveclassnotes , negativeclassnotes = [],[]
        reconstructed, latent, classes = self.model(imagetensor)
        self.model.zero_grad()
        torch.cuda.empty_cache()
        classes = torch.squeeze(classes, dim=0)
        classes =  F.softmax(classes/softmaxtemperature, dim=-1)
        print("Winner",self.classes[torch.argmax(classes, dim=0).item()])
        for classname in positiveclassnames:
            classidx = self.class_to_idx[classname]
            #print(reconstructed, mu, logvar, classes)
            #print("classes",classes)
            positiveclassnotes.append(classes[classidx])
        if not negativeclassnames == []:
            for classname in negativeclassnames:
                classidx = self.class_to_idx[classname]
                #print(reconstructed, mu, logvar, classes)
                #print("classes",classes)
                negativeclassnotes.append(classes[classidx])
        return positiveclassnotes,negativeclassnotes

    def getReconstructionError(self,imagetensor):
        reconstructed, latent, classes = self.model(imagetensor)
        #error,_,_ = self.model.loss_function(reconstructed,imagetensor,mu, logvar)
        error = F.binary_cross_entropy(reconstructed, imagetensor, reduction='sum')

        return error
