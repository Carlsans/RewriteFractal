import socket

import torch

from src.autoencoder.ConvAutoEncoder import ConvAutoencoder
from src.autoencoder.ConvAutoEncoderwithClassifier import ConvAutoencoderwithClassifier
#from src.autoencoder.DatasetLoader import DatasetLoader


#from src.autoencoder.DatasetLoader import DatasetLoader


class Configuration:
    def __init__(self):
        print("Machine name",socket.gethostname())
        if socket.gethostname()== 'arch-laptop':
            self.datasetfolder = "/home/carl/DeepLearning/fractalimagesdataset/"
            #self.modelspath = "/home/carl/PycharmProjects/FractalAutoencoder/models/"
            self.modelspath =  "/home/carl/PycharmProjects/rewriteFractals/src/model/"
            self.classificationfolder = "/home/carl/DeepLearning/classificationfolder/"
        if socket.gethostname()== 'ordi':
            self.datasetfolder = "/home/carl/PycharmProjects/graphfractals/GraphFractals/lowres/"
            #self.datasetfolder = "/home/carl/PycharmProjects/graphfractals/GraphFractals/"
            self.modelspath = "/home/carl/PycharmProjects/FractalAutoencoder/model/"
            self.classificationfolder = "/home/carl/PycharmProjects/FractalAutoencoder/classificationfolder/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        print("Running on",self.device)
        self.num_epochs = 1000000
        self.modelname = 'autoencodervae2.pth'
        self.modellatentdim = 128





