import io

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import sys

from src.Configuration import Configuration
from src.autoencoder.Inference import Inference
class Sieve:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("new sieve")
        self.inferer = Inference()
    def isbailouttensorvalid(self,tensor):
        if not torch.min(tensor) == torch.max(tensor):
            return True

        return False
    def getimageinformationsnote(self,tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.cpu()
        result = self.save_image_to_memory_and_measure(np.uint8(255 * tensor))
        return result

    def getimageautoencoderclassentropy(self,tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.to(self.device)
            max, min = torch.max(tensor), torch.min(tensor)
            if max == min:
                print("Fractal with only one value")
                return None
            tensor = (tensor - min) / (max - min)
            normalizedentropy = self.inferer.getClassEntropy(torch.reshape(tensor,(1,1,512,512)))
            #print("normalizedentropy",normalizedentropy)
            return normalizedentropy
        return 0

    def getimageautoencoderclasscorrespondance(self,tensor,positiveclassnames,negativeclassnames =[],positiveweights = None,negativeweights = None,softmaxtemperature=1):
        if positiveweights is None:
            positiveweights = [1 for i in range(len(positiveclassnames))]
        if negativeweights is None:
            if negativeclassnames == []:
                negativeweights = []
            else:
                negativeweights = [1 for i in range(len(negativeclassnames))]
        if not (len(positiveweights) == len(positiveclassnames) and len(negativeweights) == len(negativeclassnames) ):
            print("getimageautoencoderclasscorrespondance require (len(positiveweights) == len(positiveclassnames) and len(negativeweights) == len(negativeclassnames)")
        #print(positiveclassnames,negativeclassnames,positiveweights,negativeweights)

        if torch.is_tensor(tensor):
            tensor = tensor.to(self.device)
            max, min = torch.max(tensor), torch.min(tensor)
            if max == min:
                print("Fractal with only one value")
                return None
            tensor = (tensor - min) / (max - min)
            #print('tensor.shape',tensor.shape)
            positiveloss,negativeloss = self.inferer.getClassProbability(torch.reshape(tensor,(1,1,512,512)),positiveclassnames,negativeclassnames,softmaxtemperature)
            positiveloss = [positiveweights[i] * torch.relu(positiveloss[i]).item() for i in range(len(positiveloss))]
            negativeloss = [negativeweights[i] * torch.relu(negativeloss[i]).item() for i in range(len(negativeloss))]
            #print("positiveloss,negativeloss",positiveloss,negativeloss)
        else:
            print("Fractal is not a tensor...")
        return torch.sum(torch.tensor(positiveloss)),torch.sum(torch.tensor(negativeloss))


    def getimageautoencoderreconstructionerror(self,tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.to(self.device)
            max, min = torch.max(tensor), torch.min(tensor)
            if max == min:
                return None
            tensor = (tensor - min) / (max - min)
            tensor = torch.reshape(tensor,(1,1,512,512))
            #print("tensor.sum",tensor.sum())
            return self.inferer.getReconstructionError(tensor).item()
        else:
            return None

    def comparetootherimagenote(self,tensor):
        if torch.is_tensor(tensor):
            max,min = torch.max(tensor),torch.min(tensor)
            if max == min:
                return None
            tensor = (tensor - min)/(max-min)

            tensor = tensor.cpu()
        img_path = 'src/baceinsteinfullpng.png'
        img = Image.open(img_path).convert("L")

        # define custom transform function
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(tensor.shape)
        ])

        # transform the pIL image to tensor
        # image
        img_tr = torch.squeeze(transform(img),dim=0)
        #img_tr =  (img_tr - torch.min(img_tr))/(torch.max(img_tr)-torch.min(img_tr))
        #print("img_tr",img_tr.shape)
        difference = img_tr-tensor
        #print("difference",difference)
        #absdifference = torch.linalg.norm(difference, ord=1)
        absdifference = difference.flatten().pow(2).sum().sqrt()
        #print("absdifference",absdifference)

        #mean = torch.sum(absdifference)/(512*512)

        return absdifference / img_tr.shape[0]


    def save_image_to_memory_and_measure(self,array_data):
        #print("array_data",array_data)
        if array_data.sum() == 0:
            #print("array_data sum 0")
            return None
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(array_data)

        # Create a binary stream to hold the image in memory
        memory_buffer = io.BytesIO()

        # Save the image to the memory buffer in PNG format
        pil_image.save(memory_buffer, format='PNG')

        # Get the size of the image in memory
        memory_size = memory_buffer.tell()

        # Get image dimensions
        image_dimensions = pil_image.size

        # Calculate size in different units
        size_kb = memory_size / 1024
        size_mb = size_kb / 1024
        return memory_size/(image_dimensions[0]*image_dimensions[1])
