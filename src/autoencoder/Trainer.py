import time

from PIL import Image
import os

from src.autoencoder.ConvAutoEncoderwithClassifier import ConvAutoencoderwithClassifier
from src.autoencoder.ConvVAEwithClassifier import ConvVAEwithClassifier
from src.autoencoder.GradientScaler import GradientScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from src.autoencoder.Configuration import Configuration
from src.autoencoder.ConvAutoEncoder import ConvAutoencoder
from src.autoencoder.ConvVAE import ConvVAE
from src.autoencoder.DatasetLoader import DatasetLoader
from src.autoencoder.LatentSpaceAnalyser import LatentSpaceAnalyzer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
import torch.nn.functional as F
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self):
        self.conf = Configuration()
        self.device = self.conf.device
        self.writer = SummaryWriter()
        self.num_epochs = self.conf.num_epochs
        self.modelspath = self.conf.modelspath
        self.modelname = self.conf.modelname
        #self.model = ConvAutoencoder()
        self.datasetloader = DatasetLoader()
        #self.model = ConvVAEwithClassifier(self.datasetloader.dataset.classes,latent_dim=256)
        self.model = ConvAutoencoderwithClassifier(self.datasetloader.dataset.classes, latent_dim=self.conf.modellatentdim)
        self.gradientscaler = GradientScaler()
        self.LSAnalyser = LatentSpaceAnalyzer(self.model)
        self.istrainingclassifier = True
        self.istrainingautoencoder = True
        self.lr = 0.0001
        # Initialize interactive mode for matplotlib
        plt.ion()


    def saveclusterdiagram(self):
        self.model.eval()
        latents,filenames = self.LSAnalyser.extract_latent_vectors(self.datasetloader.dataloader)
        self.LSAnalyser.visualize_latent_space(latents,filenames,savewithtimestamp=True)
        self.model.train()
    def trainclassifier(self,istraining):
        if istraining == self.istrainingclassifier:
            return
        self.istrainingclassifier = istraining
        print("istrainingclassifier",istraining)
        for param in self.model.classifier.parameters():
            param.requires_grad = istraining
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    def trainautoencoder(self,istraining):
        if istraining == self.istrainingautoencoder:
            return
        self.istrainingautoencoder = istraining
        print("istrainingautoencoder",istraining)
        for param in self.model.encoder_conv.parameters():
            param.requires_grad = istraining
        for param in self.model.fc_mu.parameters():
            param.requires_grad = istraining
        for param in self.model.fc_logvar.parameters():
            param.requires_grad = istraining
        for param in self.model.decoder_linear.parameters():
            param.requires_grad = istraining
        for param in self.model.decoder_conv.parameters():
            param.requires_grad = istraining
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)


    def train(self):
        self.model.to(self.device)  # Move model to device
        self.visualize_reconstruction()
        self.model.train()  # Set model to training mode
        self.criterion = nn.MSELoss()  # Example loss function for autoencoder

        self.classlossfunc = nn.BCEWithLogitsLoss()
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        # Optional: Add learning rate scheduler for even more stability
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            min_lr=1e-10
        )
        for epoch in range(self.num_epochs):
            #self.istrainingclassifier = epoch > 10
            trainingswitch = 100
            #self.trainclassifier(epoch > trainingswitch)
            #self.trainautoencoder(not (epoch > trainingswitch))
            # if epoch == 11:
            #     self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0001)
            #
            # if self.istrainingclassifier:
            #     for param in self.model.classifier.parameters():
            #         param.requires_grad = True
            # else:
            #     for param in self.model.classifier.parameters():
            #         param.requires_grad = False
            running_loss = 0.0
            for i, (images, target, filename, islabeled) in enumerate(
                    self.datasetloader.dataloader):  # Assuming the data is a tuple of (images, labels)
                images = images.to(self.device)
                target = target.to(self.device)
                islabeled = islabeled.to(self.device)
                #print((images.shape, target, filename, islabeled))
                labeled = images[islabeled]
                if labeled.shape[0] == 0 and self.istrainingclassifier and not self.istrainingautoencoder:
                    continue
                unlabeled = images[~islabeled]
                #print("unlabeled.shape",unlabeled.shape)
                labeledtarget = target[islabeled]
                islabeledlist = list(islabeled)
                labeledfilename = [ filename[i]  for i in range(len(islabeledlist)) if islabeledlist[i]]

                #labeledfilename.remove([])
                unlabeledfilename = [ filename[i]  for i in range(len(islabeledlist)) if not islabeledlist[i]]

                # print("labeled",labeled.shape)
                # print("unlabeled", unlabeled.shape)
                # print("labeledtarget", labeledtarget)
                # print("labeledfilename", labeledfilename)
                # print("unlabeledfilename", unlabeledfilename)
                # Forward pass
                #reconstructed, latent = self.model(images)
                #loss = self.criterion(reconstructed, images)  # Compare reconstructed images to input

                # Backward pass and optimization
                #self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()
                # VAE !
                # In training loop
                totalreconloss,recon_loss,kl_loss = torch.zeros(1).to(self.device),torch.zeros(1).to(self.device),torch.zeros(1).to(self.device)
                if unlabeled.shape[0] > 0 and self.istrainingautoencoder:
                    reconstructed, latent,logits = self.model(unlabeled)
                    totalreconlossunlabeled = F.binary_cross_entropy(reconstructed, unlabeled, reduction='sum')
                    # totalreconlossunlabeled, recon_lossunlabeled, kl_lossunlabeled = self.model.loss_function(reconstructed, unlabeled, mu, logvar)
                    # totalreconlossunlabeled, recon_lossunlabeled, kl_lossunlabeled = totalreconlossunlabeled* unlabeled.shape[0], recon_lossunlabeled* unlabeled.shape[0], kl_lossunlabeled * unlabeled.shape[0]
                    totalreconloss += totalreconlossunlabeled
                    # recon_loss +=recon_lossunlabeled
                    # kl_loss += kl_lossunlabeled

                if labeled.shape[0] > 0 and self.istrainingclassifier:
                    reconstructed, latent,logits = self.model(labeled)
                    totalreconlosslabeled = F.binary_cross_entropy(reconstructed, labeled, reduction='sum')
                    # totalreconlosslabeled, recon_losslabeled, kl_losslabeled = self.model.loss_function(reconstructed, labeled, mu, logvar)
                    # totalreconlosslabeled, recon_losslabeled, kl_losslabeled = totalreconlosslabeled * labeled.shape[0], recon_losslabeled * labeled.shape[0], kl_losslabeled * labeled.shape[0]
                    totalreconloss += totalreconlosslabeled
                    # recon_loss += recon_losslabeled
                    # kl_loss += kl_losslabeled
                    # Classification loss only for labeled data
                    #class_loss =  F.cross_entropy(logits, labeledtarget)
                    #print(logits)
                    #F.one_hot(targets, num_classes=num_classes).float()
                    #class_loss = self.classlossfunc(logits, labeledtarget)
                    class_loss = self.classlossfunc(logits, F.one_hot(labeledtarget, num_classes=self.model.classcount).float())
                else:
                    class_loss = torch.zeros(1).to(self.device)
                #scaledreconloss,scaledcalssloss = self.gradientscaler.update(totalreconloss,class_loss)
                #print("scaledreconloss,scaledcalssloss",scaledreconloss.item(),scaledcalssloss.item())
                #loss = scaledreconloss + scaledcalssloss
                # minimum alpha 10000
                alpha = 100000
                loss = totalreconloss/(labeled.shape[0]+unlabeled.shape[0]) + class_loss * alpha
                # tensorboard --logdir=runs
                self.writer.add_scalar("reconloss", totalreconloss/(labeled.shape[0]+unlabeled.shape[0]), time.time())
                self.writer.add_scalar("class_loss", class_loss,
                                       time.time())
                try:
                    self.optimizer.zero_grad()
                except:
                    print("labeled",labeled.shape)
                    print("unlabeled", unlabeled.shape)
                    print("labeledtarget", labeledtarget)
                    print("labeledfilename", labeledfilename)
                    print("unlabeledfilename", unlabeledfilename)
                    print("loss", loss, "totalreconloss", totalreconloss, "class_loss", class_loss * alpha)

                    #print("loss",loss,"scaledreconloss",scaledreconloss, "scaledcalssloss",scaledcalssloss)

                #print("loss",loss)

                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                running_loss += loss.item()

                if (i + 1) % 10 == 0:  # Print every 10 batches
                    print(
                        f"Epoch [{epoch + 1}/{self.num_epochs}], Batch [{i + 1}/{len(self.datasetloader.dataloader)}], Loss: {loss.item():.4f}")
                    print("loss", loss.item(), "totalreconloss", totalreconloss.item(), "class_loss",
                          class_loss.item() * alpha)
                    print("Scheduler learning rate",scheduler.get_last_lr())

            # Print average loss at the end of each epoch
            epoch_loss = running_loss / len(self.datasetloader.dataloader)
            scheduler.step(epoch_loss)
            self.writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch + 1)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Loss: {epoch_loss:.4f}")
            self.writer.add_scalar("Loss/train", epoch_loss, epoch+1)
            if epoch % 2 == 0 :
                self.save_model()
                self.visualize_reconstruction()
                #self.saveclusterdiagram()

            if (epoch+1) % 10 == 0:
                self.datasetloader.loadDataset()

    def save_model(self):
        """Save the model to the specified path."""
        torch.save(self.model.state_dict(), self.modelspath + self.modelname)
        print(f"Model saved to { self.modelspath + self.modelname}")

    def visualize_reconstruction(self, num_images=5):
        """Visualizes original vs. reconstructed images."""
        self.model.eval()  # Set the model to evaluation mode
        data_iter = iter(self.datasetloader.dataloader)

        # Select a few random images from the dataset
        images, _, filename, _ = next(data_iter)  # Get a single batch
        indices = random.sample(range(images.size(0)), num_images)
        images = images[indices].to(self.device)

        # Run through the model to get reconstructions
        with torch.no_grad():
            reconstructed, latent , logits = self.model(images)
        print(reconstructed.shape,logits.shape, images.shape)
        imageloss = []
        for i in range(num_images):
            loss = F.binary_cross_entropy(reconstructed[i,:,:,:],  images[i,:,:,:], reduction='sum')

            #loss = self.model.loss_function(reconstructed[i,:,:,:], images[i,:,:,:], mu[i,:], logvar[i,:])
            imageloss.append([loss])
        # Move images and reconstructed images to CPU for plotting
        images = images.cpu()
        reconstructed = reconstructed.cpu()
        latent = latent.cpu()
        plt.figure(1,figsize=(12, 7))  # Use figure 1 to keep reusing the same window
        plt.clf()  # Clear the figure to prevent overlay
        text_kwargs = dict(ha='center', va='bottom', fontsize=14, color='C1')
        # Plot the original and reconstructed images side by side
        for i in range(num_images):
            # Original image
            plt.subplot(3, num_images, i + 1)
            plt.imshow(images[i].squeeze(), cmap="gray")
            _ , choosensclass = torch.topk(logits[i],3)

            plt.title(str(self.model.classes[choosensclass[0]])+"\n"+str(self.model.classes[choosensclass[1]])+"\n"+str(self.model.classes[choosensclass[2]]))
            plt.axis("off")

            # Reconstructed image
            plt.subplot(3, num_images, num_images + i + 1  )
            plt.imshow(reconstructed[i].squeeze(), cmap="gray")
            plt.title(str(imageloss[i][0].item()),fontsize=7)

            # Reconstructed image
            plt.subplot(3, num_images, num_images*2 + i + 1)
            #muandlogvar = torch.vstack((mu[i].cpu().unsqueeze(0)* torch.ones((70,1)),logvar[i].cpu().unsqueeze(0)* torch.ones((70,1))))# * torch.ones((70,2))
            #latentvariable = (latent[i].cpu().unsqueeze(0)* torch.ones((70,1))
            #print(muandlogvar)
            #exit()
            #plt.imshow(latentvariable, cmap="gray")
            plt.title(str(imageloss[i][0].item()), fontsize=7)
            # #plt.text(1, 1, 'colored text in axes coords',
            #         verticalalignment='bottom', horizontalalignment='right',
            #         color='green', fontsize=15)
            plt.axis("off")

        plt.pause(0.001)  # Briefly pause to update the figure without blocking
        self.model.train()


train = Trainer()
train.train()