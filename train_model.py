import torch
import torchvision
import numpy as np
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# from model import VGG16
from tqdm import tqdm
from collections import OrderedDict
import json
import os
from ImageNet import ImageNet
from ImageNet import ImageNetFull
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

class Trainer():
    def __init__(self,data_dir,batch_size=128,type="mini"):
        super().__init__()
        self.batch_size=batch_size
        traindir = os.path.join(data_dir,'train')
        valdir = os.path.join(data_dir,'val')
        
        self.transformTrain = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std  = [ 0.229, 0.224, 0.225 ]),
        ]) 
        
        self.transformVal = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std  = [ 0.229, 0.224, 0.225 ]),
        ])
        if type=="mini":
            self.trainData = ImageNet(traindir, self.transformTrain)
            self.trainLoader = torch.utils.data.DataLoader(dataset=self.trainData, batch_size=batch_size, shuffle=True, num_workers=4)
            
            self.valdata = ImageNet(valdir, self.transformVal)
            self.valLoader = torch.utils.data.DataLoader(dataset=self.valdata, batch_size=batch_size, shuffle=False, num_workers=4)
            
        elif type=="full":
            self.trainData = ImageNetFull(traindir,"train", self.transformTrain)
            self.trainLoader = torch.utils.data.DataLoader(dataset=self.trainData, batch_size=batch_size, shuffle=True, num_workers=4)
            
            self.valdata = ImageNetFull(valdir,"val", self.transformVal)
            self.valLoader = torch.utils.data.DataLoader(dataset=self.valdata, batch_size=batch_size, shuffle=False, num_workers=4)
        
    def train(self,layer_name,model,lr=0.001,epoch=1,rate=[1]*13,lr_cyclic=False):
        # torch.backends.cudnn.benchmark = True
        BATCH_SIZE = self.batch_size
        LEARNING_RATE = lr
        EPOCH = epoch

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model = VGG16(rate = rate)
        # wts = torch.load("new_wts.pth")

        # model.load_state_dict(wts)
        if(lr_cyclic):
            torch.save(model.state_dict(),"Before12epoch.pth")
        model.to(device)

        cost = tnn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE,momentum=0.9,weight_decay=0.0005)
        scheduler= None

        scaler = GradScaler(enabled=False)
        if(lr_cyclic):
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.00001, 0.001)

        model.train()

        step = 0
        dir_path = f'runs/{layer_name}'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        writer = SummaryWriter(dir_path)
        for epoch in range(EPOCH):

            avg_loss = 0

            loop = tqdm(enumerate(self.trainLoader), total=len(self.trainLoader), leave=False)
            for batch_idx,(images, labels) in loop:
                images = images.to(device)
                labels = labels.to(device)

                #forward
                with autocast(enabled=False):
                    output = model(images)
                    loss = cost(output, labels)
                    avg_loss+= loss.data

                    acc1, acc5 = self.accuracy(output, labels, topk=(1, 5))
                    acc1/=labels.size(0)
                    acc5/=labels.size(0)

                    #tensorboard logs
                    writer.add_scalar("Training Accuracy Top1",acc1,global_step=step)
                    writer.add_scalar("Training Accuracy Top5",acc5,global_step=step)
                    writer.add_scalar("Training Loss",loss,global_step=step)
                    step+=1

                #backward
                # loss.backward()
                scaler.scale(loss).backward()

                #SGD Step
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                #Scheduler Step
                if(lr_cyclic):
                    scheduler.step()
                optimizer.zero_grad()

                loop.set_description(f"Epoch {epoch+1}/{EPOCH}")
                loop.set_postfix(loss=loss.item())


        torch.save(model.state_dict(),"new_wts.pth")


    def test(self,layer_name,model,rate=[1]*13, lr=0.001):

        BATCH_SIZE = self.batch_size
        LEARNING_RATE = lr
      
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model = VGG16(rate = rate)
        # wts = torch.load("new_wts.pth")
        # model.load_state_dict(wts)

        model.to(device)
        model.eval()

        optimizer = torch.optim.SGD(model.parameters(),lr = LEARNING_RATE)

        dir_path = f'runs/{layer_name}' + 'val'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        writer = SummaryWriter(dir_path)
        step=0
        correct1 = 0
        correct5 = 0
        total = 0
        loop = tqdm(self.valLoader, leave = False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(images)
                c1, c5 = self.accuracy(outputs, labels, topk=(1, 5))
                batch_size = labels.size(0)
                total+=batch_size
                writer.add_scalar("Training Accuracy Top1",c1/batch_size,global_step=step)
                writer.add_scalar("Training Accuracy Top5",c5/batch_size,global_step=step)
                step+=1
                correct1+=c1
                correct5+=c5
                acc1 = np.round(correct1/total,2)
                acc5 = np.round(correct5/total,2)

            loop.set_description(f"Avg Top1 {acc1} Top5 {acc5}")

    #     tqdm.write(f"Top1 = {acc1} Top5 = {acc5}")

    def accuracy(self,output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.cpu().numpy())
            return res