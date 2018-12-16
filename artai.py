import glob
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw
import warnings
import torch.optim as optim
from torchvision import transforms

from torch.autograd import Variable
from skimage import io#, transform


from torch.utils.data.sampler import SubsetRandomSampler


class ConvNet(nn.Module):

    #Classifying RGB images, therefore number of input channels = 3
    #We want to apply 32 feature detectors (filters), so out channels is 32
    #3x3 filter moves 1 pixel at a time
    #ReLU" all negative values become 0, all positive values remain


    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3_drop = nn.Dropout2d(0.5)


        self.fc1 = torch.nn.Linear(32*32*8, 64)
        self.fc2 = torch.nn.Linear(64, 4)
        torch.nn.init.xavier_uniform(self.conv1.weight) #initialize weights
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        


    def forward(self, x):
        x = F.relu(self.conv1(x.cuda()))
        x = self.pool1(x)
        #print('Conv1 layer: X shape:',x.shape)
        x = F.relu(self.conv2(x.cuda()))
        x = self.pool2(x)
        #print('Conv2 layer: X shape:',x.shape)        
        x = F.relu(self.conv3(x.cuda()))
        x = self.pool3(x)
        #print('Conv3 layer: X shape:',x.shape)    
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0),-1)   #Rectify 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x)




class DataSetCatsDogs(Dataset):
    #Cats and Dogs *.jpg Sataset. 1000 images will be taken for each category
    
    def __init__(self, root_dir,transform): #download,read,transform the data
        self.root_dir = root_dir
        self.class_list = ('drawings','iconography','painting','sculpture')

        self.transform = transform

    def __getitem__(self, index): #superfast 0(1) method, return item by index
        #Goes into the folder with the database
        #According to the class list specified in the constructor goes into every folder 
        #and loads all the images in the folder. The line "img ="
        #varies to every database
        #file_number = index
        file_number = int(index / len(self.class_list))
        folder_number = index % len(self.class_list)
        class_folder = os.path.join(self.root_dir, self.class_list[folder_number])
        for filepath in glob.iglob(class_folder):
            img = filepath + '/' + self.class_list[folder_number] + '.' + str(file_number).zfill(4) + '.jpg'
        
        
        label = folder_number


        sample = Image.open(img)
        sample = sample.convert('RGB')
        sample = sample.resize((64,64)) #Resizing images to universal size
        return self.transform(sample), label     #first position is data, second is labels, both should be tensor

    def __len__(self): #return data length 

        return 1000*4


def main():

    warnings.filterwarnings("ignore")  #not to dlood the output
    torch.set_printoptions(precision=10)   #to get a nice output
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #working on cuda, not on the CPU

    dtype=torch.cuda.FloatTensor

    train_transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #normalize the data         

    db = DataSetCatsDogs('dataset_updated/training_set',train_transformer) #initiate DataBase
    train_loader = DataLoader(dataset = db, batch_size=64, shuffle=True,num_workers=2)

    cnn = ConvNet() #Create the instanse of net 

    cnn = cnn.cuda()


    criterion = torch.nn.CrossEntropyLoss().cuda() #Cross Entropy Loss
    optimizer = optim.Adam(cnn.parameters(), lr=0.001) #Optimizer with learning rate 0.001
    running_loss = 0 
    total_train_loss = 0
    for epoch in range(38):  #32 it was
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs.type(dtype)), Variable(labels.type(torch.cuda.LongTensor))


            optimizer.zero_grad()             #Set the parameter gradients to zero
            outputs = cnn(inputs)
            loss_size = criterion(outputs, labels)
            loss_size.backward()
            optimizer.step()   
            running_loss += loss_size.data[0]
        print('Running loss was:',running_loss)
        print('Finished Epoch ',epoch)
        total_train_loss += loss_size.data[0]

    cnn.eval()
    torch.save(cnn, 'artAI.pt')


 
    print('Done.')




if __name__ == "__main__":
   main()

