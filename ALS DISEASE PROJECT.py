# In this Project the goal is  create a database of images and then search thru those images using a query image using MILVUS
# Milvus is a non structured database unlike sql and uses vectors in n Dimensional space as opposed to searching thru tables


# Since we are on a windows pc an milvus runs in a linux enviroment we'll first need to download and install Docker. This will allow
# us to run separate environments as containers. My machine needed   Virtulization to be enabled in the BIOS config file in order for 
# Docker to work. Once this was done I could make use of the stand alone Milvus installation. (theres one for cloud infrastructure as well)

# wget is a common linux command for downloading material so I had to install that first.  By  adding it my PATH variable file  I could run 
# wget from any directory

# Next I downloaded the milvus-standalone-docker-compose.yml from the Milvus website and saved it as docker-compose.yml. I changed
# directories to something I could work out of and typed the following.....
# "wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml" 
# In the directory that holds the docker-compose.yml file  
# for me this was "C:\Users\jabba\OneDrive\Desktop\Sandbox\Milvus"

# start Milvus by running: 
# "sudo docker compose up -d"


# To find out which local port Milvus is running....
# $ docker port milvus-standalone 19530/tcp
#
# ....and to shut it down. 
# sudo docker compose down

# With Milvus running in a Docker Container I then created an python environment in VSCODE that holds all the pacakges required for this
# project. Once created I ran the activate file located within to turn it on. In my case it was located  at 
# C:\Users\jabba\milvus_env\Scripts\Activate.ps1
# Once up and running I then made sure my python interpretor was using this newly activated envirnoment. (reflected in the lower right
# of the vscode editor)
 
# we now have TWO "environments"!!. One is Docker that is managing a MILVUS database. The other is a Python Environment that will connect to it. 
# Your VScode terminal command line should say  "(milvus_env) PS C:\your\home\directory"
# Since our Python env is mostly empty except for the basic python packages we'll need to install things like pymilvus so that we can connect 
# to milvus. Vscode also will then know what we're talking about when we start using commands from this package. At the Terminal of 
# our activated environment type ...... 
# $ "pip3 install pymilvus protobuf grpcio-tools jupyterlab torchvision"

# if you're like me you may  have "pip3 ssl" issues  where pip cant find find ssl certificates. if so use...
# $ "pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pymilvus protobuf grpcio-tools jupyterlab"



# With our Python env populated with the right packages and a Docker container running Milvus
# Lets connecting to our Milvus Database

import pymilvus


from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

######################################################### CONNECT TO MILVUS  ########################################################D
###########################################################################################################################################    
# Connect to existing milvus server
connections.connect(host = '127.0.0.1',port = 19530)


# We'll also need some images and the path to them. OS is a great package for making path variables

######################################################### LOCATE IMAGE PATHS  ########################################################D
###########################################################################################################################################    
import os

Path = os.path.join(os.getcwd(), "ALS images")
Files = os.listdir(path=Path)

file_paths = []
for i in Files:
    x = os.path.join(Path, i)
    file_paths.append(x)






# now that we have some test images milvus only stores vectorized information. In order
# so our goal should be to vectorize these images.  This is done with an embedding process
# there are many algorithems that do this. Here package management can be an issue since we'll 
# have to pip install a suitable version  of which there are many
# for a comphrensive list see https://pypi.org/search/?q=torchvision
# to start lets make use of the ones readily available from pytorch/vision repo callend "resnet18"
# we can experiment with others later. Since we dont have pytorch in our envirnoment we need to install it

# remembering our SSL issues type the following......
# "pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org torch==1.3.1+cpu torchvision"

#random.shuffle(paths)




# Here we load an embedding model from the pytorch/vision repo called "resnet18". This is a a Convolutional Neural Network  
# that has been trained on many images (probably not pathology images). If you have a basic understanding of neural 
# networks, its no surprise that Convolutional Networks are best for image processing tasks. You might also remember 
# that the weights between nodes must be optimized using gradient descent on test images  in a process call back propagation.
# Here we Set the pretrained argumnet to true to  make use of those weights. 

# Inspecting the archithecture of this network we can see the number of nodes in each layer. The algorithem used in back propogation, pooling layer
# settings and more. For an indepth review watch....
# https://www.youtube.com/watch?v=nc7FzLiB_AY&t=103s

# The first argument is the repo name, the second argument is the model name
# The third argument is an optional parameter to specify the model version

import torch
from torchvision import transforms


model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()


# well need to string some preprocessing procedures together to perform on each image
# Tranform.compose is a nice way to accomplish this and save that in something we'll call
# preprocess

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



#  Intially our image looks like this, but lets test this out workflow and inspect our first image,
# the Image module from the Pillow library opens jpegs
from PIL import Image
Image.open(file_paths[0])
im = Image.open(file_paths[0])
imp = preprocess(im)



# since our preprocessing converts the image to a tensor we need to convert it back
# to an array the Image module lets us do that. 
# since we create an array that 3, 300, 300  we need to move the axis to 300 300 3
# or length width depth RGB we do this with moveaxis

import numpy as np
import matplotlib.pyplot as plt

imp = np.array(imp) 
np.array(imp).shape
imp = np.moveaxis(np.array(imp), 0, -1)
plt.imshow(imp)
plt.show()



# We have succesfully cropped our image and can move on....



# Function to create embeddings from the model
def embed(data):
    with torch.no_grad():
        emb = np.array(model(data.unsqueeze(0)))
        return emb.flatten().tolist()



# Test the embedding generation 
test = 'animals/raw-img/cane/OIP--2z_zAuTMzgYM_KynUl9CQHaE7.jpeg'
im = Image.open(test)
im = preprocess(im)
print(im.shape)
emb = embed(im)
print(len(emb))



# This gives us a nice write up on the math under the hood 

# https://towardsdatascience.com/deep-dive-into-vector-databases-by-hand-e9ab71f54f80