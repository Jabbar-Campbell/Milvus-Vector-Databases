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
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



#  Intially our image looks like this, but lets test this out workflow and inspect our first image,
# the Image module from the Pillow library opens jpegs
from PIL import Image

im = Image.open(file_paths[0])
imp = preprocess(im)



# since our preprocessing converts the image to a tensor we need to convert it back
# to an array the Image module lets us do that. 
# since we create an array that 3, 300, 300  we need to move the axis to 300 300 3
# or length width depth RGB we do this with moveaxis

import numpy as np
import matplotlib.pyplot as plt

imp2 = np.array(imp) 
np.array(imp).shape
imp2 = np.moveaxis(np.array(imp), 0, -1)
plt.imshow(imp2)
plt.show()
Image.open(file_paths[0])


# We have succesfully cropped our image and can move on....
# using out model we should be able to generate embeddings of our images
# we feed tensors to out data
# convert them to arrays
# then flatten them to a single dimensions for milvus
# these arrays will all be the same length thanks to our preprocessing steps!
# a fuction allows us to do this work on each image in an object oriented way



# Function to create embeddings from the model
# .unsqueeze returns a new tensor with a dimension of size one inserted at the specified position


# as a function
def embed(data):
    with torch.no_grad():
        emb = np.array(model(data.unsqueeze(0)))
        return emb.flatten().tolist()




# Test the embedding generation when we open each image and preprocess we can 
# embed each of them
test = file_paths[0]
im = Image.open(test)
im = preprocess(im)
print(im.shape)
emb = embed(im)
print(len(emb))

# a loop might look something like this....
emb_list = []
for i in range(len(file_paths)):
    im = Image.open(file_paths[i])
    im = preprocess(im)
    emb = embed(im)
    emb_list.append(emb)

# This gives us a nice write up on the math under the hood 
# https://towardsdatascience.com/deep-dive-into-vector-databases-by-hand-e9ab71f54f80




######################################################### DEFINITIONS AND  COLLECTION SETUP ########################################################D
###########################################################################################################################################      
# with Field Schema we can intialize the columns of a table
# and the kind of data that will be present

from pymilvus import FieldSchema, DataType

image_name_field = FieldSchema(
   name= "Image_Name",
   dtype= DataType.VARCHAR,
   max_length = 200,
)

image_id_field = FieldSchema(
   name= "Image_Id",
   dtype= DataType.INT64,
   is_primary = True,
)

image_date_field = FieldSchema(
   name= "Image_Date",
   dtype= DataType.VARCHAR,
   dim = 8,
   max_length=10
)


image_vector_field = FieldSchema(
   name= "Image_Vector",
   dtype= DataType.FLOAT_VECTOR,
   dim = 1000,
)





# Now we can define the Collection as a group of Field Schemas  
# on the Field Schema and the collection a name
# its like assigning a table column  names
collection_schema = CollectionSchema(fields=[image_name_field, image_id_field, image_date_field, image_vector_field], description="ALS_IMAGES")




# Now we can create an instance of the above schema
# calling in collection 1
collection_1 = Collection(
    name = "Cohort1",
    schema =  collection_schema,
    using = 'default')



#If we add more collections we can see and inspect  them with
utility.has_collection()
utility.list_collections()
utility.drop_collection()

# we could check if a collection exist and add based on some logic
# adding in some print statements to check the status
try:
    if utility.has_collection("Cohort_2"):
        print(f"Collection '{"Cohort_2"}' already exists.")
    else:
        # Create collection
        collection = Collection(name="Cohort_1", schema=collection_schema, using='default')
        print(f"Collection '{"Cohort_2"}' created successfully.")
except Exception as e:
    print(f"Failed to create collection: {e}")


# now we just have to get the data into our database collection
# Packages like Pytorch which make use of tensors the data is entered in a 
# rowise fashion
# In Milvus our vectors of data will be columnwise. 
# A list of names will be our "Image_name" Column, A sequence of numbers can be our
# "Image_Id", and a list of embeddings will be our "Image_vector" 
# Fortunaltely milvus.insert makes this task simple to add 
# 6 names, 6 ids, 6 dates, and 6 2d vectors


import random
import string




collection_1.insert(data = [
    Files,                      # 6 names
    [1, 2, 3, 4, 5, 6],         # 6 ids
    ['20240714'] * 6,           # 6 dates as strings
    emb_list                    # 6 2D vectors
])

# we can remove entries based on collection schema
collection_1.delete()





# Since the vectors are long and there can be many of them
# we can use various methods in Index our vectors to make
# similarity searches easier and more efficient.
# after all comparing every postion at every dimension is computationalll exhausting 
# first we create an dicitonary of index parameters based on on arguements
# the create.index function will expect
# we then use these parameters on our Vector embedding column

######################################################## INDEXING #####################################################################D
##########################################################################################################################################

# Prepare the index parameters as follows:
# see also https://milvus.io/docs/v2.3.x/build_index.md
# for all the parameter types and definitions for
# example L2 is for euclidean distance but there are others
# once indexed a single value x becomes the vector [x1,...,xn]
# for every value in that column
# Here we use "L2" and "IVF_FLAT" but  see https://milvus.io/docs/v2.0.x/index.md for possible index types
index_params = {
  "metric_type":"L2",        # Euclidean distance
  "index_type":"IVF_FLAT",   # Quantization-based index for high accuracy
  "params":{"nlist":1024}    # Number of cluster units
}

 
 
 
collection_1.create_index(
  field_name="Image_Vector", 
  index_params=index_params
)


