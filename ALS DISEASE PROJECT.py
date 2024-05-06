# In this Project the goal is  create a data base of images and then search thru those images using a query image using MILVUS
# Milvus is a non structured database unlike sql and uses vectors in n Dimensional space as opposed to searching thru tables


# Since we are on a windows pc an milvus runs in a linux enviroment we'll first need to download and install Docker. This will allow
# us to run separate environments as containers. My machine need the Virtulization to be enabled in the BIOS confif file in order for 
# Docker to work. Once this was done I could make use of the stand alone Milvus installation. 
# wget is a linux command so I had to install that first and add it my PATH variable file so I could run wget from any directory
# Next I downloaded the milvus-standalone-docker-compose.yml from the Milvus website and saved it as docker-compose.yml. I changed
# directories to something I could work out of and typed the following.....
#  "wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml"

# In the directory that holds docker-compose.yml, 
# cd "C:\Users\jabba\OneDrive\Desktop\Sandbox\Milvus"

# start Milvus by running: 
# "sudo docker compose up -d"


# To find out which local port Milvus is running....
# $ docker port milvus-standalone 19530/tcp
#
# ....and to shut it down. 
# sudo docker compose down

# With Milvus running in a Docker Container I then created an python environment in VSCODE that holds all the pacakges required for this
# project. Once created a ran the activate file located within to turn it onOnce up and running I then instructed my python interpretor 
# to use this newly activated envirnoment. we now have TWO environoment. One is Docker that is managing a MILVUS database. The other 
# is a Python Environment that will connect to it. Your VScode terminal command line should say  "(milvus_env) PS C:\your\directory"
# our Python env is mostly empty except for the basic python packages we'll need to install things like pymilvus so that we can connect 
# to milvus. Vscode also will then know what we're talking about when we start using commands from this package. At the Terminal of 
# our activated environment type ...... 
# $ "pip3 install pymilvus protobuf grpcio-tools jupyterlab"
# if your like me you may  have "pip3 ssl" issues  where pip cant find find ssl certificates. if so use...
# $ "pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pymilvus protobuf grpcio-tools jupyterlab"



# With our Python env populated with the right packages and a Docker container running Milvus
# Lets connecting to our Milvus Database
import pymilvus


from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

######################################################### CONNECT TO MILVUS  ########################################################D
###########################################################################################################################################    
# Connect to existing milvus server
connections.connect(host = '127.0.0.1',port = 19530)



# lets get ourselves some images and the paths that lead to them
# we wrote to target ALS but lets load some test image while we wait
# 















# Get the filepaths of the images
paths = glob.glob('animals/raw-img/*/*.j*', recursive=True)
random.shuffle(paths)


# Load the embedding model from the tensorflow hub with the last layer removed
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()

# Preprocessing for images
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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

