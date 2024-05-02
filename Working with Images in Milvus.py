#!/usr/bin/env python
# coding: utf-8

# ######################################################### Image search using PyTorch and Milvus #################################################################
# 
# In this example, we will perform image similarity search using PyTorch and Milvus. 
# 
# We are going to use the Animals-10 dataset available in Kaggle. Download and extract the compressed archive containing the images. 
# 
# https://www.kaggle.com/datasets/alessiocorrado99/animals10
# 
# We shall make use of pre-trained Inception model to generate the vector embeddings from the images and use them for our similarity search
# The process is as follows
# 1) get images and format size and intensities
# 2) Get a model for emeddings from pytorch
#    ---------------retrieve embedding size
# 3) Generate embeddings of images using model
# 4) Format embeddings into flat list of vectors with same embedding size
# 5) Create Milvus framework to store embeddings and file paths
# 6) Add embedding and file path records
# 
# 7) Embed query image using the same model
# 8) Search our Milvus Database for similar images 
# 9) Map file path and plot that image to verify
######################################################################################################################################################################

import torch
import glob
from torchvision import transforms
from PIL import Image
#!pip3 install --upgrade pymilvus
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import connections
from getpass import getpass
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pickle
import random



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



# Configs
COLLECTION_NAME = 'SIM_SEARCH_TORCH'  # Collection name
DIMENSION = 1000  # Embedding vector size in this example
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# Inference Arguments used for index_param()
BATCH_SIZE = 128
TOP_K = 3
img_limit = 2000


# connect to Milvus database
connections.connect(
  alias="default",
  host='localhost',
  port='19530',
  # user='root',
  # password=getpass('Milvus Password: ')
)


# Milvus
# Drop the old collection to start fresh
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)



filepath_field = FieldSchema(name='filepath', dtype=DataType.VARCHAR,is_primary=True, max_length=4000)
embedding_field = FieldSchema(name='inception_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)

fields = [filepath_field, embedding_field]

# Create collection schema
schema = CollectionSchema(fields=fields)

# Create collection
collection = Collection(
    name=COLLECTION_NAME,
    schema=schema,
    using='default')
utility.list_collections()



data_batch = [[],[]]

for ind, path in enumerate(paths):
    im = Image.open(path).convert('RGB')
    im = preprocess(im)
    embedding = embed(im)
    data_batch[0].append(path)
    data_batch[1].append(embedding)
    # print([[path], [embedding]])

    if ind%100==0 and ind>0:
        print(f'Completed {ind} of {len(paths)} images')

    if ind==img_limit:
        break

        
print(f'Completed all the images')


# Pickle the data
pickle_file = open('img_embeddings.pkl', 'wb')
pickle.dump(data_batch, pickle_file)
pickle_file.close()

 

# insert the data in batches
with open('img_embeddings.pkl', 'rb') as handle:
    data_batch = pickle.load(handle)

tmp_batch = [[], []]
insert_bath_size = 1000
for x in range(len(data_batch[0])):
    tmp_batch[0].append(data_batch[0][x])
    tmp_batch[1].append(data_batch[1][x])

    if x>0 and x%insert_bath_size==0:
        collection.insert(tmp_batch)
        tmp_batch = [[], []]
        print(f'Inserted the batch {int(x/insert_bath_size)} to Milvus collection with insert batch size of {insert_bath_size}')
        
if tmp_batch[0]:
    collection.insert(tmp_batch)

# collection.flush()
# collection.compact()
print(f'Flushed the data to Milvus')

 

print(type(data_batch[1][x][0]))

 

# Vectorize and Create an index for collection. 
# Drop any old remnant index with the same name.
collection.drop_index(index_name="IVF_FLAT_INDX_IMG_SEARCH")

index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024},
  "index_name": "IVF_FLAT_INDX_IMG_SEARCH"
}

collection.create_index(field_name="inception_embedding", index_params=index_params)



# Test using an image 
test = 'sheep.jpg'
im = Image.open(test)
im = preprocess(im)
search_embedding = embed(im)


# Load the collection to search
# collection.flush()
# collection.compact()
# collection.release()
collection.load(replica_number=1)



# Search for similar images in our collection
search_res = collection.search(data=[search_embedding], anns_field='inception_embedding', param={'nprobe': 128, 'metric_type': 'L2',}, limit=5, output_fields=['filepath'])

# View output model and Path
plt.figure()
f, axarr = plt.subplots(6, 1, figsize=(32, 32))
axarr[0].imshow(Image.open(test).resize((512, 512),  Image.Resampling.LANCZOS))
axarr[0].set_axis_off()
axarr[0].set_title('Query Image')

for indx, result in enumerate(search_res[0]):
    axarr[indx+1].set_title('Distance: ' + str(result.distance))
    axarr[indx+1].imshow(Image.open(result.entity.get('filepath')).resize((512, 512),  Image.Resampling.LANCZOS))
    axarr[indx+1].set_axis_off()

plt.show()


for ind, path in enumerate(paths):
    print(path)



print(model)




