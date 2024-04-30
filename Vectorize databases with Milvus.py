## THIS SCRIPT IS FOR THE USE OF A MILVIS VECTORIZED DATABASE IT WILL MAKE USE OF THE PACKAGE "pymilvus"
### BE SURE TO HAVE A MILVUS CONTAINER RUNNING AT THE COMMAND LINE SET UP BY  FIRST INSTALLING MILVUS


# Download milvus-standalone-docker-compose.yml and save it as docker-compose.yml manually, or with the following command.
#  $ wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# In the directory that holds docker-compose.yml, start Milvus by running:
# $ sudo docker compose up -d


# Now check if the containers are up and running.
# $ sudo docker compose ps


# Verify which local port the Milvus server is listening on. Replace the container name with your own.
# $ docker port milvus-standalone 19530/tcp

# To stop Milvus standalone, run:
# $ sudo docker compose down

# To delete data after stopping Milvus, run:
# $ sudo rm -rf  volumes

# SKIP STEPS 1 AND 2 OF YOU ALREADY HAVE A MILVUS ENV SETUP
# 1) SET UP AN ENVIROMENT WITH "python3 -m venv milvus_env"
# 2) ACTIVATE ENVIRONMENT WITH "C:\pAth to milvus_env\Scripts\Activate.ps1 "
# 3) INSTALL NECESARRY PACKAGES "pip3 install pymilvus protobuf grpcio-tools jupyterlab"

import pymilvus

from pymilvus import connections, utility, Collection ,CollectionSchenma, FieldSchema, Datatype

######################################################### CONNECT TO MILVUS  ########################################################D
###########################################################################################################################################    
# Connect to existing milvus server
connections.connect(
    alias = "default ",
    host = 'locahost',
    port = '19538'
)
######################################################### DEFINITIONS AND  COLLECTION SETUP ########################################################D
###########################################################################################################################################      
# Setup and define Field Schema
# this is like naming the columns of a table
# and the kind of data that will be present
song_name = FieldSchema(
   name= "song_name",
   dtype= Datatype.VARCHAR,
   max_length = 200,
)

song_id = FieldSchema(
   name= "song_id",
   dtype= Datatype.INT64,
   is_primary = True,
)

song_vec = FieldSchema(
   name= "song_vec",
   dtype= Datatype.FLOAT.VECTOR,
   dim = 2,
)

# Now we can define a Collection Schema based
# on the Field Schema and the collection a name
# its like naming the table
collection_schema =  CollectionSchema (
    fields = [song_name ,song_id,song_vec]
    description = "Album Songs"
)




# Now we can now name the  collection 
collection = Collection(
    name = "Album1",
    schema =  collection_schema,
    using = 'default')

utility.list_collections()
utility.drop_collection()

######################################################### PARTIONING #####################################################################D
##########################################################################################################################################

collection.create_partition("Disc1")
collection.create_partition("Disc2")
collection.create_partition("Disc3")
collection.has_partitiont("Disc1")


######################################################### ENTERING RECORDS #####################################################################D
##########################################################################################################################################

# first data must be generated each column is a vector
# this is different than say pytorch where each row is a vector
# in any case this code will lists om a column wise fashion ...
# 5 names, 5 song ids, 5 listen counts, 5 2d vectors
import random
import string

num_entities =5
data = [
    [''.join(random.choices(string.ascii_uppercase, k=7)) for _ in range(num_entities)],  # song name
    [i for i in range(num_entities)],  # song ID
    [random.randint(0, 10000) for _ in range(num_entities)],  # some random integer
    [[random.random() for _ in range(2)] for _ in range(num_entities)]  # song_vec - 2d vector
]

# we can remove entries based on collection schema
expr = 'song_id in [0,20]'
collection.delete(expr)


######################################################## INDEXING #####################################################################D
##########################################################################################################################################

# Prepare the index parameters as follows:
# see also https://milvus.io/docs/v2.3.x/build_index.md
# for all the parameter types and definitions for
# example L2 is for euclidean distance but there are others
# once indexed a single value x becomes the vector [x1,...,xn]
# for every value in that column
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}

collection.create_index(
  field_name="song_id", 
  index_params=index_params
)




######################################################### SEARCHING AND QUERY #############################################################
##########################################################################################################################################

# before we can search we must first connect to milvus and load the 
# collection
collection.load(replica_number =1)


# the magic happens here
# we have input data in vector form. This search will use the field song vector
# to look for similarity based on indexing parameters in the param 
# and then look up the song names that corresponds
results = collection.search(
    data = [[0.1,0.2]],
    anns_field = "song_vec",
    param = { "metric_type": "L2", "params": {"search_k" :64} },
    limit = 5,
    expr = None,
    output_fields = ['song_name']
)

for result in results[0]:
    print (result)