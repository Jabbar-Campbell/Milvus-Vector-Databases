# ALS Disease Image Search Project

This project demonstrates how to create a vector database of images using Milvus and search through those images using a query image. The database is used to store and search embeddings of ALS (Amyotrophic Lateral Sclerosis) disease-related images.

## Overview

Milvus is a vector database that uses n-dimensional vectors instead of traditional tables, making it ideal for tasks like image similarity search. This project includes:

- Setting up Milvus in a Docker container on a Windows machine.
- Creating a Python environment to interact with the Milvus server.
- Preprocessing images and generating embeddings using a pre-trained ResNet-18 model.
- Storing image embeddings in Milvus and performing similarity searches.

## Prerequisites

- Windows machine with Docker installed
- Python 3.x
- Visual Studio Code (VSCode) or any other preferred IDE
- Basic understanding of Docker, Python, and image processing

## Installation

### 1. Docker Setup

1. Install Docker on your Windows machine.
2. Enable virtualization in your BIOS settings if necessary.
3. Download the `milvus-standalone-docker-compose.yml` file from the Milvus GitHub repository:
    ```bash
    wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
    ```
4. Start Milvus in a Docker container:
    ```bash
    sudo docker compose up -d
    ```

### 2. Python Environment Setup

1. Create a Python virtual environment:
    ```bash
    python -m venv milvus_env
    ```
2. Activate the environment:
    ```bash
    .\milvus_env\Scripts\Activate.ps1
    ```
3. Install required Python packages:
    ```bash
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pymilvus protobuf grpcio-tools jupyterlab torchvision
    ```

### 3. Image Preprocessing and Embedding Generation

1. Load the ResNet-18 model from the `torchvision` repository:
    ```python
    import torch
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    model.eval()
    ```
2. Preprocess images using the `torchvision.transforms` module.
3. Generate embeddings using the pre-trained model and prepare them for insertion into Milvus.

### 4. Inserting Data into Milvus

1. Define the collection schema using `FieldSchema` and `CollectionSchema`.
2. Create a collection and insert image embeddings along with metadata (image names, IDs, dates).
3. Create an index on the image vector field to facilitate efficient search.

### 5. Performing Searches

1. Load the collection into memory.
2. Search for similar images using the vector of a query image.
3. Retrieve and display the most similar images based on the search results.

## Usage

1. Activate your Python environment.
2. Ensure Milvus is running in Docker.
3. Run the Python script `ALS DISEASE PROJECT.py` to preprocess images, generate embeddings, insert them into Milvus, and perform image similarity searches.

## Troubleshooting

- **Docker Issues**: Ensure Docker is running and virtualization is enabled in BIOS.
- **SSL Issues with `pip`**: Use the `--trusted-host` option to bypass SSL errors when installing packages.

## References

- [Milvus Documentation](https://milvus.io/docs/)
- [PyTorch/vision Repository](https://pytorch.org/vision/stable/index.html)
- [Vector Databases Overview](https://towardsdatascience.com/deep-dive-into-vector-databases-by-hand-e9ab71f54f80)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
