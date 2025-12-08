import clip
import os
from glob import glob
from PIL import Image
import faiss
import numpy as np
from pathlib import Path
import argparse
from sentence_transformers import SentenceTransformer,util

# load CLIP model for vision rag
# 
def generate_clip_embedding(images_path, model):
    #image_paths = glob(os.path.join(images_path), recursive=True)
    #extenstions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = [str(file) for file in Path(images_path).rglob('*')]
    print(image_paths[:5])
    embeddings = []
    count = 0
    for img_path in image_paths:
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            count += 1
            continue
        try:
            embedding = model.encode(image)
        except Exception as e:
            print(f"Error generating embedding for image {img_path}: {e}")
            count += 1
            continue
        embeddings.append(embedding)
    print(f"Total images failed to load: {count}")
    return embeddings, image_paths



def generate_faiss_index(embeddings, image_paths, output_path):
    # query vector against every single vector in the dataset, 
    # ensuring exact results at the cost of higher computational complexity.

    dimension = len(embeddings[0])
    print(f"Embedding dimension: {dimension}")
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    vectors = np.array(embeddings).astype(np.float32)
   
    # add vectors to the index with IDs
    print(f"vectors shape: {vectors.shape} embeddings length: {len(embeddings)}")
    index.add_with_ids(vectors, np.array(range(len(embeddings))))

    #save the index
    faiss.write_index(index, output_path)
    print(f"FAISS index saved to {output_path}")

    with open(output_path + '.paths', 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")

    return index

    
    

if __name__ == "__main__":
    #modify paths get from arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--images_path', type=str, required=True, help='Path to the directory containing images')
    argparser.add_argument('--output_faiss_path', type=str, required=True, help='Path to save the FAISS index')
    args = argparser.parse_args()   
    image_path = args.images_path
    output_faiss_path = args.output_faiss_path
    
    model = SentenceTransformer('clip-ViT-B-32')
    #image_path = "/nobackup/riyad/NAVICLUES/data/sample_image"
    embedding, img_path = generate_clip_embedding(image_path, model)
    #output_faiss_path = "/nobackup/riyad/NAVICLUES/NaviClues/Navig/guidebook/index.index"
    faiss_index = generate_faiss_index(embedding, img_path, output_faiss_path)


    




