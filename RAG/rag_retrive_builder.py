import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")
    with open(index_path + '.paths', 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    return index, image_paths

def retrieve_similar_images(query,model, index, image_paths, top_k=5):
    if query.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print("Input is an image path, loading image...")
        image = Image.open(query)
    query_embedding = model.encode(image)
    query_vector = np.array([query_embedding]).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)
    
    similar_images = []
    for idx in indices[0]:
        similar_images.append(image_paths[idx])
    
    return similar_images

if __name__ == "__main__":
    
    model = SentenceTransformer('clip-ViT-B-32')
    index_path = "/nobackup/riyad/NAVICLUES/NaviClues/Navig/guidebook/index.index"
    index, image_paths = load_faiss_index(index_path)
    
    query_image_path = "/nobackup/riyad/NAVICLUES/data/sample_image/Albania_28.jpg"
    similar_images = retrieve_similar_images(query_image_path, model, index, image_paths, top_k=5)
    
    print("Top 5 similar images:")
    for img_path in similar_images:
        img = Image.open(img_path)
        img.show()
        print(img_path)