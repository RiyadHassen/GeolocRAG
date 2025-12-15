import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")
    with open(index_path + '.paths', 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    return index, image_paths

def retrieve_similar_images(query,model, index_path, top_k=5):
    print("Loading RAG Index")
    index, image_paths = load_faiss_index(index_path)
    if query.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print("Input is an image path, loading image...")
        image = Image.open(query)
    print("Encoding query image")
    query_embedding = model.encode(image)
    query_vector = np.array([query_embedding]).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)
    print("Returning Candidate images")
    similar_images = []
    for idx in indices[0]:
        similar_images.append(image_paths[idx])
    
    return similar_images, distances

if __name__ == "__main__":
    
    model = SentenceTransformer('clip-ViT-B-32')
    index_path = "/nobackup/riyad/NAVIG/NaviClues/Navig/guidebook/index.index"
    query_image_path = "/nobackup/riyad/NAVIG/data/sample_image/Albania_28.jpg"
    retrived_images, distances = retrieve_similar_images(query_image_path, model, index_path, top_k=5)
    BASE_PATH = "/nobackup/riyad/NAVIG/data/"
    print("Top 5 similar images:")
    for img_path, distance in zip(retrived_images, distances):
        img = Image.open(BASE_PATH+img_path)
        plt.imshow(img)
        print(distance)
    plt.show()