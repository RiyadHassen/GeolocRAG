import faiss
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from matplotlib import pyplot as plt


def load_faiss_index(index_path):
    """
    Load FAISS index and corresponding image paths.
    
    Args:
        index_path: Path to the FAISS index file (e.g., 'index.index')
    
    Returns:
        index: FAISS index object
        image_paths: List of image paths
    """
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")
    
    # Load image paths from .paths file
    paths_file = index_path + '.paths'
    if not os.path.exists(paths_file):
        raise FileNotFoundError(f"Paths file not found: {paths_file}")
    
    with open(paths_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(image_paths)} image paths")
    return index, image_paths


def retrieve_similar_images(query, model, index_path, 
         image_paths= "/content/drive/MyDrive/Navig/data/guidebook", top_k=5):
    """
    Retrieve top-k similar images from FAISS index.
    
    Args:
        query: Path to query image or PIL Image object
        model: SentenceTransformer model for encoding
        index: FAISS index object
        image_paths: List of image paths corresponding to index
        top_k: Number of similar images to retrieve
    
    Returns:
        similar_images: List of top-k similar image paths
        distances: Array of distances for the retrieved images
    """
    # Load image if query is a path
    if isinstance(query, str) and query.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Loading query image: {query}")
        image = Image.open(query)
    else:
        image = query
    
    # Encode query image
    print("Encoding query image...")
    query_embedding = model.encode(image)
    query_vector = np.array([query_embedding]).astype(np.float32)

    index, image_list = load_faiss_index(index_path)
    
    # Search in FAISS index
    print(f"Searching for top {top_k} similar images...")
    distances, indices = index.search(query_vector, top_k)
    
    # Retrieve image paths
    similar_images = []
    for idx in indices[0]:
        similar_images.append(image_list[idx])
    
    print(f"Retrieved {len(similar_images)} similar images")
    return similar_images, distances[0]


if __name__ == "__main__":
    # Configuration
    BASE_DIR = "/nobackup/riyad/NAVIG/data"
    INDEX_PATH = "/nobackup/riyad/NAVIG/NaviClues/Navig/guidebook/index.index"
    QUERY_IMAGE_PATH = "/nobackup/riyad/NAVIG/data/sample_image/Albania_28.jpg"
    TOP_K = 5
    
    # Load model
    print("Loading CLIP model...")
    model = SentenceTransformer('clip-ViT-B-32')
    
    # Load FAISS index (do this once!)
    print("\nLoading FAISS index...")
    index, image_paths = load_faiss_index(INDEX_PATH)
    
    # Retrieve similar images
    print(f"\nQuerying with: {QUERY_IMAGE_PATH}")
    similar_images, distances = retrieve_similar_images(
        QUERY_IMAGE_PATH, 
        model, 
        index, 
        image_paths, 
        top_k=TOP_K
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"Top {TOP_K} similar images:")
    print(f"{'='*80}")
    
    # Show query image
    query_image = Image.open(QUERY_IMAGE_PATH)
    plt.figure(figsize=(6, 6))
    plt.title("Query Image")
    plt.imshow(query_image)
    plt.axis('off')
    plt.show()
    
    # Show retrieved images
    for i, (img_path, distance) in enumerate(zip(similar_images, distances), 1):
        full_path = os.path.join(BASE_DIR, img_path)
        print(f"\n[{i}] Distance: {distance:.4f}")
        print(f"    Path: {img_path}")
        
        img = Image.open(full_path)
        plt.figure(figsize=(6, 6))
        plt.title(f"Retrieved Image {i} (Distance: {distance:.4f})")
        plt.imshow(img)
        plt.axis('off')
        plt.show()