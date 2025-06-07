from elasticsearch import Elasticsearch
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

es = Elasticsearch("http://localhost:9200")
index_name = "patents"

# Load CLIP model for both text and image embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# CLIP text embedding
def embed_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features[0].cpu().numpy().astype(np.float32).tolist()

# CLIP image embedding
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].cpu().numpy().astype(np.float32).tolist()

def search_by_embedding(embedding, dims=512, top_k=5):
    print(f"Embedding length: {len(embedding)} (should be {dims})")
    if len(embedding) != dims:
        print(f"Error: Embedding length {len(embedding)} does not match expected dims {dims}.")
        return
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": embedding}
                }
            }
        }
    }
    try:
        res = es.search(index=index_name, body=body)
        for hit in res['hits']['hits']:
            doc = hit['_source']
            print(f"Patent ID: {doc.get('doc_numbers', [''])[0] if doc.get('doc_numbers') else ''}")
            print(f"Invention Title: {doc.get('invention_title', '')}")
            print(f"Assignors: {doc.get('assignors', '')}")
            print(f"Assignees: {doc.get('assignees', '')}")
            print(f"Conveyance Text: {doc.get('conveyance_text', '')}")
            print(f"Reel No: {doc.get('reel_no', '')}")
            print(f"Frame No: {doc.get('frame_no', '')}")
            print("-"*40)
    except Exception as e:
        print(f"Elasticsearch error: {e}")

if __name__ == "__main__":
    mode = input("Enter 'text' to search by prompt or 'image' to search by image: ").strip().lower()
    if mode == 'text':
        query = input("Enter your search prompt: ")
        emb = embed_text(query)
        search_by_embedding(emb, dims=512)
    elif mode == 'image':
        image_path = input("Enter the path to your image: ")
        emb = embed_image(image_path)
        search_by_embedding(emb, dims=512)
    else:
        print("Invalid mode.")
