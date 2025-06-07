import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Parse the XML file
xml_file = r"c:\Users\91942\Downloads\ad20250605\ad20250605.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "patents"

# Load embedding model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Create the index with vector mapping for embeddings
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "reel_no": {"type": "keyword"},
                "frame_no": {"type": "keyword"},
                "assignors": {"type": "text"},
                "assignees": {"type": "text"},
                "invention_title": {"type": "text"},
                "conveyance_text": {"type": "text"},
                "doc_numbers": {"type": "keyword"},
                "raw_text": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 512, "index": True, "similarity": "cosine"}
            }
        }
    })

# Helper to extract text from an element
get_text = lambda el: el.text.strip() if el is not None and el.text else ""

# Prepare documents for bulk upload
bulk_data = []
for pa in root.findall('.//patent-assignment'):
    record = pa.find('assignment-record')
    if record is None:
        continue
    reel_no = get_text(record.find('reel-no'))
    frame_no = get_text(record.find('frame-no'))
    conveyance_text = get_text(record.find('conveyance-text'))
    # Assignors
    assignors = ", ".join([
        get_text(a.find('name')) for a in pa.findall('.//patent-assignor') if get_text(a.find('name'))
    ])
    # Assignees
    assignees = ", ".join([
        get_text(a.find('name')) for a in pa.findall('.//patent-assignee') if get_text(a.find('name'))
    ])
    # Invention title
    invention_title = ""
    doc_numbers = []
    for prop in pa.findall('.//patent-property'):
        title = prop.find('invention-title')
        if title is not None:
            invention_title = get_text(title)
        for doc in prop.findall('document-id'):
            doc_num = get_text(doc.find('doc-number'))
            if doc_num:
                doc_numbers.append(doc_num)
    # Only embed the invention title
    embedding = None
    if invention_title:
        inputs = clip_processor(text=[invention_title], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)[0].cpu().numpy().astype(np.float32).tolist()
    else:
        embedding = [0.0]*512  # fallback for missing title
    doc = {
        "reel_no": reel_no,
        "frame_no": frame_no,
        "assignors": assignors,
        "assignees": assignees,
        "invention_title": invention_title,
        "conveyance_text": conveyance_text,
        "doc_numbers": doc_numbers,
        "raw_text": invention_title,
        "embedding": embedding
    }
    bulk_data.append({"_index": index_name, "_source": doc})

# Bulk insert
if bulk_data:
    helpers.bulk(es, bulk_data)
    print(f"Inserted {len(bulk_data)} patent records with embeddings into Elasticsearch.")
else:
    print("No patent records found to insert.")
