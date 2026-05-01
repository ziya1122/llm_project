#%%
import os
import uuid
import pandas as pd
from fastembed import TextEmbedding, ImageEmbedding
from qdrant_client import QdrantClient, models
import sys

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings_utils import (
    convert_text_to_embeddings,
    convert_image_to_embeddings,
    TEXT_MODEL_NAME,
    IMAGE_MODEL_NAME
)

# ✅ FIXED BASE DATA PATH (folder, not file)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def create_uuid_from_image_id(image_id):
    NAMESPACE_UUID = uuid.UUID('12345678-1234-5678-1234-567812345678')
    return str(uuid.uuid5(NAMESPACE_UUID, image_id))


def create_embeddings(collection_name):
    # ✅ FIX captions path (NO duplicate captions.txt)
    captions_path = os.path.join(DATA_DIR, "captions.txt")

    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"captions.txt not found at {captions_path}")

    caption_df = pd.read_csv(
        captions_path,
        sep='\t',
        header=None,
        names=['image_id', 'caption']
    )

    # ✅ FIX images path
    images_dir = os.path.join(DATA_DIR, "images")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"images folder not found at {images_dir}")

    image_directory = os.listdir(images_dir)

    # Filter valid images
    images = []
    for image in image_directory:
        if image.split('.')[0] in caption_df['image_id'].values:
            images.append(image)

    # Create structured docs
    image_docs = []
    for image in images:
        image_id = image.split('.')[0]
        caption = caption_df[caption_df['image_id'] == image_id]['caption'].values[0]

        # ✅ FIX path join (important)
        image_path = os.path.join(images_dir, image)

        image_docs.append({
            'image_id': image_id,
            'caption': caption,
            'image_path': image_path
        })

    # Convert text to embeddings
    captions = [doc['caption'] for doc in image_docs]
    embeddings = convert_text_to_embeddings(captions)

    for idx, embedding in enumerate(embeddings):
        image_docs[idx]['caption_embedding'] = embedding

    # Convert images to embeddings
    image_embeddings = convert_image_to_embeddings(
        [doc['image_path'] for doc in image_docs]
    )

    for idx, embedding in enumerate(image_embeddings):
        image_docs[idx]['image_embedding'] = embedding

    # ✅ Using in-memory Qdrant (no Docker needed)
    client = QdrantClient(":memory:")

    # Model sizes
    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    text_embeddings_size = text_model._get_model_description(TEXT_MODEL_NAME).dim

    image_model = ImageEmbedding(model_name=IMAGE_MODEL_NAME)
    image_embeddings_size = image_model._get_model_description(IMAGE_MODEL_NAME).dim

    # Create collection if not exists
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image": models.VectorParams(size=image_embeddings_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=text_embeddings_size, distance=models.Distance.COSINE),
            }
        )

    # Upload data
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=create_uuid_from_image_id(doc['image_id']),
                vector={
                    "text": doc['caption_embedding'],
                    "image": doc['image_embedding'],
                },
                payload={
                    "image_id": doc['image_id'],
                    "caption": doc['caption'],
                    "image_path": doc['image_path']
                }
            )
            for doc in image_docs
        ]
    )

    print("✅ Embeddings created successfully!")
    return client

#%%