from typing import List
from fastembed import TextEmbedding, ImageEmbedding

# Models
TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
IMAGE_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"


# 🔹 Convert text to embeddings
def convert_text_to_embeddings(documents: List[str], embedding_model: str = TEXT_MODEL_NAME) -> List:
    text_embedding_model = TextEmbedding(model_name=embedding_model)
    return list(text_embedding_model.embed(documents))


# 🔹 Convert images to embeddings
def convert_image_to_embeddings(images: List[str], embedding_model: str = IMAGE_MODEL_NAME) -> List:
    image_model = ImageEmbedding(model_name=embedding_model)
    return list(image_model.embed(images))


# 🔹 Search similar TEXT
def search_similar_text(collection_name, client, query, limit=3):
    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    query_embedding = list(text_model.embed([query]))[0]

    search_results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,   # ✅ direct embedding
        using="text",            # ✅ specify vector name
        limit=limit
    )

    return search_results.points


# 🔹 Search similar IMAGE
def search_similar_image(collection_name, client, query_image_path, limit=3):
    image_model = ImageEmbedding(model_name=IMAGE_MODEL_NAME)
    query_embedding = list(image_model.embed([query_image_path]))[0]

    search_results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,   # ✅ direct embedding
        using="image",           # ✅ specify vector name
        limit=limit
    )

    return search_results.points


# 🔹 Merge results
def merge_results(text_results, image_results):
    return text_results + image_results


# 🔹 Test block
if __name__ == "__main__":
    image_path = ["../data/images/sample.jpg"]
    image_embedding = convert_image_to_embeddings(image_path)
    print("✅ Image embedding generated:", len(image_embedding))