from sentence_transformers import SentenceTransformer
from weaviate import Client as WeaviateClient
import time


def main():
    # connect to weaviate running on localhost
    client = WeaviateClient("http://weaviate:8080")

    # wait a moment for weaviate to be healthy
    try:
        client.is_ready()
    except Exception:
        print("Weaviate does not appear ready at http://weaviate:8080. Make sure the container is running.")
        raise

    # load sentence-transformer model (will download on first run)
    print("Loading sentence-transformers model (all-MiniLM-L6-v2). This may take a moment.)")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Sample object
    text = "We are going to have a tremendous victory."
    video_desc = "a man in a suit and red neck tie"
    combined_text = f"In the video you can hear: {text} In the video you can see: {video_desc}"

    vector = model.encode(combined_text)

    properties = {
        "text": combined_text,
        "starttime": 0.0,
        "endtime": 2.0,
        "metadata": "sample",
        "video_id": "sample_video_1",
    }

    print("Adding sample object to class Video_text_description...")
    # We use batch to add and then flush
    client.batch.add_data_object(properties, "Video_text_description", vector=vector)
    client.batch.create_objects()

    # small wait and verify via an aggregate count
    time.sleep(1)
    agg = client.query.aggregate("Video_text_description").with_meta_count().do()
    print("Aggregate result:", agg)

    print("Sample object inserted. You can now retry your /search endpoint or run queries against Weaviate.")


if __name__ == '__main__':
    main()
