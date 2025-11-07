import json
import os
from sentence_transformers import SentenceTransformer
from weaviate import Client as WeaviateClient

# Connect to Weaviate
client = WeaviateClient("http://localhost:8080")

# Load the embedding model
print("Loading sentence-transformers model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def insert_video_json(json_file_path, video_id=None, metadata_str=None):
    """
    Insert a video JSON file into Weaviate.
    
    Args:
        json_file_path: Path to the JSON file
        video_id: Optional video ID (defaults to filename without extension)
        metadata_str: Optional metadata string
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use provided video_id or derive from filename
    if video_id is None:
        video_id = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # Use provided metadata or default
    if metadata_str is None:
        metadata_str = f"Video: {video_id}"
    
    print(f"Processing {len(data.get('sentences', []))} sentences from {video_id}...")
    
    with client.batch(batch_size=100) as batch:
        for sent in data.get('sentences', []):
            # Build combined text from sentence and frame_data (OCR text)
            frame_texts = sent.get('frame_data', [])
            # Filter out empty strings and join
            frame_text = ", ".join([f.strip() for f in frame_texts if f and f.strip()])
            
            if frame_text:
                combined_text = f"In the video you can hear: {sent['sentence']} In the video you can see: {frame_text}"
            else:
                # Fallback: just use sentence if no frame data
                combined_text = f"In the video you can hear: {sent['sentence']}"
            
            # Generate embedding
            embedding = model.encode(combined_text)
            
            # Prepare properties
            properties = {
                "text": combined_text,
                "starttime": float(sent.get('starttime', 0)),
                "endtime": float(sent.get('endtime', 0)),
                "metadata": metadata_str,
                "video_id": video_id,
            }
            
            # Insert into Video_text_description class
            batch.add_data_object(
                properties,
                "Video_text_description",
                vector=embedding
            )
    
    print(f"Successfully inserted data from {video_id} into Weaviate!")
    print(f"Total sentences processed: {len(data.get('sentences', []))}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python insert_video_to_weaviate.py <json_file> [video_id] [metadata]")
        print("Example: python insert_video_to_weaviate.py output_data/YourVideo.v4.json YourVideo 'Java OOP Lecture'")
        sys.exit(1)
    
    json_file = sys.argv[1]
    video_id = sys.argv[2] if len(sys.argv) > 2 else None
    metadata = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    insert_video_json(json_file, video_id, metadata)
