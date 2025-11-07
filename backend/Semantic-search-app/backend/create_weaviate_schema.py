from weaviate import Client as WeaviateClient
import sys


def class_exists(client, class_name):
    schema = client.schema.get()
    classes = schema.get('classes') or []
    return any(c.get('class') == class_name for c in classes)


def create_class(client, class_obj):
    name = class_obj.get('class')
    if class_exists(client, name):
        print(f"Class '{name}' already exists, skipping.")
        return
    client.schema.create_class(class_obj)
    print(f"Created class '{name}'")


def main():
    client = WeaviateClient("http://weaviate:8080")

    # Common properties used by classes
    props = [
        {
            "name": "text",
            "dataType": ["text"]
        },
        {
            "name": "starttime",
            "dataType": ["number"]
        },
        {
            "name": "endtime",
            "dataType": ["number"]
        },
        {
            "name": "metadata",
            "dataType": ["text"]
        },
        {
            "name": "video_id",
            "dataType": ["text"]
        }
    ]

    class_objects = [
        {
            "class": "Video_text_description",
            "description": "Text + frame descriptions for a video sentence",
            "vectorizer": "none",
            "properties": props,
        },
        {
            "class": "Video_text",
            "description": "Text descriptions from transcripts",
            "vectorizer": "none",
            "properties": props,
        },
        {
            "class": "Video_description",
            "description": "Video frame descriptions",
            "vectorizer": "none",
            "properties": props,
        },
    ]

    for c in class_objects:
        try:
            create_class(client, c)
        except Exception as e:
            print(f"Failed to create class {c.get('class')}: {e}")
            sys.exit(1)

    print("Schema creation complete.")


if __name__ == '__main__':
    main()
