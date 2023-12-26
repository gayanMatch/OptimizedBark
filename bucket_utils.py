import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "voice-npz.json"

def download_voice(bucket_name, voice, destination_folder):
    # Initialize a storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(voice + ".npz")

    if blob.exists():
        destination_file_name = f"{destination_folder}/{blob.name}"
        blob.download_to_filename(destination_file_name,)
        return True
    else:
        return False


if __name__ == '__main__':
    # Call the function with your bucket name and the destination folder path
    download_voice('tts-voices-npz', 'dude_from_1_10_to_1_18_base', 'bark/assets/prompts')
