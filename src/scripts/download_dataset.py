from src.storage import storage_client

storage_client.download_from_gcs_to_local_directory_or_file(
    '.', 'datasets/final.csv')
