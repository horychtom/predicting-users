"""This module contains helper methods to access GCP storage."""

import os

from google.cloud import storage

from config import GCP_BUCKET_NAME, GCP_STORAGE_SERVICE_ACCOUNT
from src.storage.storage_exceptions import BlobNotExistException, FileNotExistException


class StorageClient:
    """A GCS client."""

    def __init__(self):
        """Initialize a GCS client."""
        storage_client = storage.Client.from_service_account_info(
            GCP_STORAGE_SERVICE_ACCOUNT)
        self._bucket = storage_client.get_bucket(GCP_BUCKET_NAME)

    def upload_local_directory_or_file_to_gcs(self, local_path, gcs_path):
        """
        Upload a local directory or file to a bucket.

        WARNING: Use this method if and only if your source directory contains a limited amount of files.
        This method uploads recursively and will only continue with the next file when the current file is processed.
        :param local_path: The local file or directory.
        :param gcs_path: Where to upload it in GCS.
        """
        # recursion base case when the local path is a file
        if not os.path.exists(local_path):
            raise FileNotExistException

        if os.path.isfile(local_path):
            self._upload_blob_to_bucket(
                local_path=local_path, gcs_path=gcs_path)

        else:
            files_or_dirs = os.listdir(local_path)
            files = [f for f in files_or_dirs if os.path.isfile(
                os.path.join(local_path, f))]
            dirs = [d for d in files_or_dirs if os.path.isdir(
                os.path.join(local_path, d))]

            # if it is not a file, upload all files that are within that directory and ...
            for f in files:
                self._upload_blob_to_bucket(local_path=os.path.join(
                    local_path, f), gcs_path=os.path.join(gcs_path, f))

            # ... recursively upload all subdirectories
            for d in dirs:
                self.upload_local_directory_or_file_to_gcs(
                    local_path=os.path.join(local_path, d), gcs_path=os.path.join(gcs_path, d)
                )

    def download_from_gcs_to_local_directory_or_file(self, local_path, gcs_path):
        """Download a directory or file from GCS."""
        if not self.blob_exists(gcs_path=gcs_path):
            raise BlobNotExistException

        blobs = self._bucket.list_blobs(prefix=gcs_path)  # Get list of files
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            os.makedirs(os.path.join(local_path, directory), exist_ok=True)
            self._download_blob_from_bucket(local_path=os.path.join(
                local_path, blob.name), gcs_path=blob.name)

    def _upload_blob_to_bucket(self, local_path, gcs_path):
        """Upload data to a bucket."""
        blob = self._bucket.blob(gcs_path)  # Filename that will be saved
        blob.upload_from_filename(local_path)

    def _download_blob_from_bucket(self, local_path, gcs_path):
        """Download a file from a bucket."""
        blob = self._bucket.blob(gcs_path)
        print(f"Downloading blob: {blob.name}")
        blob.download_to_filename(filename=local_path)

    # todo: Maybe add exception? We have to discuss the intended behaviour.
    def delete_blob(self, gcs_path):
        """Delete a blob by its name."""
        blob = self._bucket.blob(gcs_path)
        blob.delete()

    def blob_exists(self, gcs_path) -> bool:
        """Check if a blob exists."""
        # Case 1: path points to file directly
        blob_exists = self._bucket.blob(gcs_path).exists()
        if blob_exists:
            return blob_exists

        # Case 2: path points to directory. Check if directory contains something.
        blobs = self._bucket.list_blobs(prefix=gcs_path)
        try:
            blob = next(blobs)
            return True if blob else False
        except StopIteration:
            return False
