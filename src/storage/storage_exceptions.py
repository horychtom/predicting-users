"""This module contains the exceptions needed for the StorageClient."""

from typing import Optional


class StorageException(Exception):
    """Abstract Storage Exception class.

    This class must be extended and not instantiated.
    """

    def __init__(self, error_message: str, code: Optional[int]):
        """Raise RuntimeError if this exception is instantiated."""
        if type(self) == StorageException:
            raise RuntimeError(
                "Abstract class <StorageException> must not be instantiated.")
        self.status = error_message
        self.error = {"message": error_message}
        self.code = code or 400  # DEFAULT_EXCEPTION_CODE = 400


class FileNotExistException(StorageException):
    """Raise this exception if we want to upload a file that does not exist locally."""

    def __init__(
        self, error_message: str = "Can't upload a file that does not exist locally.", code: Optional[int] = 404
    ):
        """Initialize this exception."""
        super().__init__(error_message, code=code)


class BlobNotExistException(StorageException):
    """Raise this exception if we want to download a blob that does not exist in GCS."""

    def __init__(
        self, error_message: str = "Can't download a file that does not exist in GCS.", code: Optional[int] = 404
    ):
        """Initialize this exception."""
        super().__init__(error_message, code=code)
