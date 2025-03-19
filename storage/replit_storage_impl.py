import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import mimetypes
from replit.object_storage import Client as ObjectStorageClient
from .base import ObjectStorageInterface

logger = logging.getLogger(__name__)

class ReplitObjectStorage(ObjectStorageInterface):
    """Replit Object Storage implementation"""

    def __init__(self, bucket_name: str = "default"):
        self.bucket_name = bucket_name
        self.client = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        try:
            self.client = ObjectStorageClient()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Replit Storage: {str(e)}")
            raise

    def store_file(self, file_data: bytes, file_name: str, content_type: str) -> str:
        """Store a file and return its URL"""
        try:
            # Generate a unique file path using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.bucket_name}/{timestamp}_{file_name}"

            # Store the file
            self.client.upload_bytes(
                data=file_data,
                path=file_path,
                mime_type=content_type or mimetypes.guess_type(file_name)[0]
            )

            # Get the file URL
            url = self.client.get_url(file_path)
            return url

        except Exception as e:
            self.logger.error(f"Error storing file: {str(e)}")
            raise

    def get_file(self, file_identifier: str) -> Optional[bytes]:
        """Retrieve a file by its identifier"""
        try:
            return self.client.get_bytes(file_identifier)
        except Exception as e:
            self.logger.error(f"Error retrieving file: {str(e)}")
            raise

    def delete_file(self, file_identifier: str) -> bool:
        """Delete a file by its identifier"""
        try:
            self.client.delete(file_identifier)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file: {str(e)}")
            raise

    def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files with optional prefix filter"""
        try:
            # Get all keys with the given prefix
            full_prefix = f"{self.bucket_name}/{prefix}" if prefix else self.bucket_name
            files = self.client.list(prefix=full_prefix)

            # Get metadata for each file
            result = []
            for file_path in files:
                try:
                    url = self.client.get_url(file_path)
                    size = len(self.client.get_bytes(file_path))
                    result.append({
                        'name': file_path.split('/')[-1],  # Get filename from path
                        'path': file_path,
                        'size': size,
                        'url': url
                    })
                except Exception as e:
                    self.logger.warning(f"Error getting metadata for {file_path}: {str(e)}")
                    continue

            return result

        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            raise