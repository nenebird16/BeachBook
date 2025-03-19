import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import mimetypes
from replit.web import storage
from .base import ObjectStorageInterface

logger = logging.getLogger(__name__)

class ReplitObjectStorage(ObjectStorageInterface):
    """Replit Object Storage implementation"""
    
    def __init__(self, bucket_name: str = "default"):
        self.bucket_name = bucket_name
        self.client = None
        self.bucket = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        try:
            self.client = storage.Client()
            self.bucket = self.client.get_bucket(self.bucket_name)
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Replit Storage: {str(e)}")
            raise
    
    def store_file(self, file_data: bytes, file_name: str, content_type: str) -> str:
        """Store a file and return its URL"""
        try:
            # Generate a unique file path using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{timestamp}_{file_name}"
            
            # Store the file
            blob = self.bucket.blob(file_path)
            blob.upload_from_string(
                file_data,
                content_type=content_type or mimetypes.guess_type(file_name)[0]
            )
            
            # Return the public URL
            return blob.public_url
            
        except Exception as e:
            self.logger.error(f"Error storing file: {str(e)}")
            raise
    
    def get_file(self, file_identifier: str) -> Optional[bytes]:
        """Retrieve a file by its identifier"""
        try:
            blob = self.bucket.blob(file_identifier)
            return blob.download_as_bytes()
        except Exception as e:
            self.logger.error(f"Error retrieving file: {str(e)}")
            raise
    
    def delete_file(self, file_identifier: str) -> bool:
        """Delete a file by its identifier"""
        try:
            blob = self.bucket.blob(file_identifier)
            blob.delete()
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file: {str(e)}")
            raise
    
    def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files with optional prefix filter"""
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [{
                'name': blob.name,
                'size': blob.size,
                'updated': blob.updated,
                'url': blob.public_url
            } for blob in blobs]
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            raise
