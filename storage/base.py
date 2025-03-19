from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

class GraphDatabaseInterface(ABC):
    """Abstract base class for graph database operations"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node with given label and properties"""
        pass
    
    @abstractmethod
    def create_relationship(self, start_node_id: int, end_node_id: int, 
                          relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between nodes"""
        pass
    
    @abstractmethod
    def query(self, query_string: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a database query"""
        pass
    
    @abstractmethod
    def get_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID"""
        pass

class ObjectStorageInterface(ABC):
    """Abstract base class for object storage operations"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the storage service"""
        pass
    
    @abstractmethod
    def store_file(self, file_data: bytes, file_name: str, content_type: str) -> str:
        """Store a file and return its URL/identifier"""
        pass
    
    @abstractmethod
    def get_file(self, file_identifier: str) -> Optional[bytes]:
        """Retrieve a file by its identifier"""
        pass
    
    @abstractmethod
    def delete_file(self, file_identifier: str) -> bool:
        """Delete a file by its identifier"""
        pass
    
    @abstractmethod
    def list_files(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """List files with optional prefix filter"""
        pass
