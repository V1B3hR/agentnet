"""
Vector Database Integrations

This module provides native support for multiple vector databases
including Pinecone, Weaviate, and Milvus, enabling AgentNet to
leverage advanced vector search capabilities.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector database search."""

    id: str
    score: float
    metadata: Dict[str, Any]
    content: Optional[str] = None
    vector: Optional[List[float]] = None


class VectorDatabaseAdapter(ABC):
    """Base class for vector database adapters."""

    def __init__(self, **kwargs):
        """Initialize the adapter."""
        self.config = kwargs
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the vector database."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the vector database."""
        pass

    @abstractmethod
    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection/index."""
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Delete a collection/index."""
        pass

    @abstractmethod
    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Insert vectors into the collection."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def update(
        self,
        collection_name: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector or its metadata."""
        pass

    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass

    @abstractmethod
    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        pass


class PineconeAdapter(VectorDatabaseAdapter):
    """
    Pinecone vector database adapter.

    Features:
    - Cloud-native vector database
    - Automatic scaling
    - Metadata filtering
    - Real-time updates
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        project_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Pinecone adapter.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            project_name: Project name (optional)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        try:
            import pinecone

            self.pinecone = pinecone
            PINECONE_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "Pinecone integration requires: pip install pinecone-client>=2.2.0"
            )

        self.api_key = api_key
        self.environment = environment
        self.project_name = project_name
        self.pc = None
        self.indexes = {}

    def connect(self) -> bool:
        """Connect to Pinecone."""
        try:
            self.pc = self.pinecone.Pinecone(
                api_key=self.api_key, environment=self.environment
            )
            self.connected = True
            logger.info("Connected to Pinecone successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from Pinecone."""
        self.pc = None
        self.indexes = {}
        self.connected = False
        return True

    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a Pinecone index."""
        try:
            if not self.connected:
                self.connect()

            # Set default configuration
            config = {
                "dimension": dimension,
                "metric": kwargs.get("metric", "cosine"),
                "pods": kwargs.get("pods", 1),
                "replicas": kwargs.get("replicas", 1),
                "pod_type": kwargs.get("pod_type", "p1.x1"),
            }

            self.pc.create_index(name=name, **config)

            # Wait for index to be ready
            import time

            while not self.pc.describe_index(name).status["ready"]:
                time.sleep(1)

            logger.info(f"Created Pinecone index: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            return False

    def delete_collection(self, name: str) -> bool:
        """Delete a Pinecone index."""
        try:
            if not self.connected:
                self.connect()

            self.pc.delete_index(name)
            if name in self.indexes:
                del self.indexes[name]

            logger.info(f"Deleted Pinecone index: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {e}")
            return False

    def _get_index(self, collection_name: str):
        """Get or create index connection."""
        if collection_name not in self.indexes:
            self.indexes[collection_name] = self.pc.Index(collection_name)
        return self.indexes[collection_name]

    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Insert vectors into Pinecone index."""
        try:
            index = self._get_index(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]

            # Prepare upsert data
            upsert_data = []
            for i, vector in enumerate(vectors):
                item = {
                    "id": ids[i],
                    "values": vector,
                }
                if metadata and i < len(metadata):
                    item["metadata"] = metadata[i]
                upsert_data.append(item)

            # Batch upsert (Pinecone has limits on batch size)
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i : i + batch_size]
                index.upsert(vectors=batch)

            logger.info(f"Inserted {len(vectors)} vectors into {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert vectors into Pinecone: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Pinecone."""
        try:
            index = self._get_index(collection_name)

            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False,
            }

            if filter_conditions:
                query_params["filter"] = filter_conditions

            response = index.query(**query_params)

            results = []
            for match in response["matches"]:
                result = VectorSearchResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {}),
                    content=match.get("metadata", {}).get("content"),
                )
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Failed to search Pinecone: {e}")
            return []

    def update(
        self,
        collection_name: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector in Pinecone."""
        try:
            index = self._get_index(collection_name)

            update_data = {"id": id}
            if vector:
                update_data["values"] = vector
            if metadata:
                update_data["set_metadata"] = metadata

            index.update(**update_data)
            return True
        except Exception as e:
            logger.error(f"Failed to update vector in Pinecone: {e}")
            return False

    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete vectors from Pinecone."""
        try:
            index = self._get_index(collection_name)
            index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}")
            return False

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get Pinecone index information."""
        try:
            if not self.connected:
                self.connect()

            index_info = self.pc.describe_index(name)
            index = self._get_index(name)
            stats = index.describe_index_stats()

            return {
                "name": name,
                "dimension": index_info.dimension,
                "metric": index_info.metric,
                "pods": index_info.pods,
                "replicas": index_info.replicas,
                "pod_type": index_info.pod_type,
                "status": index_info.status,
                "total_vector_count": stats["total_vector_count"],
                "namespaces": stats.get("namespaces", {}),
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone index info: {e}")
            return {"error": str(e)}


class WeaviateAdapter(VectorDatabaseAdapter):
    """
    Weaviate vector database adapter.

    Features:
    - GraphQL API
    - Multi-modal search
    - Automatic vectorization
    - Schema management
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout_config: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        Initialize Weaviate adapter.

        Args:
            url: Weaviate instance URL
            api_key: API key for authentication
            timeout_config: (connect_timeout, read_timeout)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        try:
            import weaviate

            self.weaviate = weaviate
            WEAVIATE_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "Weaviate integration requires: pip install weaviate-client>=3.15.0"
            )

        self.url = url
        self.api_key = api_key
        self.timeout_config = timeout_config or (10, 60)
        self.client = None

    def connect(self) -> bool:
        """Connect to Weaviate."""
        try:
            auth_config = None
            if self.api_key:
                auth_config = self.weaviate.AuthApiKey(api_key=self.api_key)

            self.client = self.weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config,
                timeout_config=self.timeout_config,
            )

            # Test connection
            self.client.schema.get()
            self.connected = True
            logger.info("Connected to Weaviate successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from Weaviate."""
        self.client = None
        self.connected = False
        return True

    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a Weaviate class (collection)."""
        try:
            if not self.connected:
                self.connect()

            # Define class schema
            class_schema = {
                "class": name,
                "description": kwargs.get("description", f"Collection {name}"),
                "vectorizer": kwargs.get("vectorizer", "none"),  # Manual vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Content text",
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Additional metadata",
                    },
                ],
            }

            # Add custom properties if specified
            custom_properties = kwargs.get("properties", [])
            class_schema["properties"].extend(custom_properties)

            self.client.schema.create_class(class_schema)
            logger.info(f"Created Weaviate class: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Weaviate class: {e}")
            return False

    def delete_collection(self, name: str) -> bool:
        """Delete a Weaviate class."""
        try:
            if not self.connected:
                self.connect()

            self.client.schema.delete_class(name)
            logger.info(f"Deleted Weaviate class: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Weaviate class: {e}")
            return False

    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Insert vectors into Weaviate class."""
        try:
            if not self.connected:
                self.connect()

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]

            # Batch insert
            with self.client.batch as batch:
                batch.batch_size = 100

                for i, vector in enumerate(vectors):
                    properties = {"content": "", "metadata": {}}

                    if metadata and i < len(metadata):
                        properties.update(metadata[i])
                        if "content" not in properties:
                            properties["content"] = str(metadata[i])

                    batch.add_data_object(
                        data_object=properties,
                        class_name=collection_name,
                        uuid=ids[i],
                        vector=vector,
                    )

            logger.info(f"Inserted {len(vectors)} vectors into {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert vectors into Weaviate: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Weaviate."""
        try:
            if not self.connected:
                self.connect()

            query = (
                self.client.query.get(collection_name, ["content", "metadata"])
                .with_near_vector({"vector": query_vector})
                .with_limit(top_k)
                .with_additional(["id", "distance"])
            )

            # Add where filter if specified
            if filter_conditions:
                query = query.with_where(filter_conditions)

            result = query.do()

            results = []
            if "errors" not in result:
                objects = result["data"]["Get"][collection_name]
                for obj in objects:
                    result_obj = VectorSearchResult(
                        id=obj["_additional"]["id"],
                        score=1.0
                        - obj["_additional"][
                            "distance"
                        ],  # Convert distance to similarity
                        metadata=obj.get("metadata", {}),
                        content=obj.get("content", ""),
                    )
                    results.append(result_obj)

            return results
        except Exception as e:
            logger.error(f"Failed to search Weaviate: {e}")
            return []

    def update(
        self,
        collection_name: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an object in Weaviate."""
        try:
            if not self.connected:
                self.connect()

            update_data = {}
            if metadata:
                update_data.update(metadata)

            self.client.data_object.update(
                data_object=update_data,
                class_name=collection_name,
                uuid=id,
                vector=vector,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update object in Weaviate: {e}")
            return False

    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete objects from Weaviate."""
        try:
            if not self.connected:
                self.connect()

            for obj_id in ids:
                self.client.data_object.delete(uuid=obj_id)

            logger.info(f"Deleted {len(ids)} objects from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete objects from Weaviate: {e}")
            return False

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get Weaviate class information."""
        try:
            if not self.connected:
                self.connect()

            schema = self.client.schema.get(name)

            # Get object count
            result = self.client.query.aggregate(name).with_meta_count().do()

            count = 0
            if "errors" not in result:
                count = result["data"]["Aggregate"][name][0]["meta"]["count"]

            return {
                "name": name,
                "description": schema.get("description", ""),
                "vectorizer": schema.get("vectorizer", ""),
                "properties": schema.get("properties", []),
                "object_count": count,
            }
        except Exception as e:
            logger.error(f"Failed to get Weaviate class info: {e}")
            return {"error": str(e)}


class MilvusAdapter(VectorDatabaseAdapter):
    """
    Milvus vector database adapter.

    Features:
    - High-performance vector search
    - Hybrid search capabilities
    - Distributed architecture
    - Multiple index types
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Milvus adapter.

        Args:
            host: Milvus server host
            port: Milvus server port
            user: Username for authentication
            password: Password for authentication
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        try:
            from pymilvus import (
                connections,
                Collection,
                CollectionSchema,
                FieldSchema,
                DataType,
            )

            self.pymilvus = __import__("pymilvus")
            self.connections = connections
            self.Collection = Collection
            self.CollectionSchema = CollectionSchema
            self.FieldSchema = FieldSchema
            self.DataType = DataType
            MILVUS_AVAILABLE = True
        except ImportError:
            raise ImportError(
                "Milvus integration requires: pip install pymilvus>=2.3.0"
            )

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connection_name = kwargs.get("connection_name", "default")
        self.collections = {}

    def connect(self) -> bool:
        """Connect to Milvus."""
        try:
            connect_params = {
                "alias": self.connection_name,
                "host": self.host,
                "port": self.port,
            }

            if self.user and self.password:
                connect_params.update(
                    {
                        "user": self.user,
                        "password": self.password,
                    }
                )

            self.connections.connect(**connect_params)
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from Milvus."""
        try:
            self.connections.disconnect(self.connection_name)
            self.connected = False
            self.collections = {}
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {e}")
            return False

    def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a Milvus collection."""
        try:
            if not self.connected:
                self.connect()

            # Define schema
            fields = [
                self.FieldSchema(
                    name="id",
                    dtype=self.DataType.VARCHAR,
                    max_length=36,
                    is_primary=True,
                    auto_id=False,
                ),
                self.FieldSchema(
                    name="vector", dtype=self.DataType.FLOAT_VECTOR, dim=dimension
                ),
                self.FieldSchema(
                    name="content", dtype=self.DataType.VARCHAR, max_length=65535
                ),
            ]

            schema = self.CollectionSchema(
                fields=fields,
                description=kwargs.get("description", f"Collection {name}"),
            )

            collection = self.Collection(
                name=name, schema=schema, using=self.connection_name
            )

            # Create index
            index_params = {
                "metric_type": kwargs.get("metric_type", "L2"),
                "index_type": kwargs.get("index_type", "IVF_FLAT"),
                "params": kwargs.get("index_params", {"nlist": 128}),
            }

            collection.create_index(field_name="vector", index_params=index_params)

            self.collections[name] = collection
            logger.info(f"Created Milvus collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Milvus collection: {e}")
            return False

    def delete_collection(self, name: str) -> bool:
        """Delete a Milvus collection."""
        try:
            if not self.connected:
                self.connect()

            if name in self.collections:
                self.collections[name].drop()
                del self.collections[name]
            else:
                collection = self.Collection(name, using=self.connection_name)
                collection.drop()

            logger.info(f"Deleted Milvus collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Milvus collection: {e}")
            return False

    def _get_collection(self, collection_name: str):
        """Get or create collection connection."""
        if collection_name not in self.collections:
            self.collections[collection_name] = self.Collection(
                collection_name, using=self.connection_name
            )
        return self.collections[collection_name]

    def insert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Insert vectors into Milvus collection."""
        try:
            collection = self._get_collection(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]

            # Prepare data
            contents = []
            for i in range(len(vectors)):
                if metadata and i < len(metadata):
                    content = json.dumps(metadata[i])
                else:
                    content = ""
                contents.append(content)

            data = [ids, vectors, contents]

            # Insert data
            collection.insert(data)
            collection.flush()

            logger.info(f"Inserted {len(vectors)} vectors into {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert vectors into Milvus: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Milvus."""
        try:
            collection = self._get_collection(collection_name)
            collection.load()

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            # Prepare search expression for filtering
            expr = None
            if filter_conditions:
                # Simple filter expression (would need more sophisticated parsing)
                expressions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, str):
                        expressions.append(f'{key} == "{value}"')
                    else:
                        expressions.append(f"{key} == {value}")
                expr = " and ".join(expressions)

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "content"],
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    # Parse metadata from content
                    metadata = {}
                    if hit.entity.get("content"):
                        try:
                            metadata = json.loads(hit.entity.get("content"))
                        except:
                            metadata = {"content": hit.entity.get("content")}

                    result = VectorSearchResult(
                        id=hit.entity.get("id"),
                        score=1.0
                        / (1.0 + hit.distance),  # Convert distance to similarity
                        metadata=metadata,
                        content=metadata.get("content", ""),
                    )
                    search_results.append(result)

            return search_results
        except Exception as e:
            logger.error(f"Failed to search Milvus: {e}")
            return []

    def update(
        self,
        collection_name: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector in Milvus (delete and insert)."""
        try:
            # Milvus doesn't support direct updates, so we delete and insert
            self.delete(collection_name, [id])

            if vector and metadata:
                self.insert(
                    collection_name, vectors=[vector], ids=[id], metadata=[metadata]
                )

            return True
        except Exception as e:
            logger.error(f"Failed to update vector in Milvus: {e}")
            return False

    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete vectors from Milvus collection."""
        try:
            collection = self._get_collection(collection_name)

            # Create expression for deletion
            id_list = '", "'.join(ids)
            expr = f'id in ["{id_list}"]'

            collection.delete(expr)
            collection.flush()

            logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from Milvus: {e}")
            return False

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get Milvus collection information."""
        try:
            if not self.connected:
                self.connect()

            collection = self._get_collection(name)

            return {
                "name": name,
                "schema": collection.schema.to_dict(),
                "num_entities": collection.num_entities,
                "indexes": [index.to_dict() for index in collection.indexes],
                "description": collection.description,
            }
        except Exception as e:
            logger.error(f"Failed to get Milvus collection info: {e}")
            return {"error": str(e)}


# Utility functions
def create_vector_database_adapter(provider: str, **config) -> VectorDatabaseAdapter:
    """
    Create a vector database adapter based on provider.

    Args:
        provider: Database provider (pinecone, weaviate, milvus)
        **config: Provider-specific configuration

    Returns:
        Configured vector database adapter
    """
    provider = provider.lower()

    if provider == "pinecone":
        return PineconeAdapter(**config)
    elif provider == "weaviate":
        return WeaviateAdapter(**config)
    elif provider == "milvus":
        return MilvusAdapter(**config)
    else:
        raise ValueError(f"Unsupported vector database provider: {provider}")


def embed_and_store(
    adapter: VectorDatabaseAdapter,
    collection_name: str,
    texts: List[str],
    embedding_function: callable,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Utility function to embed texts and store in vector database.

    Args:
        adapter: Vector database adapter
        collection_name: Collection name
        texts: List of texts to embed
        embedding_function: Function to generate embeddings
        metadata: Optional metadata for each text

    Returns:
        True if successful
    """
    try:
        # Generate embeddings
        vectors = [embedding_function(text) for text in texts]

        # Prepare metadata with content
        if metadata is None:
            metadata = []

        # Ensure metadata includes content
        final_metadata = []
        for i, text in enumerate(texts):
            meta = metadata[i] if i < len(metadata) else {}
            meta["content"] = text
            final_metadata.append(meta)

        # Insert into database
        return adapter.insert(
            collection_name=collection_name, vectors=vectors, metadata=final_metadata
        )

    except Exception as e:
        logger.error(f"Failed to embed and store texts: {e}")
        return False
