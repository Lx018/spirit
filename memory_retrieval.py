import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import os
import argparse
import sys
import numpy as np
from pydantic import BaseModel, Field, field_validator


# Database schema version
SCHEMA_VERSION = 2


# Pydantic models for type checking and validation
class MemoryItem(BaseModel):
    """Model for a memory item with validation."""
    id: Optional[int] = None
    content: str = Field(..., min_length=1, description="Memory content/text")
    timestamp: Optional[str] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score (0.0 to 1.0)")
    access_count: int = Field(default=0, ge=0)
    last_accessed: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = Field(default=None, description="Embedding vector (any dimension)")
    links: Optional[List[Tuple[int, float]]] = Field(default=None, description="List of (memory_id, weight) tuples")
    tags: Optional[List[str]] = Field(default=None, description="List of tags")
    created_at: Optional[str] = None
    
    # Additional fields for query results
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Cosine similarity score")
    link_weight: Optional[float] = Field(default=None, description="Weight of link connection")
    
    @field_validator('links')
    @classmethod
    def validate_links(cls, v):
        """Ensure all link weights are valid."""
        if v is not None:
            for link_id, weight in v:
                if not isinstance(link_id, int) or link_id < 1:
                    raise ValueError(f"Link ID must be a positive integer, got {link_id}")
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise ValueError(f"Link weight must be non-negative, got {weight}")
        return v
    
    @field_validator('vector')
    @classmethod
    def validate_vector(cls, v):
        """Ensure vector contains only numbers."""
        if v is not None:
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Vector must contain only numeric values")
        return v
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        arbitrary_types_allowed = True


class SearchQuery(BaseModel):
    """Model for search query parameters."""
    query: Optional[str] = Field(default=None, description="Text search query")
    limit: int = Field(default=10, ge=1, le=1000, description="Maximum number of results")
    importance_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum importance")
    time_window: Optional[Tuple[int, int]] = Field(default=None, description="Unix timestamp range (start, end)")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    
    @field_validator('time_window')
    @classmethod
    def validate_time_window(cls, v):
        """Ensure time window is valid."""
        if v is not None:
            start, end = v
            if start >= end:
                raise ValueError("Time window start must be before end")
        return v


class MemoryDatabase:
    """
    A SQLite-based memory storage system for storing and retrieving conversation memories.
    
    Features:
    - Store memory items with metadata (timestamp, tags, importance)
    - Full-text search capabilities
    - Retrieve memories by relevance, recency, or importance
    - Tag-based filtering
    - Automatic memory statistics
    """
    
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize the memory database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database connection and initialize tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Enable foreign keys
        self.cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create schema_info table first
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Check current schema version
        self.cursor.execute("SELECT value FROM schema_info WHERE key = 'version'")
        result = self.cursor.fetchone()
        current_version = int(result[0]) if result else 0
        
        if current_version == 0:
            # Fresh database - create all tables
            print(f"  Creating new database schema (v{SCHEMA_VERSION})...")
            self._create_tables()
            self._set_schema_version(SCHEMA_VERSION)
        elif current_version < SCHEMA_VERSION:
            # Migrate existing database
            print(f"  Migrating database from v{current_version} to v{SCHEMA_VERSION}...")
            self._migrate_database(current_version, SCHEMA_VERSION)
        elif current_version > SCHEMA_VERSION:
            print(f"  WARNING: Database schema v{current_version} is newer than code v{SCHEMA_VERSION}")
        
        self.conn.commit()
    
    def _create_tables(self):
        """Create all database tables from schema model."""
        # Create memories table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed DATETIME,
                metadata TEXT,
                vector TEXT,
                links TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create tags table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        # Create memory_tags junction table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id INTEGER,
                tag_id INTEGER,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                PRIMARY KEY (memory_id, tag_id)
            )
        """)
        
        # Create FTS tables and triggers
        self._create_fts_tables()
    
    def _create_fts_tables(self):
        """Create full-text search tables and triggers."""
        # Create full-text search virtual table
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content=memories,
                content_rowid=id
            )
        """)
        
        # Create triggers to keep FTS in sync
        self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
        
        self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) 
                VALUES('delete', old.id, old.content);
            END
        """)
        
        self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) 
                VALUES('delete', old.id, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
            END
        """)
    
    def _set_schema_version(self, version: int):
        """Set the schema version in the database."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?)
        """, (str(version),))
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get list of column names for a table."""
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        return [col[1] for col in self.cursor.fetchall()]
    
    def _migrate_database(self, from_version: int, to_version: int):
        """Migrate database from one version to another."""
        print(f"  Running migrations from v{from_version} to v{to_version}...")
        
        # Get current columns
        columns = self._get_table_columns("memories")
        
        # Migration v0/v1 -> v2: Add vector and links columns
        if from_version < 2:
            if 'vector' not in columns:
                print("    - Adding 'vector' column to memories table")
                self.cursor.execute("ALTER TABLE memories ADD COLUMN vector TEXT")
            
            if 'links' not in columns:
                print("    - Adding 'links' column to memories table")
                self.cursor.execute("ALTER TABLE memories ADD COLUMN links TEXT")
        
        # Update schema version
        self._set_schema_version(to_version)
        print(f"  ✓ Migration complete")
        print(f"✓ Memory database initialized: {self.db_path}")
    
    def add_memory(self, 
                   content: str, 
                   tags: Optional[List[str]] = None,
                   importance: float = 0.5,
                   metadata: Optional[Dict] = None,
                   vector: Optional[List[float]] = None,
                   links: Optional[List[Tuple[int, float]]] = None) -> int:
        """
        Add a new memory to the database.
        
        Args:
            content: The memory content/text
            tags: Optional list of tags for categorization
            importance: Importance score (0.0 to 1.0)
            metadata: Optional dictionary of additional metadata
            vector: Optional embedding vector (list of floats, any dimension)
            links: Optional list of (memory_id, weight) tuples for linking to other memories
            
        Returns:
            The ID of the newly created memory
        """
        # Validate with Pydantic model
        memory_item = MemoryItem(
            content=content,
            importance=importance,
            metadata=metadata,
            vector=vector,
            links=links,
            tags=tags
        )
        
        # Insert memory
        self.cursor.execute("""
            INSERT INTO memories (content, importance, metadata, vector, links)
            VALUES (?, ?, ?, ?, ?)
        """, (memory_item.content, 
              memory_item.importance, 
              json.dumps(memory_item.metadata) if memory_item.metadata else None,
              json.dumps(memory_item.vector) if memory_item.vector else None,
              json.dumps(memory_item.links) if memory_item.links else None))
        
        memory_id = self.cursor.lastrowid
        
        # Add tags if provided
        if tags:
            for tag in tags:
                self._add_tag_to_memory(memory_id, tag.lower().strip())
        
        self.conn.commit()
        return memory_id
    
    def _add_tag_to_memory(self, memory_id: int, tag_name: str):
        """Internal method to add a tag to a memory."""
        # Insert tag if it doesn't exist
        self.cursor.execute("""
            INSERT OR IGNORE INTO tags (name) VALUES (?)
        """, (tag_name,))
        
        # Get tag ID
        self.cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
        tag_id = self.cursor.fetchone()[0]
        
        # Link memory and tag
        self.cursor.execute("""
            INSERT OR IGNORE INTO memory_tags (memory_id, tag_id)
            VALUES (?, ?)
        """, (memory_id, tag_id))
    
    def search_memories(self, 
                       query: Optional[str] = None,
                       limit: int = 10,
                       importance_threshold: Optional[float] = None,
                       time_window: Optional[Tuple[int, int]] = None,
                       tags: Optional[List[str]] = None) -> List[MemoryItem]:
        """
        Search memories with multiple optional filters.
        
        Args:
            query: Optional text search query (uses FTS if provided)
            limit: Maximum number of results
            importance_threshold: Minimum importance score filter
            time_window: Tuple of (start_timestamp, end_timestamp) in unix time
            tags: Optional list of tags to filter by
            
        Returns:
            List of MemoryItem objects
        """
        # Validate search parameters with Pydantic
        search_query = SearchQuery(
            query=query,
            limit=limit,
            importance_threshold=importance_threshold,
            time_window=time_window,
            tags=tags
        )
        
        conditions = []
        params = []
        
        # Build query based on filters
        if search_query.query:
            # Full-text search
            base_query = """
                SELECT m.id, m.content, m.timestamp, m.importance, 
                       m.access_count, m.last_accessed, m.metadata, m.vector, m.links
                FROM memories_fts fts
                JOIN memories m ON fts.rowid = m.id
            """
            conditions.append("fts.content MATCH ?")
            params.append(search_query.query)
        else:
            # Regular search
            base_query = """
                SELECT m.id, m.content, m.timestamp, m.importance,
                       m.access_count, m.last_accessed, m.metadata, m.vector, m.links
                FROM memories m
            """
        
        # Add importance filter
        if search_query.importance_threshold is not None:
            conditions.append("m.importance >= ?")
            params.append(search_query.importance_threshold)
        
        # Add time window filter
        if search_query.time_window:
            start_time, end_time = search_query.time_window
            conditions.append("CAST(strftime('%s', m.timestamp) AS INTEGER) BETWEEN ? AND ?")
            params.extend([start_time, end_time])
        
        # Add tag filter
        if search_query.tags:
            base_query = base_query.replace("FROM memories m", """
                FROM memories m
                JOIN memory_tags mt ON m.id = mt.memory_id
                JOIN tags t ON mt.tag_id = t.id
            """)
            placeholders = ','.join('?' * len(search_query.tags))
            conditions.append(f"t.name IN ({placeholders})")
            params.extend([tag.lower() for tag in search_query.tags])
        
        # Construct WHERE clause
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        else:
            where_clause = ""
        
        # Order by relevance if FTS, otherwise by importance and recency
        if search_query.query:
            order_by = "ORDER BY rank, m.importance DESC, m.timestamp DESC"
        else:
            order_by = "ORDER BY m.importance DESC, m.timestamp DESC"
        
        # Add DISTINCT if using tags to avoid duplicates
        if search_query.tags:
            base_query = base_query.replace("SELECT m.", "SELECT DISTINCT m.")
        
        full_query = f"{base_query} {where_clause} {order_by} LIMIT ?"
        params.append(search_query.limit)
        
        self.cursor.execute(full_query, params)
        results = self._format_results(self.cursor.fetchall())
        
        # Update access statistics
        for result in results:
            self._update_access(result.id)
        
        return results
    

    
    def get_memory_by_id(self, memory_id: int) -> Optional[MemoryItem]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            MemoryItem object or None if not found
        """
        self.cursor.execute("""
            SELECT id, content, timestamp, importance,
                   access_count, last_accessed, metadata, vector, links
            FROM memories
            WHERE id = ?
        """, (memory_id,))
        
        result = self.cursor.fetchone()
        if result:
            self._update_access(memory_id)
            return self._format_results([result])[0]
        return None
    
    def update_memory(self,
                     memory_id: int,
                     content: Optional[str] = None,
                     importance: Optional[float] = None,
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict] = None,
                     vector: Optional[List[float]] = None,
                     links: Optional[List[Tuple[int, float]]] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: The memory ID
            content: New content (if updating)
            importance: New importance score (if updating)
            tags: New list of tags (replaces existing tags if provided)
            metadata: New metadata (if updating)
            vector: New embedding vector (if updating)
            links: New list of (memory_id, weight) tuples (if updating)
            
        Returns:
            True if successful, False if memory not found
        """
        updates = []
        params = []
        
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if importance is not None:
            importance = max(0.0, min(1.0, importance))
            updates.append("importance = ?")
            params.append(importance)
        
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        
        if vector is not None:
            updates.append("vector = ?")
            params.append(json.dumps(vector))
        
        if links is not None:
            updates.append("links = ?")
            params.append(json.dumps(links))
        
        if updates:
            params.append(memory_id)
            query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
            self.cursor.execute(query, params)
        
        if tags is not None:
            # Remove existing tags
            self.cursor.execute("""
                DELETE FROM memory_tags WHERE memory_id = ?
            """, (memory_id,))
            
            # Add new tags
            for tag in tags:
                self._add_tag_to_memory(memory_id, tag.lower().strip())
        
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            True if successful, False if memory not found
        """
        self.cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def get_all_tags(self) -> List[str]:
        """
        Get all tags in the database.
        
        Returns:
            List of tag names
        """
        self.cursor.execute("SELECT name FROM tags ORDER BY name")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_memory_tags(self, memory_id: int) -> List[str]:
        """
        Get all tags for a specific memory.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            List of tag names
        """
        self.cursor.execute("""
            SELECT t.name
            FROM tags t
            JOIN memory_tags mt ON t.id = mt.tag_id
            WHERE mt.memory_id = ?
            ORDER BY t.name
        """, (memory_id,))
        return [row[0] for row in self.cursor.fetchall()]
    
    def search_by_vector(self, 
                        query_vector: List[float], 
                        limit: int = 10,
                        min_importance: float = 0.0) -> List[MemoryItem]:
        """
        Search memories by vector similarity using cosine similarity.
        Only returns memories that have vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            min_importance: Minimum importance threshold
            
        Returns:
            List of MemoryItem objects with similarity scores in metadata
        """
        # Get all memories with vectors
        self.cursor.execute("""
            SELECT id, content, timestamp, importance,
                   access_count, last_accessed, metadata, vector, links
            FROM memories
            WHERE vector IS NOT NULL AND importance >= ?
        """, (min_importance,))
        
        results = []
        for row in self.cursor.fetchall():
            memory_vector = json.loads(row[7])
            if memory_vector:
                similarity = self._cosine_similarity(query_vector, memory_vector)
                metadata = json.loads(row[6]) if row[6] else {}
                metadata['similarity'] = similarity  # Add similarity to metadata
                
                memory = MemoryItem(
                    id=row[0],
                    content=row[1],
                    timestamp=row[2],
                    importance=row[3],
                    access_count=row[4],
                    last_accessed=row[5],
                    metadata=metadata,
                    vector=memory_vector,
                    links=json.loads(row[8]) if row[8] else None,
                    tags=self.get_memory_tags(row[0])
                )
                results.append(memory)
        
        # Sort by similarity
        results.sort(key=lambda x: x.metadata.get('similarity', 0), reverse=True)
        
        # Update access statistics for top results
        for result in results[:limit]:
            self._update_access(result.id)
        
        return results[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors using numpy.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1, higher is more similar)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        self.cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM tags")
        total_tags = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT AVG(importance) FROM memories")
        avg_importance = self.cursor.fetchone()[0] or 0.0
        
        self.cursor.execute("""
            SELECT SUM(access_count) FROM memories
        """)
        total_accesses = self.cursor.fetchone()[0] or 0
        
        return {
            "total_memories": total_memories,
            "total_tags": total_tags,
            "average_importance": round(avg_importance, 3),
            "total_accesses": total_accesses,
            "database_size": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        }
    
    def _update_access(self, memory_id: int):
        """Internal method to update access statistics."""
        self.cursor.execute("""
            UPDATE memories
            SET access_count = access_count + 1,
                last_accessed = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (memory_id,))
        self.conn.commit()
    
    def _format_results(self, rows: List[Tuple]) -> List[MemoryItem]:
        """Format database rows into MemoryItem objects."""
        results = []
        for row in rows:
            memory = MemoryItem(
                id=row[0],
                content=row[1],
                timestamp=row[2],
                importance=row[3],
                access_count=row[4],
                last_accessed=row[5],
                metadata=json.loads(row[6]) if row[6] else None,
                vector=json.loads(row[7]) if row[7] else None,
                links=json.loads(row[8]) if row[8] else None,
                tags=self.get_memory_tags(row[0])
            )
            results.append(memory)
        return results
    
    def set_links(self, memory_id: int, links: List[Tuple[int, float]]) -> bool:
        """
        Set the complete list of links for a memory (replaces existing links).
        
        Args:
            memory_id: Source memory ID
            links: List of (memory_id, weight) tuples
            
        Returns:
            True if successful, False if memory not found
        """
        return self.update_memory(memory_id, links=links)
    
    def get_linked_memories(self, memory_id: int, min_weight: float = 0.0) -> List[MemoryItem]:
        """
        Get all memories linked from a given memory.
        
        Args:
            memory_id: Source memory ID
            min_weight: Minimum link weight threshold
            
        Returns:
            List of linked MemoryItem objects with link weights in metadata
        """
        memory = self.get_memory_by_id(memory_id)
        if not memory or not memory.links:
            return []
        
        results = []
        for link_id, weight in memory.links:
            if weight >= min_weight:
                linked_memory = self.get_memory_by_id(link_id)
                if linked_memory:
                    # Add link_weight to metadata
                    if not linked_memory.metadata:
                        linked_memory.metadata = {}
                    linked_memory.metadata['link_weight'] = weight
                    results.append(linked_memory)
        
        # Sort by weight descending
        results.sort(key=lambda x: x.metadata.get('link_weight', 0), reverse=True)
        return results
    
    def get_backlinks(self, memory_id: int) -> List[MemoryItem]:
        """
        Get all memories that link to a given memory (backlinks).
        
        Args:
            memory_id: Target memory ID
            
        Returns:
            List of MemoryItem objects that link to this memory, with link weights in metadata
        """
        self.cursor.execute("""
            SELECT id, content, timestamp, importance,
                   access_count, last_accessed, metadata, vector, links
            FROM memories
            WHERE links IS NOT NULL
        """)
        
        results = []
        for row in self.cursor.fetchall():
            links = json.loads(row[8]) if row[8] else []
            for link_id, weight in links:
                if link_id == memory_id:
                    metadata = json.loads(row[6]) if row[6] else {}
                    metadata['link_weight'] = weight
                    
                    memory = MemoryItem(
                        id=row[0],
                        content=row[1],
                        timestamp=row[2],
                        importance=row[3],
                        access_count=row[4],
                        last_accessed=row[5],
                        metadata=metadata,
                        vector=json.loads(row[7]) if row[7] else None,
                        links=links,
                        tags=self.get_memory_tags(row[0])
                    )
                    results.append(memory)
                    break
        
        # Sort by weight descending
        results.sort(key=lambda x: x.metadata.get('link_weight', 0), reverse=True)
        return results
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Memory database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def print_memory(mem: MemoryItem, show_full: bool = True):
    """Pretty print a memory."""
    print(f"\n{'='*60}")
    print(f"ID: {mem.id}")
    print(f"Content: {mem.content}")
    if show_full:
        print(f"Importance: {mem.importance:.2f}")
        print(f"Timestamp: {mem.timestamp}")
        print(f"Access Count: {mem.access_count}")
        print(f"Last Accessed: {mem.last_accessed or 'Never'}")
        print(f"Tags: {', '.join(mem.tags) if mem.tags else 'None'}")
        if mem.metadata and mem.metadata.get('similarity') is not None:
            print(f"Similarity: {mem.metadata['similarity']:.4f}")
        if mem.metadata and mem.metadata.get('link_weight') is not None:
            print(f"Link Weight: {mem.metadata['link_weight']:.2f}")
        if mem.vector:
            print(f"Vector: [{len(mem.vector)} dimensions]")
        if mem.links:
            print(f"Links: {len(mem.links)} outgoing link(s)")
            for link_id, weight in mem.links:
                print(f"  -> Memory {link_id} (weight: {weight:.2f})")
        if mem.metadata and len(mem.metadata) > 0:
            # Filter out our special keys for display
            display_meta = {k: v for k, v in mem.metadata.items() 
                          if k not in ['similarity', 'link_weight']}
            if display_meta:
                print(f"Metadata: {json.dumps(display_meta, indent=2)}")


def interactive_mode(db_path: str = "memory.db"):
    """Run interactive command-line interface for memory management."""
    print("="*60)
    print("Memory Database - Interactive Mode")
    print("="*60)
    print(f"Database: {db_path}")
    print("\nAvailable commands:")
    print("  add        - Add a new memory")
    print("  search     - Search memories (text/importance/time/tags)")
    print("  vsearch    - Vector similarity search")
    print("  get        - Get memory by ID")
    print("  update     - Update a memory")
    print("  delete     - Delete a memory")
    print("  setlinks   - Set links for a memory")
    print("  links      - Show linked memories")
    print("  backlinks  - Show memories linking to this one")
    print("  tags       - List all tags")
    print("  stats      - Show database statistics")
    print("  help       - Show this help message")
    print("  exit/quit  - Exit interactive mode")
    print("="*60)
    
    with MemoryDatabase(db_path) as db:
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if not command:
                    continue
                
                if command in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                
                elif command == "help":
                    print("\nAvailable commands:")
                    print("  add, search, vsearch, get, update, delete")
                    print("  setlinks, links, backlinks, tags, stats, help, exit/quit")
                
                elif command == "add":
                    content = input("Content: ").strip()
                    if not content:
                        print("Error: Content cannot be empty")
                        continue
                    
                    tags_input = input("Tags (comma-separated, optional): ").strip()
                    tags = [t.strip() for t in tags_input.split(",")] if tags_input else None
                    
                    importance_input = input("Importance (0.0-1.0, default 0.5): ").strip()
                    importance = float(importance_input) if importance_input else 0.5
                    
                    vector_input = input("Vector (comma-separated floats, optional): ").strip()
                    vector = [float(x.strip()) for x in vector_input.split(",")] if vector_input else None
                    
                    memory_id = db.add_memory(content, tags=tags, importance=importance, vector=vector)
                    print(f"✓ Memory added with ID: {memory_id}")
                
                elif command == "search":
                    query = input("Search query (optional, press Enter to skip): ").strip()
                    query = query if query else None
                    
                    limit_input = input("Limit (default 10): ").strip()
                    limit = int(limit_input) if limit_input else 10
                    
                    importance_input = input("Min importance (optional, 0.0-1.0): ").strip()
                    importance_threshold = float(importance_input) if importance_input else None
                    
                    tags_input = input("Filter by tags (comma-separated, optional): ").strip()
                    tags = [t.strip() for t in tags_input.split(",")] if tags_input else None
                    
                    time_input = input("Time window unix timestamps (start,end, optional): ").strip()
                    time_window = None
                    if time_input:
                        try:
                            start, end = time_input.split(',')
                            time_window = (int(start.strip()), int(end.strip()))
                        except ValueError:
                            print("Warning: Invalid time window format, ignoring")
                    
                    results = db.search_memories(
                        query=query,
                        limit=limit,
                        importance_threshold=importance_threshold,
                        tags=tags,
                        time_window=time_window
                    )
                    if results:
                        print(f"\nFound {len(results)} result(s):")
                        for mem in results:
                            print(f"\n[{mem.id}] [{mem.importance:.2f}] {mem.timestamp}")
                            print(f"  {mem.content}")
                            if mem.tags:
                                print(f"  Tags: {', '.join(mem.tags)}")
                    else:
                        print("No results found.")
                
                elif command == "vsearch":
                    vector_input = input("Query vector (comma-separated floats): ").strip()
                    if not vector_input:
                        print("Error: Vector cannot be empty")
                        continue
                    
                    try:
                        query_vector = [float(x.strip()) for x in vector_input.split(",")]
                    except ValueError:
                        print("Error: Invalid vector format")
                        continue
                    
                    limit_input = input("Limit (default 10): ").strip()
                    limit = int(limit_input) if limit_input else 10
                    
                    results = db.search_by_vector(query_vector, limit=limit)
                    if results:
                        print(f"\nFound {len(results)} result(s) with vectors:")
                        for mem in results:
                            sim = mem.metadata.get('similarity', 0) if mem.metadata else 0
                            print(f"\n[{mem.id}] [sim: {sim:.4f}] [{mem.importance:.2f}] {mem.content}")
                            if mem.tags:
                                print(f"  Tags: {', '.join(mem.tags)}")
                    else:
                        print("No results found.")
                
                elif command == "get":
                    id_input = input("Memory ID: ").strip()
                    if not id_input:
                        print("Error: ID cannot be empty")
                        continue
                    
                    memory = db.get_memory_by_id(int(id_input))
                    if memory:
                        print_memory(memory)
                    else:
                        print("Memory not found.")
                
                elif command == "update":
                    id_input = input("Memory ID: ").strip()
                    if not id_input:
                        print("Error: ID cannot be empty")
                        continue
                    
                    memory_id = int(id_input)
                    
                    # First, show current memory
                    current = db.get_memory_by_id(memory_id)
                    if not current:
                        print("Memory not found.")
                        continue
                    
                    print("\nCurrent memory:")
                    print_memory(current)
                    
                    print("\nEnter new values (leave empty to keep current):")
                    
                    content_input = input("Content: ").strip()
                    content = content_input if content_input else None
                    
                    importance_input = input("Importance (0.0-1.0): ").strip()
                    importance = float(importance_input) if importance_input else None
                    
                    tags_input = input("Tags (comma-separated): ").strip()
                    tags = [t.strip() for t in tags_input.split(",")] if tags_input else None
                    
                    success = db.update_memory(memory_id, content=content, 
                                              importance=importance, tags=tags)
                    if success:
                        print("✓ Memory updated")
                    else:
                        print("Failed to update memory")
                
                elif command == "delete":
                    id_input = input("Memory ID: ").strip()
                    if not id_input:
                        print("Error: ID cannot be empty")
                        continue
                    
                    memory_id = int(id_input)
                    
                    # Show memory before deletion
                    memory = db.get_memory_by_id(memory_id)
                    if memory:
                        print("\nMemory to delete:")
                        print(f"  {memory.content}")
                        confirm = input("Are you sure? (yes/no): ").strip().lower()
                        if confirm in ["yes", "y"]:
                            if db.delete_memory(memory_id):
                                print("✓ Memory deleted")
                            else:
                                print("Failed to delete memory")
                        else:
                            print("Deletion cancelled")
                    else:
                        print("Memory not found.")
                
                elif command == "tags":
                    tags = db.get_all_tags()
                    if tags:
                        print(f"\nAll tags ({len(tags)}):")
                        for tag in tags:
                            print(f"  - {tag}")
                    else:
                        print("No tags found.")
                
                elif command == "setlinks":
                    mem_id = input("Memory ID: ").strip()
                    if not mem_id:
                        print("Error: Memory ID is required")
                        continue
                    
                    print("Enter links in format: id,weight id,weight ...")
                    print("Example: 2,0.9 5,0.7 10,0.5")
                    links_input = input("Links: ").strip()
                    
                    if not links_input:
                        # Clear all links
                        if db.set_links(int(mem_id), []):
                            print(f"✓ All links cleared for memory {mem_id}")
                        else:
                            print("Failed to update links")
                        continue
                    
                    try:
                        links = []
                        for pair in links_input.split():
                            link_id, weight = pair.split(',')
                            links.append([int(link_id), float(weight)])
                        
                        if db.set_links(int(mem_id), links):
                            print(f"✓ Links set for memory {mem_id}: {len(links)} link(s)")
                        else:
                            print("Failed to set links (memory not found)")
                    except ValueError:
                        print("Error: Invalid format. Use: id,weight id,weight ...")
                
                elif command == "links":
                    id_input = input("Memory ID: ").strip()
                    if not id_input:
                        print("Error: ID cannot be empty")
                        continue
                    
                    min_weight = input("Minimum weight (default 0.0): ").strip()
                    min_weight = float(min_weight) if min_weight else 0.0
                    
                    linked = db.get_linked_memories(int(id_input), min_weight=min_weight)
                    if linked:
                        print(f"\nMemories linked from {id_input} ({len(linked)}):")
                        for mem in linked:
                            weight = mem.metadata.get('link_weight', 0) if mem.metadata else 0
                            print(f"\n[{mem.id}] [weight: {weight:.2f}] {mem.content}")
                    else:
                        print("No linked memories found.")
                
                elif command == "backlinks":
                    id_input = input("Memory ID: ").strip()
                    if not id_input:
                        print("Error: ID cannot be empty")
                        continue
                    
                    backlinks = db.get_backlinks(int(id_input))
                    if backlinks:
                        print(f"\nMemories linking to {id_input} ({len(backlinks)}):")
                        for mem in backlinks:
                            weight = mem.metadata.get('link_weight', 0) if mem.metadata else 0
                            print(f"\n[{mem.id}] [weight: {weight:.2f}] {mem.content}")
                    else:
                        print("No backlinks found.")
                
                elif command == "stats":
                    stats = db.get_statistics()
                    print("\nDatabase Statistics:")
                    print(f"  Total Memories: {stats['total_memories']}")
                    print(f"  Total Tags: {stats['total_tags']}")
                    print(f"  Average Importance: {stats['average_importance']:.3f}")
                    print(f"  Total Accesses: {stats['total_accesses']}")
                    print(f"  Database Size: {stats['database_size']:,} bytes")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n\nUse 'exit' or 'quit' to leave interactive mode")
            except ValueError as e:
                print(f"Error: Invalid input - {e}")
            except Exception as e:
                print(f"Error: {e}")


def run_example():
    """Run example usage demonstration."""
    # Create/open database
    with MemoryDatabase("memory.db") as db:
        print("\n=== Adding Memories ===")
        
        # Add some example memories with vectors
        id1 = db.add_memory(
            content="User prefers Python over JavaScript for backend development",
            tags=["preference", "programming"],
            importance=0.8,
            metadata={"source": "conversation", "context": "discussing tech stack"},
            vector=[0.8, 0.2, 0.9, 0.1, 0.7]  # Example embedding vector
        )
        print(f"Added memory {id1}")
        
        id2 = db.add_memory(
            content="User's favorite color is blue",
            tags=["preference", "personal"],
            importance=0.5,
            vector=[0.1, 0.9, 0.2, 0.8, 0.3]  # Different embedding
        )
        print(f"Added memory {id2}")
        
        id3 = db.add_memory(
            content="User is working on a voice assistant project called Spirit",
            tags=["project", "current"],
            importance=0.9,
            metadata={"project_name": "Spirit", "status": "in_progress"},
            vector=[0.7, 0.3, 0.8, 0.2, 0.6]  # Similar to id1
        )
        print(f"Added memory {id3}")
        
        # Set links between memories
        print("\n=== Setting Links ===")
        db.set_links(id3, [[id1, 0.9]])  # Spirit project is related to Python preference
        print(f"Set links for memory {id3}: [{id1}] (weight: 0.9)")
        db.set_links(id1, [[id3, 0.7]])  # Python preference relates back to project
        print(f"Set links for memory {id1}: [{id3}] (weight: 0.7)")
        
        print("\n=== Search: Text Query ===")
        results = db.search_memories(query="Python programming")
        for mem in results:
            print(f"- [{mem.importance:.2f}] {mem.content}")
        
        print("\n=== Search: By Importance ===")
        important = db.search_memories(importance_threshold=0.7, limit=5)
        for mem in important:
            print(f"- [{mem.importance:.2f}] {mem.content}")
        
        print("\n=== Search: By Tags ===")
        tagged = db.search_memories(tags=["programming"], limit=5)
        for mem in tagged:
            print(f"- {mem.content}")
        
        print("\n=== Search: Vector Similarity ===")
        query_vec = [0.75, 0.25, 0.85, 0.15, 0.65]  # Similar to programming-related memories
        vector_results = db.search_by_vector(query_vec, limit=3)
        for mem in vector_results:
            sim = mem.metadata.get('similarity', 0) if mem.metadata else 0
            print(f"- [sim: {sim:.4f}] [{mem.importance:.2f}] {mem.content}")
        
        print("\n=== All Tags ===")
        tags = db.get_all_tags()
        print(f"Tags: {', '.join(tags)}")
        
        print("\n=== Linked Memories ===")
        linked = db.get_linked_memories(id3)
        if linked:
            print(f"Memories linked from {id3}:")
            for mem in linked:
                weight = mem.metadata.get('link_weight', 0) if mem.metadata else 0
                print(f"  -> [{mem.id}] [weight: {weight:.2f}] {mem.content}")
        
        print("\n=== Statistics ===")
        stats = db.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory Retrieval System - SQLite-based memory storage and retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python memory_retrieval.py -i              # Start interactive mode
  python memory_retrieval.py -i -d custom.db # Use custom database
  python memory_retrieval.py                 # Run example demonstration
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive command-line interface"
    )
    
    parser.add_argument(
        "-d", "--database",
        type=str,
        default="memory.db",
        help="Database file path (default: memory.db)"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.database)
    else:
        run_example()
