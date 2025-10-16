# Memory Retrieval System

A SQLite-based memory storage and retrieval system with vector similarity search support.

## Features

- ✅ **Text-based storage** with full-text search (SQLite FTS5)
- ✅ **Vector embeddings** support (variable dimensions)
- ✅ **Tag-based organization** for categorization
- ✅ **Importance scoring** (0.0 to 1.0)
- ✅ **Metadata storage** (JSON format)
- ✅ **Access tracking** (count and timestamps)
- ✅ **Vector similarity search** (cosine similarity)
- ✅ **Memory linking** with weights (graph connections)
- ✅ **Schema versioning** with automatic migration
- ✅ **Interactive CLI** for easy management

## Quick Start

### Interactive Mode
```bash
python3 memory_retrieval.py -i
```

### Using a Custom Database
```bash
python3 memory_retrieval.py -i -d my_memories.db
```

### Run Example Demo
```bash
python3 memory_retrieval.py
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `add` | Add a new memory (with optional vector) |
| `search` | Search memories with filters (text/importance/time/tags) |
| `vsearch` | Vector similarity search |
| `get` | Get specific memory by ID |
| `update` | Update existing memory |
| `delete` | Delete memory (with confirmation) |
| `setlinks` | Set all links for a memory (replaces existing) |
| `links` | Show outgoing links from a memory |
| `backlinks` | Show incoming links to a memory |
| `tags` | List all tags |
| `stats` | Show database statistics |
| `help` | Show command list |
| `exit/quit` | Exit interactive mode |

## Python API Usage

```python
from memory_retrieval import MemoryDatabase

# Create/open database
with MemoryDatabase("memory.db") as db:
    # Add a memory with vector and links
    memory_id = db.add_memory(
        content="User prefers Python for backend",
        tags=["preference", "programming"],
        importance=0.8,
        vector=[0.1, 0.5, 0.9, 0.2, 0.7],  # Any dimension
        links=[[2, 0.9], [5, 0.7]],  # Link to memories 2 and 5
        metadata={"source": "conversation"}
    )
    
    # Search with multiple filters
    
    # Text search
    results = db.search_memories(query="Python", limit=5)
    
    # Search by importance only
    important = db.search_memories(importance_threshold=0.7, limit=10)
    
    # Search by tags
    tagged = db.search_memories(tags=["programming", "AI"], limit=10)
    
    # Search with time window (unix timestamps)
    import time
    week_ago = int(time.time()) - (7 * 24 * 60 * 60)
    now = int(time.time())
    recent = db.search_memories(time_window=(week_ago, now), limit=10)
    
    # Combined filters
    results = db.search_memories(
        query="Python",
        importance_threshold=0.5,
        tags=["programming"],
        limit=5
    )
    
    # Vector similarity search
    query_vector = [0.15, 0.55, 0.85, 0.25, 0.65]
    similar = db.search_by_vector(query_vector, limit=5)
    for mem in similar:
        print(f"Similarity: {mem['similarity']:.4f}")
        print(f"Content: {mem['content']}")
    
    # Update memory
    db.update_memory(
        memory_id, 
        importance=0.9,
        vector=[0.2, 0.6, 0.95, 0.3, 0.8]
    )
    
    # Set links (replaces all existing links)
    db.set_links(memory_id=1, links=[[3, 0.8], [5, 0.6]])
    linked = db.get_linked_memories(1)  # Get memories linked from ID 1
    backlinks = db.get_backlinks(3)  # Get memories linking to ID 3
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Total memories: {stats['total_memories']}")
```

## Database Schema

### memories table
- `id` - Primary key
- `content` - Memory text (indexed for FTS)
- `timestamp` - Creation timestamp
- `importance` - Score (0.0 to 1.0)
- `access_count` - Number of times accessed
- `last_accessed` - Last access timestamp
- `metadata` - JSON metadata
- `vector` - JSON array of floats (variable dimension)
- `links` - JSON array of [id, weight] pairs for linking to other memories
- `created_at` - Creation timestamp

### tags table
- `id` - Primary key
- `name` - Unique tag name

### memory_tags table
- `memory_id` - Foreign key to memories
- `tag_id` - Foreign key to tags

### schema_info table
- `key` - Configuration key (e.g., "version")
- `value` - Configuration value

## Vector Embeddings

The system supports storing embedding vectors of **any dimension**. Vectors are stored as JSON arrays and can be used for semantic similarity search.

### Similarity Calculation
Uses **cosine similarity** with numpy (range: 0.0 to 1.0, higher = more similar)

**Requirements:**
```bash
pip install numpy
```

### Example Vector Workflow
1. Generate embeddings from your favorite model (OpenAI, Sentence Transformers, etc.)
2. Store with memory: `db.add_memory(content, vector=embedding)`
3. Search: `results = db.search_by_vector(query_embedding)`

## Schema Versioning & Auto-Migration

The system uses a **declarative schema model** defined at the top of the code:

```python
SCHEMA_VERSION = 2
SCHEMA_MODEL = {
    "version": 2,
    "tables": {
        "memories": {...},
        "tags": {...},
        # etc.
    }
}
```

**Automatic Migration Features:**
- Detects schema version mismatch
- Automatically adds missing columns (e.g., `vector`, `links`)
- Preserves existing data during migration
- Tracks version in `schema_info` table

**Migration Log Example:**
```
Migrating database from v1 to v2...
  - Adding 'vector' column to memories table
  - Adding 'links' column to memories table
✓ Migration complete
```

## Memory Linking (Graph Structure)

Create knowledge graphs by linking related memories with weighted connections.

### Link Operations

```python
# Set links for a memory (replaces all existing links)
db.set_links(memory_id=1, links=[
    [3, 0.9],   # Link to memory 3 with weight 0.9
    [5, 0.7],   # Link to memory 5 with weight 0.7
    [10, 0.5]   # Link to memory 10 with weight 0.5
])

# Clear all links
db.set_links(memory_id=1, links=[])

# Get outgoing links
linked = db.get_linked_memories(memory_id=1, min_weight=0.5)
for mem in linked:
    print(f"-> {mem['content']} (weight: {mem['link_weight']})")

# Get incoming links (backlinks)
backlinks = db.get_backlinks(memory_id=3)
```

### Use Cases for Links
- **Context chains**: Link conversation turns
- **Topic relationships**: Connect related concepts
- **Causal connections**: Link cause-effect memories
- **Reference networks**: Build knowledge graphs
- **Importance propagation**: High-weight links to important memories

## Statistics

Get database stats:
```python
stats = db.get_statistics()
# Returns: total_memories, total_tags, average_importance, 
#          total_accesses, database_size
```

## Best Practices

1. **Normalize vectors** before storage for consistent similarity scores
2. Use **consistent dimensions** within your application
3. Set **appropriate importance** scores (0.8+ for critical memories)
4. Add **descriptive tags** for better organization
5. Use **metadata** for additional context (source, timestamp, etc.)
6. Use **unified search** with multiple filters for flexible querying
7. Use `set_links()` to replace all links at once (atomic operation)
8. Link weights typically range 0.0-1.0 for consistency
9. Combine filters: `search_memories(query="...", importance_threshold=0.7, tags=["..."])`

## Example: Integration with Embeddings

```python
from sentence_transformers import SentenceTransformer
from memory_retrieval import MemoryDatabase

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

with MemoryDatabase("memory.db") as db:
    # Add memory with embedding
    text = "User loves hiking in the mountains"
    vector = model.encode(text).tolist()
    db.add_memory(text, vector=vector, tags=["hobby"])
    
    # Search with query embedding
    query = "outdoor activities"
    query_vector = model.encode(query).tolist()
    results = db.search_by_vector(query_vector, limit=5)
```

## License

MIT
