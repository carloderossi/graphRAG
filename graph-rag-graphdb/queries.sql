// Cypher queries

// Check that Neo4j can see import files
CALL apoc.load.json("file:///ai_reg_semantic_index.json") YIELD value
RETURN value LIMIT 1;

/////////////////////////////////////////////////////////////////////////////////
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.id IS UNIQUE;

/////////////////////////////////////////////////////////////////////////////////
CALL apoc.load.json('file:///reg_kg_triples_v2.jsonl') YIELD value
WITH value
UNWIND value.entities AS e
WITH value, e,
     value.chunk_id AS chunk_id,
     value.source   AS source
MERGE (ent:Entity {id: chunk_id + ':' + e.local_id})
SET ent.name     = e.name,
    ent.type     = e.type,
    ent.chunk_id = chunk_id,
    ent.source   = source;

/////////////////////////////////////////////////////////////////////////////////
CALL apoc.load.json('file:///reg_kg_triples_v2.jsonl') YIELD value
WITH value
UNWIND value.relations AS r
WITH value, r,
     value.chunk_id AS chunk_id,
     value.source   AS source
MATCH (s:Entity {id: chunk_id + ':' + r.source_local_id})
MATCH (t:Entity {id: chunk_id + ':' + r.target_local_id})
MERGE (s)-[rel:RELATION {type: r.type}]->(t)
SET rel.chunk_id = chunk_id,
    rel.source   = source;

/////////////////////////////////////////////////////////////////////////////////
CALL apoc.load.json('file:///reg_kg_triples_v2.jsonl') YIELD value
WITH value
UNWIND value.relations AS r
WITH value, r,
     value.chunk_id AS chunk_id,
     value.source   AS source
MATCH (s:Entity {id: chunk_id + ':' + r.source_local_id})
MATCH (t:Entity {id: chunk_id + ':' + r.target_local_id})
CALL apoc.create.relationship(s, r.type, {chunk_id: chunk_id, source: source}, t) YIELD rel
RETURN count(rel);

/////////////////////////////////////////////////////////////////////////////////
CALL apoc.load.json('file:///ai_reg_semantic_index.json') YIELD value
WITH value.chunks AS chunks
UNWIND keys(chunks) AS chunkId
WITH chunkId, chunks[chunkId] AS c
MERGE (ch:Chunk {id: chunkId})
SET ch.text   = c.text,
    ch.source = c.source,
    ch.vec    = c.vec;   // Neo4j can store lists as properties


/////////////////////////////////////////////////////////////////////////////////
CALL apoc.load.json('file:///ai_reg_semantic_index.json') YIELD value
WITH value.chunks AS chunks
UNWIND keys(chunks) AS chunkId
WITH chunkId, chunks[chunkId] AS c
MATCH (ch:Chunk {id: chunkId})
SET ch.embedding = c.vec;

CALL apoc.load.json('file:///reg_kg_triples_v2.jsonl') YIELD value
WITH value
UNWIND value.entities AS e
MATCH (c:Chunk {id: value.chunk_id})
MATCH (ent:Entity {id: value.chunk_id + ':' + e.local_id})
MERGE (c)-[:MENTIONS]->(ent);

/////////////////////////////////////////////////////////////////////////////////
MATCH (ch:Chunk)
MATCH (e:Entity {chunk_id: ch.id})
MERGE (ch)-[:HAS_ENTITY]->(e);

// Counts
MATCH (e:Entity) RETURN count(e) AS entities;
MATCH ()-[r]->() RETURN count(r) AS relations;

// Sample
MATCH (e:Entity)-[r]->(f:Entity)
RETURN e, r, f
LIMIT 25;

// Isolated nodes
MATCH (e:Entity)
WHERE NOT (e)--()
RETURN count(e) AS isolated_entities;

MATCH (c:Chunk)
RETURN c.id, size(c.embedding) AS dim
LIMIT 5;

/////////////////////////////////////////////////////////////////////////////////
// Chunk + embedding importer
CALL apoc.load.json("file:///ai_reg_semantic_index.json") YIELD value

WITH value.chunks AS chunks
UNWIND keys(chunks) AS chunk_id
WITH chunk_id, chunks[chunk_id] AS c

MERGE (ch:Chunk {id: chunk_id})
SET ch.text = c.text,
    ch.source = c.source,
    ch.embedding = c.vec;
// Created 681 nodes, set 2,724 properties, added 681 labels
// Completed after 1,442 ms

/////////////////////////////////////////////////////////////////////////////////
// KG triples importer
CALL apoc.load.json("file:///reg_kg_triples_repaired.jsonl") YIELD value

// Create or update the Chunk
MERGE (chunk:Chunk {id: value.chunk_id})
SET chunk.text = value.text,
    chunk.source = value.source

// Create Entities
WITH value, chunk
UNWIND value.entities AS e
MERGE (ent:Entity {local_id: e.local_id, chunk_id: value.chunk_id})
SET ent.name = e.name,
    ent.type = e.type

// Create Relations
WITH value
UNWIND value.relations AS r
MATCH (src:Entity {local_id: r.source_local_id, chunk_id: value.chunk_id})
MATCH (tgt:Entity {local_id: r.target_local_id, chunk_id: value.chunk_id})
MERGE (src)-[rel:RELATION {type: r.type}]->(tgt)
SET rel.source = value.source;
// Created 2,094 nodes, created 1,307 relationships, set 16,694 properties, added 2,094 labels
// Completed after 7,956 ms

/////////////////////////////////////////////////////////////////////////////////
// Communities importer
CALL apoc.load.json("file:///ai_reg_semantic_index.json") YIELD value

// Iterate over each community object
UNWIND value.communities AS comm

// Create the Community node
MERGE (co:Community {id: comm.community_id})
SET co.title = comm.metadata.title,
    co.summary = comm.metadata.summary,
    co.findings = comm.metadata.findings,
    co.embedding = comm.community_embedding

// Pass both comm and co forward
WITH comm, co

// Link community → chunk members
UNWIND comm.member_ids AS mid
MATCH (c:Chunk {id: mid})
MERGE (co)-[:HAS_MEMBER]->(c);
// Created 681 relationships, set 180 properties
// Completed after 1,030 ms

/////////////////////////////////////////////////////////////////////////////////
// Heuristic linking
MATCH (c:Chunk), (e:Entity)
WHERE c.text CONTAINS e.name
MERGE (c)-[:MENTIONS]->(e);
// Created 39,651 relationships
// Completed after 2,638 ms

/////////////////////////////////////////////////////////////////////////////////
// Import validations
// 🧪 1. Node & Relationship Inventory
- MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC;
- MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC;

🧩 2. Check for orphaned entities
MATCH (e:Entity)
WHERE NOT (e)--()
RETURN e.local_id, e.name, e.type LIMIT 50;

🔗 3. Check for orphaned chunk
MATCH (c:Chunk)
WHERE NOT (c)--()
RETURN c.id, left(c.text, 200) LIMIT 20;

🧭 4. Validate community membership
MATCH (co:Community)
OPTIONAL MATCH (co)-[:HAS_MEMBER]->(ch)
RETURN co.id, count(ch) AS members ORDER BY members ASC;

🧠 5. Validate entity–relation consistency
MATCH ()-[r:RELATION]->()
WHERE r.type IS NULL OR r.source IS NULL
RETURN r LIMIT 20;

🧬 6. Spot duplicate entities (same name/type)
MATCH (e:Entity)
WITH e.name AS name, e.type AS type, count(*) AS c
WHERE c > 1
RETURN name, type, c ORDER BY c DESC;


🧱 7. Check embedding integrity
MATCH (c:Chunk)
RETURN size(c.embedding) AS dim, count(*) AS chunks
ORDER BY chunks DESC;

🧲 8. Quick semantic sanity check
MATCH (c1:Chunk {id: "EU_AI_Act_0"}), (c2:Chunk)
WHERE c1 <> c2
WITH c1, c2,
     gds.similarity.cosine(c1.embedding, c2.embedding) AS score
RETURN c2.id, score
ORDER BY score DESC LIMIT 10;


🎯 9. Visual sanity check
MATCH (c:Chunk)-[*1..2]-(x)
RETURN c, x LIMIT 200;



