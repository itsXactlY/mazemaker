-- ============================================================================
-- Neural Memory Adapter - SQL Server Schema
-- Complete schema for vector storage, graph operations, and spreading activation.
-- Compatible with SQL Server 2019+.
-- ============================================================================

SET NOCOUNT ON;
GO

-- ============================================================================
-- Database creation (adjust as needed)
-- ============================================================================
-- CREATE DATABASE NeuralMemory;
-- GO
-- USE NeuralMemory;
-- GO

-- ============================================================================
-- CLR Integration (required for custom cosine similarity function)
-- Must be enabled at server level before creating assemblies
-- ============================================================================
-- EXEC sp_configure 'clr enabled', 1;
-- RECONFIGURE;
-- GO

-- ============================================================================
-- Table: NeuralMemory
-- Stores embedding vectors as VARBINARY alongside JSON metadata.
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'NeuralMemory')
BEGIN
    CREATE TABLE NeuralMemory (
        id              BIGINT          NOT NULL PRIMARY KEY,
        vector_data     VARBINARY(MAX)  NOT NULL,              -- Raw float32 bytes
        vector_dim      INT             NOT NULL DEFAULT 0,    -- Dimensionality
        metadata_json   NVARCHAR(MAX)   NOT NULL DEFAULT N'{}',
        created_at      DATETIME2(3)    NOT NULL DEFAULT SYSUTCDATETIME(),
        updated_at      DATETIME2(3)    NOT NULL DEFAULT SYSUTCDATETIME(),
        access_count    BIGINT          NOT NULL DEFAULT 0,
        last_accessed   DATETIME2(3)    NULL,

        -- Check: metadata must be valid JSON
        CONSTRAINT CK_NeuralMemory_Metadata CHECK (ISJSON(metadata_json) = 1)
    );
END
GO

-- ============================================================================
-- Table: GraphNodes
-- Nodes in the associative memory graph.
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'GraphNodes')
BEGIN
    CREATE TABLE GraphNodes (
        node_id         BIGINT          NOT NULL PRIMARY KEY,
        node_type       VARCHAR(64)     NOT NULL DEFAULT 'concept',
        properties_json NVARCHAR(MAX)   NOT NULL DEFAULT N'{}',
        activation      REAL            NOT NULL DEFAULT 0.0,
        created_at      DATETIME2(3)    NOT NULL DEFAULT SYSUTCDATETIME(),
        updated_at      DATETIME2(3)    NOT NULL DEFAULT SYSUTCDATETIME(),

        CONSTRAINT CK_GraphNodes_Properties CHECK (ISJSON(properties_json) = 1)
    );
END
GO

-- ============================================================================
-- Table: GraphEdges
-- Weighted, typed edges between graph nodes.
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'GraphEdges')
BEGIN
    CREATE TABLE GraphEdges (
        edge_id         BIGINT          NOT NULL IDENTITY(1,1) PRIMARY KEY,
        from_node_id    BIGINT          NOT NULL,
        to_node_id      BIGINT          NOT NULL,
        edge_type       VARCHAR(64)     NOT NULL DEFAULT 'association',
        weight          REAL            NOT NULL DEFAULT 1.0,
        properties_json NVARCHAR(MAX)   NOT NULL DEFAULT N'{}',
        created_at      DATETIME2(3)    NOT NULL DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_GraphEdges_From FOREIGN KEY (from_node_id)
            REFERENCES GraphNodes(node_id) ON DELETE CASCADE,
        CONSTRAINT FK_GraphEdges_To FOREIGN KEY (to_node_id)
            REFERENCES GraphNodes(node_id) ON DELETE CASCADE,
        CONSTRAINT CK_GraphEdges_Weight CHECK (weight >= 0.0 AND weight <= 1.0),
        CONSTRAINT CK_GraphEdges_Props CHECK (ISJSON(properties_json) = 1),

        -- Prevent self-loops
        CONSTRAINT CK_GraphEdges_NoSelfLoop CHECK (from_node_id <> to_node_id)
    );
END
GO

-- ============================================================================
-- Indexes - NeuralMemory
-- ============================================================================
-- Primary lookup by ID is covered by PK clustered index.

-- Index for time-range queries (recent memories)
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_NeuralMemory_CreatedAt')
BEGIN
    CREATE INDEX IX_NeuralMemory_CreatedAt
        ON NeuralMemory (created_at DESC)
        INCLUDE (id, metadata_json);
END
GO

-- Index for access patterns (frequently accessed memories)
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_NeuralMemory_AccessCount')
BEGIN
    CREATE INDEX IX_NeuralMemory_AccessCount
        ON NeuralMemory (access_count DESC)
        INCLUDE (id, last_accessed);
END
GO

-- Index for metadata JSON queries (requires computed column for common filters)
-- Add a computed column for category extraction if needed:
-- ALTER TABLE NeuralMemory ADD category AS JSON_VALUE(metadata_json, '$.category');
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_NeuralMemory_UpdatedAt')
BEGIN
    CREATE INDEX IX_NeuralMemory_UpdatedAt
        ON NeuralMemory (updated_at DESC);
END
GO

-- ============================================================================
-- Indexes - GraphNodes
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_GraphNodes_Type')
BEGIN
    CREATE INDEX IX_GraphNodes_Type
        ON GraphNodes (node_type)
        INCLUDE (node_id, activation);
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_GraphNodes_Activation')
BEGIN
    CREATE INDEX IX_GraphNodes_Activation
        ON GraphNodes (activation DESC)
        INCLUDE (node_id, node_type);
END
GO

-- ============================================================================
-- Indexes - GraphEdges
-- ============================================================================
-- Index for outgoing edges (from_node -> neighbors)
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_GraphEdges_FromNode')
BEGIN
    CREATE INDEX IX_GraphEdges_FromNode
        ON GraphEdges (from_node_id)
        INCLUDE (to_node_id, edge_type, weight);
END
GO

-- Index for incoming edges (to_node -> reverse neighbors)
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_GraphEdges_ToNode')
BEGIN
    CREATE INDEX IX_GraphEdges_ToNode
        ON GraphEdges (to_node_id)
        INCLUDE (from_node_id, edge_type, weight);
END
GO

-- Composite index for edge type queries
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_GraphEdges_Type_Weight')
BEGIN
    CREATE INDEX IX_GraphEdges_Type_Weight
        ON GraphEdges (edge_type, weight DESC)
        INCLUDE (from_node_id, to_node_id);
END
GO

-- Unique constraint to prevent duplicate edges of the same type
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'UX_GraphEdges_Unique')
BEGIN
    CREATE UNIQUE INDEX UX_GraphEdges_Unique
        ON GraphEdges (from_node_id, to_node_id, edge_type);
END
GO

-- ============================================================================
-- CLR Assembly for Cosine Similarity (stub)
-- Requires SQL Server CLR integration enabled.
-- This is a placeholder; actual implementation would be a compiled .NET DLL.
-- ============================================================================
/*
-- Enable CLR if not already enabled
EXEC sp_configure 'clr enabled', 1;
RECONFIGURE;
GO

-- For trusted assemblies (SQL Server 2017+):
-- CREATE ASSEMBLY NeuralMemoryCLR
-- FROM 'C:\Assemblies\NeuralMemoryCLR.dll'
-- WITH PERMISSION_SET = SAFE;
-- GO

-- Or use certificate-based signing for production deployments.
*/

-- Scalar function stub: CosineSimilarity
-- In production, replace with CLR implementation or use the C++-computed similarity.
IF OBJECT_ID('dbo.CosineSimilarity', 'FN') IS NOT NULL
    DROP FUNCTION dbo.CosineSimilarity;
GO

CREATE FUNCTION dbo.CosineSimilarity(
    @vec_a VARBINARY(MAX),
    @vec_b VARBINARY(MAX)
)
RETURNS REAL
AS
BEGIN
    -- Stub implementation: returns 0.0
    -- Replace with CLR function that deserializes VARBINARY to float arrays
    -- and computes dot(a,b) / (||a|| * ||b||)
    --
    -- CLR code outline:
    --   float[] a = DeserializeVector(vec_a);
    --   float[] b = DeserializeVector(vec_b);
    --   float dot = 0, normA = 0, normB = 0;
    --   for (int i = 0; i < a.Length; i++) {
    --       dot += a[i] * b[i];
    --       normA += a[i] * a[i];
    --       normB += b[i] * b[i];
    --   }
    --   return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));

    RETURN 0.0;
END
GO

-- ============================================================================
-- Stored Procedure: SpreadingActivation
-- Implements recursive spreading activation using a recursive CTE.
-- Activates neighboring nodes with decay and propagates through the graph.
-- ============================================================================
IF OBJECT_ID('dbo.SpreadingActivation', 'P') IS NOT NULL
    DROP PROCEDURE dbo.SpreadingActivation;
GO

CREATE PROCEDURE dbo.SpreadingActivation
    @start_node_id  BIGINT,
    @decay_factor   REAL = 0.85,
    @threshold      REAL = 0.01,
    @max_depth      INT = 10
AS
BEGIN
    SET NOCOUNT ON;

    -- Temporary table to accumulate activation results
    CREATE TABLE #ActivationResults (
        node_id     BIGINT NOT NULL,
        node_type   VARCHAR(64),
        activation  REAL NOT NULL,
        depth       INT NOT NULL,
        PRIMARY KEY (node_id, depth)
    );

    -- Seed: the starting node gets full activation (1.0)
    INSERT INTO #ActivationResults (node_id, node_type, activation, depth)
    SELECT node_id, node_type, 1.0, 0
    FROM GraphNodes
    WHERE node_id = @start_node_id;

    -- Spreading activation using iterative expansion
    -- (Recursive CTEs cannot reference temp tables, so we use a loop)
    DECLARE @current_depth INT = 0;

    WHILE @current_depth < @max_depth
    BEGIN
        -- Compute activation for next depth level
        INSERT INTO #ActivationResults (node_id, node_type, activation, depth)
        SELECT
            e.to_node_id,
            n.node_type,
            SUM(ar.activation * e.weight * @decay_factor) AS propagated_activation,
            @current_depth + 1
        FROM #ActivationResults ar
        INNER JOIN GraphEdges e ON ar.node_id = e.from_node_id
        INNER JOIN GraphNodes n ON e.to_node_id = n.node_id
        WHERE ar.depth = @current_depth
          AND e.to_node_id NOT IN (
              SELECT node_id FROM #ActivationResults
          )
        GROUP BY e.to_node_id, n.node_type
        HAVING SUM(ar.activation * e.weight * @decay_factor) >= @threshold;

        -- Stop if no new nodes were activated
        IF @@ROWCOUNT = 0
            BREAK;

        SET @current_depth = @current_depth + 1;
    END

    -- Update activation levels in GraphNodes
    UPDATE gn
    SET gn.activation = ar.activation,
        gn.updated_at = SYSUTCDATETIME()
    FROM GraphNodes gn
    INNER JOIN (
        SELECT node_id, MAX(activation) AS activation
        FROM #ActivationResults
        GROUP BY node_id
    ) ar ON gn.node_id = ar.node_id;

    -- Return results
    SELECT
        ar.node_id,
        ar.node_type,
        ar.activation,
        ar.depth
    FROM #ActivationResults ar
    ORDER BY ar.activation DESC, ar.depth ASC;

    DROP TABLE #ActivationResults;
END
GO

-- ============================================================================
-- Stored Procedure: Consolidation
-- Merges weakly activated memories based on vector similarity.
-- Identifies pairs of vectors with high cosine similarity and merges them.
-- ============================================================================
IF OBJECT_ID('dbo.Consolidation', 'P') IS NOT NULL
    DROP PROCEDURE dbo.Consolidation;
GO

CREATE PROCEDURE dbo.Consolidation
    @merge_threshold REAL = 0.95
AS
BEGIN
    SET NOCOUNT ON;

    -- Find candidate pairs for merging
    -- This uses a self-join with similarity computation.
    -- For production, limit the search space with activation-based pre-filtering.

    DECLARE @merge_count INT = 0;

    -- Temporary table for merge candidates
    CREATE TABLE #MergeCandidates (
        keep_id     BIGINT NOT NULL,
        merge_id    BIGINT NOT NULL,
        similarity  REAL NOT NULL,
        PRIMARY KEY (keep_id, merge_id)
    );

    -- Find high-similarity pairs (using CLR function or application-computed similarity)
    -- Here we use a simplified approach: merge vectors with similar metadata patterns
    -- In production, the application would pre-compute similarities and call this proc
    -- with a table-valued parameter of pairs.

    -- Self-join to find pairs where one has higher access count (keep that one)
    INSERT INTO #MergeCandidates (keep_id, merge_id, similarity)
    SELECT
        nm1.id AS keep_id,
        nm2.id AS merge_id,
        -- Placeholder: actual similarity would be computed via CLR or passed in
        CASE
            WHEN nm1.metadata_json = nm2.metadata_json THEN 1.0
            ELSE 0.0
        END AS similarity
    FROM NeuralMemory nm1
    INNER JOIN NeuralMemory nm2
        ON nm1.id < nm2.id  -- avoid duplicate pairs
    WHERE nm1.access_count >= nm2.access_count
      AND nm1.metadata_json = nm2.metadata_json  -- simplified matching
      AND nm1.id <> nm2.id;

    -- Merge: update kept vector's metadata to include merged info
    UPDATE nm
    SET nm.metadata_json = JSON_MODIFY(
            JSON_MODIFY(nm.metadata_json, '$.merged_from',
                ISNULL(JSON_QUERY(nm.metadata_json, '$.merged_from'), '[]')),
            'append $.merged_from',
            mc.merge_id
        ),
        nm.updated_at = SYSUTCDATETIME()
    FROM NeuralMemory nm
    INNER JOIN #MergeCandidates mc ON nm.id = mc.keep_id;

    -- Delete merged vectors
    DELETE FROM NeuralMemory
    WHERE id IN (SELECT merge_id FROM #MergeCandidates);

    SET @merge_count = @@ROWCOUNT;

    -- Clean up orphaned graph nodes that referenced deleted vectors
    DELETE FROM GraphNodes
    WHERE node_id NOT IN (SELECT id FROM NeuralMemory)
      AND node_type = 'memory';

    -- Log consolidation results
    PRINT 'Consolidation complete: ' + CAST(@merge_count AS VARCHAR(10)) + ' vectors merged.';

    DROP TABLE #MergeCandidates;

    RETURN @merge_count;
END
GO

-- ============================================================================
-- Stored Procedure: InsertVector (convenience with upsert)
-- ============================================================================
IF OBJECT_ID('dbo.InsertVector', 'P') IS NOT NULL
    DROP PROCEDURE dbo.InsertVector;
GO

CREATE PROCEDURE dbo.InsertVector
    @id             BIGINT,
    @vector_data    VARBINARY(MAX),
    @vector_dim     INT,
    @metadata_json  NVARCHAR(MAX) = N'{}'
AS
BEGIN
    SET NOCOUNT ON;

    -- Upsert: insert or update if exists
    MERGE INTO NeuralMemory AS target
    USING (SELECT @id AS id) AS source
    ON target.id = source.id
    WHEN MATCHED THEN
        UPDATE SET
            vector_data = @vector_data,
            vector_dim = @vector_dim,
            metadata_json = @metadata_json,
            updated_at = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN
        INSERT (id, vector_data, vector_dim, metadata_json, created_at, updated_at)
        VALUES (@id, @vector_data, @vector_dim, @metadata_json, SYSUTCDATETIME(), SYSUTCDATETIME());
END
GO

-- ============================================================================
-- Stored Procedure: BulkInsertVectors (table-valued parameter)
-- ============================================================================
-- First create the TVP type
IF NOT EXISTS (SELECT * FROM sys.table_types WHERE name = 'VectorTableType')
BEGIN
    CREATE TYPE dbo.VectorTableType AS TABLE (
        id              BIGINT          NOT NULL,
        vector_data     VARBINARY(MAX)  NOT NULL,
        vector_dim      INT             NOT NULL DEFAULT 0,
        metadata_json   NVARCHAR(MAX)   NOT NULL DEFAULT N'{}'
    );
END
GO

IF OBJECT_ID('dbo.BulkInsertVectors', 'P') IS NOT NULL
    DROP PROCEDURE dbo.BulkInsertVectors;
GO

CREATE PROCEDURE dbo.BulkInsertVectors
    @vectors dbo.VectorTableType READONLY
AS
BEGIN
    SET NOCOUNT ON;

    -- Use MERGE for upsert semantics
    MERGE INTO NeuralMemory AS target
    USING @vectors AS source
    ON target.id = source.id
    WHEN MATCHED THEN
        UPDATE SET
            vector_data = source.vector_data,
            vector_dim = source.vector_dim,
            metadata_json = source.metadata_json,
            updated_at = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN
        INSERT (id, vector_data, vector_dim, metadata_json, created_at, updated_at)
        VALUES (source.id, source.vector_data, source.vector_dim,
                source.metadata_json, SYSUTCDATETIME(), SYSUTCDATETIME());

    RETURN @@ROWCOUNT;
END
GO

-- ============================================================================
-- Trigger: Update access tracking on NeuralMemory reads
-- ============================================================================
-- Note: SQL Server doesn't have per-row read triggers.
-- Access tracking is done at the application level via:
--   UPDATE NeuralMemory SET access_count = access_count + 1, last_accessed = SYSUTCDATETIME()
--   WHERE id = @id;
-- This is called from the application's get_vector method.

-- ============================================================================
-- Trigger: Auto-update updated_at on NeuralMemory modification
-- ============================================================================
IF OBJECT_ID('dbo.TR_NeuralMemory_UpdatedAt', 'TR') IS NOT NULL
    DROP TRIGGER dbo.TR_NeuralMemory_UpdatedAt;
GO

CREATE TRIGGER TR_NeuralMemory_UpdatedAt
ON NeuralMemory
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE nm
    SET nm.updated_at = SYSUTCDATETIME()
    FROM NeuralMemory nm
    INNER JOIN inserted i ON nm.id = i.id;
END
GO

-- ============================================================================
-- Trigger: Auto-update updated_at on GraphNodes modification
-- ============================================================================
IF OBJECT_ID('dbo.TR_GraphNodes_UpdatedAt', 'TR') IS NOT NULL
    DROP TRIGGER dbo.TR_GraphNodes_UpdatedAt;
GO

CREATE TRIGGER TR_GraphNodes_UpdatedAt
ON GraphNodes
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE gn
    SET gn.updated_at = SYSUTCDATETIME()
    FROM GraphNodes gn
    INNER JOIN inserted i ON gn.node_id = i.node_id;
END
GO

-- ============================================================================
-- View: ActiveMemories (frequently accessed memories)
-- ============================================================================
IF OBJECT_ID('dbo.ActiveMemories', 'V') IS NOT NULL
    DROP VIEW dbo.ActiveMemories;
GO

CREATE VIEW dbo.ActiveMemories
AS
SELECT TOP 1000
    id,
    vector_dim,
    metadata_json,
    access_count,
    last_accessed,
    created_at,
    updated_at
FROM NeuralMemory
WHERE access_count > 0
ORDER BY access_count DESC, last_accessed DESC;
GO

-- ============================================================================
-- View: GraphSummary (graph statistics)
-- ============================================================================
IF OBJECT_ID('dbo.GraphSummary', 'V') IS NOT NULL
    DROP VIEW dbo.GraphSummary;
GO

CREATE VIEW dbo.GraphSummary
AS
SELECT
    (SELECT COUNT(*) FROM GraphNodes) AS total_nodes,
    (SELECT COUNT(*) FROM GraphEdges) AS total_edges,
    (SELECT COUNT(DISTINCT node_type) FROM GraphNodes) AS node_types,
    (SELECT COUNT(DISTINCT edge_type) FROM GraphEdges) AS edge_types,
    (SELECT AVG(activation) FROM GraphNodes WHERE activation > 0) AS avg_activation,
    (SELECT AVG(weight) FROM GraphEdges) AS avg_edge_weight;
GO

-- ============================================================================
-- Schema creation complete
-- ============================================================================
PRINT 'Neural Memory Adapter schema created successfully.';
GO
