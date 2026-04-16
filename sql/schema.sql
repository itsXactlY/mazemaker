-- ============================================================================
-- NEURAL MEMORY V2 - FINAL CLEAN MIGRATION
-- ============================================================================

USE NeuralMemory;
GO

-- ============================================================================
-- GLOBAL SAFETY
-- ============================================================================
SET XACT_ABORT ON;
SET NOCOUNT ON;

EXEC sp_getapplock 
    @Resource = 'NeuralMemory_Migration_V2',
    @LockMode = 'Exclusive',
    @LockOwner = 'Session',
    @Timeout = 60000;
GO

-- ============================================================================
-- PRE-MIGRATION DATA FIX (SELF-HEALING)
-- ============================================================================
PRINT 'Fixing vector_dim...';

UPDATE NeuralMemory
SET vector_dim = DATALENGTH(vector_data) / 4
WHERE (vector_dim IS NULL OR vector_dim <= 0)
  AND vector_data IS NOT NULL;

PRINT 'vector_dim repaired.';
GO

-- ============================================================================
-- CREATE TABLE (IDEMPOTENT)
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'NeuralMemory_v2')
BEGIN
    CREATE TABLE NeuralMemory_v2 (
        surrogate_id    BIGINT IDENTITY(1,1) NOT NULL,
        legacy_id       BIGINT NOT NULL,
        vector_data     VARBINARY(MAX) NOT NULL,
        vector_dim      INT NOT NULL,
        metadata_json   NVARCHAR(MAX) NOT NULL DEFAULT N'{}',
        created_at      DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
        updated_at      DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
        access_count    BIGINT NOT NULL DEFAULT 0,
        last_accessed   DATETIME2(3) NULL,
        content_hash    BINARY(32) NULL,
        category AS CAST(JSON_VALUE(metadata_json, '$.category') AS VARCHAR(128)) PERSISTED,

        CONSTRAINT PK_NeuralMemory_v2 PRIMARY KEY NONCLUSTERED (surrogate_id),
        CONSTRAINT CK_NeuralMemory_v2_Metadata CHECK (ISJSON(metadata_json) = 1),
        CONSTRAINT CK_NeuralMemory_v2_Dim CHECK (vector_dim > 0)
    );

    CREATE CLUSTERED INDEX CX_NeuralMemory_v2_Time
        ON NeuralMemory_v2 (created_at ASC, surrogate_id ASC)
        WITH (FILLFACTOR = 85);

    CREATE UNIQUE INDEX UX_NeuralMemory_v2_Legacy
        ON NeuralMemory_v2 (legacy_id);

    CREATE INDEX IX_NeuralMemory_v2_Category
        ON NeuralMemory_v2 (category)
        WHERE category IS NOT NULL;

    CREATE INDEX IX_NeuralMemory_v2_Access
        ON NeuralMemory_v2 (access_count DESC)
        INCLUDE (legacy_id, last_accessed);

    PRINT 'NeuralMemory_v2 created.';
END
GO

-- ============================================================================
-- DATA MIGRATION (FAST + RESUMABLE)
-- ============================================================================
PRINT 'Migrating NeuralMemory...';

DECLARE @batch INT = 5000;

WHILE 1 = 1
BEGIN
    INSERT INTO NeuralMemory_v2 (
        legacy_id, vector_data, vector_dim,
        metadata_json, created_at, updated_at,
        access_count, last_accessed
    )
    SELECT TOP (@batch)
        nm.id, nm.vector_data, nm.vector_dim,
        nm.metadata_json, nm.created_at,
        nm.updated_at, nm.access_count,
        nm.last_accessed
    FROM NeuralMemory nm
    WHERE nm.vector_dim > 0
      AND NOT EXISTS (
          SELECT 1 FROM NeuralMemory_v2 v2 WHERE v2.legacy_id = nm.id
      )
    ORDER BY nm.created_at ASC;

    IF @@ROWCOUNT = 0 BREAK;

    PRINT 'Batch migrated...';
END
GO

-- ============================================================================
-- VALIDATION
-- ============================================================================
PRINT 'Validating...';

SELECT 'NeuralMemory_old' AS tbl, COUNT(*) FROM NeuralMemory
UNION ALL
SELECT 'NeuralMemory_v2', COUNT(*) FROM NeuralMemory_v2;

SELECT COUNT(*) AS invalid_vectors
FROM NeuralMemory_v2
WHERE vector_dim <= 0;
GO

-- ============================================================================
-- TRIGGER (IDEMPOTENT)
-- ============================================================================
DROP TRIGGER IF EXISTS dbo.TR_NeuralMemory_UpdatedAt;
DROP TRIGGER IF EXISTS dbo.TR_NeuralMemory_v2_UpdatedAt;
GO

CREATE TRIGGER dbo.TR_NeuralMemory_v2_UpdatedAt
ON dbo.NeuralMemory_v2
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    IF UPDATE(vector_data) OR UPDATE(metadata_json) OR UPDATE(vector_dim)
    BEGIN
        UPDATE nm
        SET updated_at = SYSUTCDATETIME()
        FROM dbo.NeuralMemory_v2 nm
        INNER JOIN inserted i 
            ON nm.surrogate_id = i.surrogate_id;
    END
END;
GO

-- ============================================================================
-- RENAME (MANUELL AUSFÜHREN NACH VALIDATION!)
-- ============================================================================
/*
BEGIN TRANSACTION;

    EXEC sp_rename 'dbo.NeuralMemory',    'NeuralMemory_old';
    EXEC sp_rename 'dbo.NeuralMemory_v2', 'NeuralMemory';

COMMIT;
GO
*/

-- ============================================================================
-- OPTIONAL: INDEX REBUILD
-- ============================================================================
/*
ALTER INDEX ALL ON NeuralMemory_v2 REBUILD;
GO
*/

-- ============================================================================
-- DONE
-- ============================================================================
PRINT '======================================';
PRINT 'NeuralMemory V2 READY (NO ANN)';
PRINT 'Next: Validate + Rename';
PRINT '======================================';
GO
