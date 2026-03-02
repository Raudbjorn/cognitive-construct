/**
 * Storage error types for the unified SurrealDB storage layer.
 */

export type StorageErrorCode =
  | 'CONNECTION_FAILED'
  | 'SCHEMA_APPLY_FAILED'
  | 'QUERY_FAILED'
  | 'NOT_FOUND'
  | 'DUPLICATE_ENTRY'
  | 'EMBEDDING_FAILED'
  | 'NOT_INITIALIZED'
  | 'CLOSE_FAILED';

export class StorageError extends Error {
  readonly code: StorageErrorCode;
  readonly context?: Record<string, unknown>;

  constructor(code: StorageErrorCode, message: string, context?: Record<string, unknown>) {
    super(message);
    this.name = 'StorageError';
    this.code = code;
    this.context = context;
  }

  static connectionFailed(cause: unknown): StorageError {
    return new StorageError(
      'CONNECTION_FAILED',
      `SurrealDB connection failed: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause: String(cause) }
    );
  }

  static schemaFailed(cause: unknown): StorageError {
    return new StorageError(
      'SCHEMA_APPLY_FAILED',
      `Schema application failed: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause: String(cause) }
    );
  }

  static queryFailed(operation: string, cause: unknown): StorageError {
    return new StorageError(
      'QUERY_FAILED',
      `Query failed during ${operation}: ${cause instanceof Error ? cause.message : String(cause)}`,
      { operation, cause: String(cause) }
    );
  }

  static embeddingFailed(cause: unknown): StorageError {
    return new StorageError(
      'EMBEDDING_FAILED',
      `Embedding generation failed: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause: String(cause) }
    );
  }

  static notInitialized(): StorageError {
    return new StorageError(
      'NOT_INITIALIZED',
      'Storage not initialized. Use SurrealStorage.create() to create an instance.'
    );
  }
}
