# Maintaining SQLite migrations

Schema migrations live in this directory and are applied during `db.init_db()` by the runner in [`__init__.py`](./__init__.py). The ordered `MIGRATIONS` registry pairs each integer version with a migration function. Versioned modules such as [`v001_initial_schema.py`](./v001_initial_schema.py) contain the schema or data changes; [`helpers.py`](./helpers.py) holds shared migration helpers.

## Adding a migration

1. Choose the next unused version number and add a module named `vNNN_short_description.py`.
2. Define `migrate(connection: sqlite3.Connection) -> None`. Use the supplied connection for all changes.
3. Import the function in [`__init__.py`](./__init__.py) and append `(version, migrate_function)` to `MIGRATIONS`.
4. Add or update a regression test that initializes a temporary database representing the relevant starting schema. The upgrade/idempotency test is in [`test_durable_jobs.py`](../../tests/test_durable_jobs.py).

For example, the next module could be `v007_add_example_field.py`, registered after version 6. Never reuse, renumber, reorder, or edit a migration that may already have run in a deployed database; add a new version for follow-up changes.

## Transaction and compatibility rules

- The runner applies all pending migrations in version order in one `BEGIN IMMEDIATE` transaction and records each successful version in `schema_migrations`. A failure rolls back the transaction, including its version records. Do not commit, roll back, or open another database connection inside a migration.
- Use `connection.execute()` rather than `executescript()`, which can interfere with SQLite transaction boundaries. Bind data values as SQL parameters.
- Keep migrations safe for existing databases that may already have some schema changes but no migration history. Use `IF NOT EXISTS`, [`add_column_if_missing`](./helpers.py), or an equivalent guard where appropriate. Data backfills should also avoid duplicate effects.
- Keep every historical version in `MIGRATIONS`. The runner rejects recorded versions it does not recognize, preventing an older application from silently starting against a newer schema.

## Extraction checkpoints

Migration v005 creates `task_extraction_cache`. The task extractor stores successful chunk results and per-task context results there, keyed by meeting, a transcript/model/budget hash, and a stage key. A retry can therefore resume after the last successful model call; changing the transcript, configured model, or context budget starts a fresh cache. Old cache rows for that meeting are removed when a new transcript hash is processed. Bump `_CACHE_VERSION` in `task_extraction.py` when changing prompts, chunking/compaction behavior, or cached-result interpretation.

Run the focused database tests and lint checks after a migration change:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_durable_jobs.py'
.venv/bin/ruff check app/db.py app/db_migrations tests/test_durable_jobs.py
```
