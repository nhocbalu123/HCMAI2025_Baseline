# Troubleshooting Guide

## Common Docker Migration Errors

### Error 1: Migration Container Script Not Found

**Error Message:**

```
exec /app/migration/app_migration.sh: no such file or directory
service "app_migration" didn't complete successfully: exit 255
```

**Root Cause:**
This happens on **Windows systems** due to **line ending incompatibility**:

-   Windows uses CRLF (`\r\n`) line endings
-   Linux containers expect LF (`\n`) line endings
-   When Windows files are mounted into Linux containers, the CRLF characters cause the shebang (`#!/bin/sh`) to be misinterpreted
-   Linux treats the script as binary/invalid format instead of executable text

**Why This Happens:**

1. You're developing on Windows (CRLF line endings by default)
2. Git may not automatically convert line endings when cloning
3. Docker mounts Windows files directly into Linux containers
4. The Linux shell can't parse the Windows line endings

**Solution:**

```bash
# Convert line endings from Windows to Unix format
dos2unix migration/app_migration.sh
dos2unix app_entrypoint.sh

# Then restart containers
docker compose down && docker compose up -d
```

**Prevention:**
Add to your `.gitattributes` file:

```
*.sh text eol=lf
*.py text eol=lf
```

---

### Error 2: Migration Data File Not Found

**Error Message:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'migration_data/combined_30082025-1756585172_embeddings.npy'
```

**Root Cause:**
The migration script has **hardcoded filenames** with timestamps that don't match your actual data files.

**Why This Happens:**

1. Migration data files are generated with timestamps (e.g., `combined_04092025-1757005847_embeddings.npy`)
2. The migration script `app_migration.sh` has hardcoded old timestamp filenames
3. When you have newer data files, the script still looks for the old names
4. This creates a mismatch between expected vs actual filenames

**Solution:**

1. **Check your actual files:**

    ```bash
    ls migration_data/
    ```

2. **Update the migration script** with correct filenames:

    ```bash
    # Edit migration/app_migration.sh line 28
    # Change from:
    python migration/embedding_migration.py --emd-file-path "migration_data/combined_30082025-1756585172_embeddings.npy" --metadata-file-path "migration_data/combined_30082025-1756585172_metadata.npy"

    # To your actual filenames:
    python migration/embedding_migration.py --emd-file-path "migration_data/combined_04092025-1757005847_embeddings.npy" --metadata-file-path "migration_data/combined_04092025-1757005847_metadata.npy"
    ```

3. **Restart migration:**
    ```bash
    docker compose up -d
    ```

**Prevention:**

-   Use dynamic filename detection in migration scripts
-   Or use standardized filenames without timestamps

---

## Complete Resolution Process

When you encounter migration failures, follow this systematic approach:

### Step 1: Check Container Logs

```bash
docker logs app_migration_ctn
```

### Step 2: Fix Line Endings (Windows Users)

```bash
dos2unix migration/app_migration.sh
dos2unix app_entrypoint.sh
```

### Step 3: Verify Data Files

```bash
ls migration_data/
```

### Step 4: Update Filenames in Migration Script

Edit `migration/app_migration.sh` to match your actual data files.

### Step 5: Restart Containers

```bash
docker compose down && docker compose up -d
```

### Step 6: Verify Success

```bash
docker logs app_migration_ctn
docker compose ps
```

**Successful migration shows:**

-   `✅ Data flushed to disk`
-   `✅ Collection loaded for search`
-   `Successfully injected embeddings! Total entities: [number]`

---

## Why These Issues Are Common

### Windows Development Environment

-   **Mixed line endings** between Windows host and Linux containers
-   **Path differences** (`\` vs `/`)
-   **Permission models** differ between Windows and Unix

### Data File Management

-   **Timestamp-based naming** creates maintenance overhead
-   **Hardcoded paths** break when data regenerates
-   **Missing validation** between expected vs actual files

### Container Architecture

-   **Host file mounting** preserves original file characteristics
-   **Cross-platform compatibility** requires careful handling
-   **Build vs runtime** environment differences

---

## Best Practices

1. **Use .gitattributes** to enforce consistent line endings
2. **Implement dynamic file discovery** instead of hardcoded names
3. **Add validation checks** before running migrations
4. **Test on target platform** (Linux) if developing on Windows
5. **Use consistent file naming conventions** without timestamps in scripts

---

## Quick Diagnostic Commands

```bash
# Check container status
docker compose ps

# View recent logs
docker logs app_migration_ctn --tail 20

# Check line endings
file migration/app_migration.sh

# List data files
ls -la migration_data/

# Test API connectivity
curl http://localhost:8000/health
```
