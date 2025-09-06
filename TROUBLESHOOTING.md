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

### Error 3: GUI Connection Error

**Error Message:**

```
❌ Connection Error: HTTPConnectionPool(host='app', port=8000): Max retries exceeded with url: /api/v1/keyframe/search (Caused by NameResolutionError("<urllib3.connection.HTTPConnection object at 0x70cb52303ad0>: Failed to resolve 'app' ([Errno -2] Name or service not known)"))"
```

**Root Cause:**
The GUI container cannot connect to the app container because the **app container is not running** or is **stuck in a restart loop**.

**Why This Happens:**

1. **App container restart loop** due to line ending issues
2. **Permission errors** preventing container startup
3. **Long model loading time** - BEiT-3 model takes 60+ seconds to load
4. **Container networking issues** between GUI and app services

**Solution:**

1. **Check app container status:**

    ```bash
    docker ps | grep rag_app_ctn
    ```

2. **If container is restarting, check logs:**

    ```bash
    docker logs rag_app_ctn --tail 20
    ```

3. **Fix line endings if needed:**

    ```bash
    dos2unix app_entrypoint.sh
    ```

4. **Fix permission errors (Windows Docker):**
   Comment out chown commands in `app_entrypoint.sh`:

    ```bash
    # chown -R webappnonroot:webappnonroot /app/data_collection
    # chown -R 777 /app/data_collection
    ```

5. **Restart containers:**

    ```bash
    docker-compose restart app
    ```

6. **Wait for model loading:**
   The BEiT-3 model takes 60+ seconds to load. Wait for these log messages:

    ```
    ✓ BEiT3 model service initialization complete
    INFO: Application startup complete.
    ```

7. **Test API connectivity:**
    ```bash
    curl -X GET http://localhost:8000/health
    curl -X POST http://localhost:8000/api/v1/keyframe/search -H "Content-Type: application/json" -d '{"query": "test", "top_k": 1, "score_threshold": 0.0}'
    ```

---

### Error 4: Docker Permission Errors (Windows)

**Error Message:**

```
chown: changing ownership of '/app/data_collection/converter': Operation not permitted
chown: changing ownership of '/app/data_collection/keyframe': Operation not permitted
chown: changing ownership of '/app/data_collection': Operation not permitted
```

**Root Cause:**
Docker on Windows doesn't support `chown` operations on mounted volumes due to filesystem permission model differences.

**Why This Happens:**

1. **Windows filesystem** doesn't use Unix-style ownership
2. **Volume mounts** preserve Windows permissions
3. **Container scripts** expect Unix permission model
4. **Cross-platform compatibility** issues

**Solution:**

Edit `app_entrypoint.sh` and comment out the chown commands:

```bash
#!/bin/sh

set -e
echo 'Syncing dependencies...'
uv sync --frozen --no-dev --compile-bytecode --python=/usr/local/bin/python3.12
# chown -R webappnonroot:webappnonroot /app/data_collection
# chown -R 777 /app/data_collection
echo 'Starting development server...'
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Prevention:**
Use conditional chown commands that check the operating system or container environment.

---

### Error 5: GUI Search Read Timeout

**Error Message:**

```
❌ Connection Error: HTTPConnectionPool(host='app', port=8000): Read timed out. (read timeout=30)
```

**Root Cause:**
The GUI has a **30-second timeout** for API requests, but large embedding datasets can cause searches to take longer than this limit.

**Why This Happens:**

1. **Large embedding datasets** (e.g., 286,948 vs 23,198 embeddings) increase search time
2. **Hardcoded 30-second timeout** in GUI is too short for complex searches
3. **Milvus performance** degrades with larger vector collections
4. **Model processing time** increases with dataset size

**Solution:**

1. **Increase GUI timeout** in `gui/main.py` line 247:

    ```python
    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120  # Increased from 30 to 120 seconds
    )
    ```

2. **Restart GUI container** to apply changes:

    ```bash
    docker-compose restart app_gui
    ```

3. **Test search performance**:
    ```bash
    time curl -X POST http://localhost:8000/api/v1/keyframe/search -H "Content-Type: application/json" -d '{"query": "test", "top_k": 5}'
    ```

**Performance Expectations:**

-   **Small dataset (23K embeddings)**: 3-5 seconds
-   **Large dataset (286K embeddings)**: 8-15 seconds
-   **Very large datasets**: May need timeout > 120 seconds

---

### Error 6: Large Dataset Performance Issues

**Error Message:**

```
curl: (28) Operation timed out after 60008 milliseconds with 0 bytes received
```

**Root Cause:**
Large embedding datasets (1GB+ files) cause significant performance degradation in Milvus vector search operations.

**Why This Happens:**

1. **Vector search complexity** increases with dataset size
2. **Memory and I/O overhead** from large collections
3. **Milvus indexing** becomes less efficient with massive datasets
4. **Model processing** takes longer on CPU-only environments

**Symptoms:**

-   Container shows high disk I/O usage
-   API requests timeout even with extended timeouts
-   Searches that previously worked become unresponsive
-   Milvus container uses excessive memory

**Solutions:**

**Option 1: Reduce Dataset Size**

```bash
# Remove large embedding files
rm migration_data/large_embeddings.npy migration_data/large_metadata.npy

# Update migration script to use smaller files
# Edit migration/app_migration.sh line 28 with smaller filenames

# Re-run migration
rm migration/.migration_locked
docker-compose restart app_migration
```

**Option 2: Optimize Milvus Configuration**

```yaml
# In docker-compose.yml, add Milvus performance settings
environment:
    MILVUS_INDEX_TYPE: IVF_FLAT
    MILVUS_METRIC_TYPE: IP
    MILVUS_NLIST: 16384
```

**Option 3: Hardware Upgrade**

-   Use machines with more RAM (8GB+ recommended for large datasets)
-   Enable GPU acceleration if available
-   Use SSD storage for better I/O performance

**Performance Comparison:**

-   **23K embeddings**: ~3-4 seconds per search
-   **286K embeddings**: ~8-15 seconds per search
-   **1M+ embeddings**: May require specialized optimization

---

### Error 7: Embedding Dataset Management

**Common Scenarios:**

**Adding New Embeddings:**

1. Place new `.npy` files in `migration_data/`
2. Update `migration/app_migration.sh` with new filenames
3. Remove lock: `rm migration/.migration_locked`
4. Restart: `docker-compose restart app_migration`

**Removing/Replacing Embeddings:**

1. Delete unwanted files from `migration_data/`
2. Update migration script to point to remaining files
3. Remove lock and restart migration
4. Old embeddings are automatically dropped and replaced

**Reverting to Previous Dataset:**

1. Ensure old embedding files exist in `migration_data/`
2. Update migration script with old filenames
3. Re-run migration process

**No Docker Rebuild Required:**

-   Embeddings are stored in Milvus volumes, not Docker images
-   Only migration container needs restart
-   App container automatically uses new embeddings

---

## Complete Resolution Process

When you encounter system failures, follow this systematic approach:

### Step 1: Check Overall Container Status

```bash
docker compose ps
docker ps | grep -E "(rag_app_ctn|app_gui_ctn|app_migration_ctn)"
```

### Step 2: Check Container Logs

```bash
# Migration container
docker logs app_migration_ctn --tail 20

# App container
docker logs rag_app_ctn --tail 20

# GUI container
docker logs app_gui_ctn --tail 20
```

### Step 3: Fix Line Endings (Windows Users)

```bash
dos2unix migration/app_migration.sh
dos2unix app_entrypoint.sh
```

### Step 4: Fix Permission Errors (Windows Users)

Edit `app_entrypoint.sh` and comment out chown commands:

```bash
# chown -R webappnonroot:webappnonroot /app/data_collection
# chown -R 777 /app/data_collection
```

### Step 5: Verify Data Files

```bash
ls migration_data/
```

### Step 6: Update Filenames in Migration Script

Edit `migration/app_migration.sh` to match your actual data files.

### Step 7: Restart Containers

```bash
docker compose down && docker compose up -d
```

### Step 8: Wait for Model Loading

The BEiT-3 model takes 60+ seconds to load. Monitor app container logs:

```bash
docker logs rag_app_ctn -f
```

Wait for these success messages:

-   `✓ BEiT3 model service initialization complete`
-   `INFO: Application startup complete.`

### Step 9: Verify Full System Success

**Migration Success:**

```bash
docker logs app_migration_ctn | grep -E "(Data flushed|Collection loaded|Successfully injected)"
```

Should show:

-   `✅ Data flushed to disk`
-   `✅ Collection loaded for search`
-   `Successfully injected embeddings! Total entities: [number]`

**API Success:**

```bash
curl -X GET http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/keyframe/search -H "Content-Type: application/json" -d '{"query": "test", "top_k": 1}'
```

Should return JSON responses with keyframe results.

**GUI Success:**

-   Navigate to `http://localhost:8501`
-   Try a search query
-   Should return results without connection errors

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
# Check all container status
docker compose ps
docker ps | grep -E "(rag_app_ctn|app_gui_ctn|app_migration_ctn)"

# View recent logs for all services
docker logs app_migration_ctn --tail 20
docker logs rag_app_ctn --tail 20
docker logs app_gui_ctn --tail 20

# Check for restart loops
docker ps | grep "Restarting"

# Check line endings
file migration/app_migration.sh
file app_entrypoint.sh

# List data files
ls -la migration_data/

# Test API connectivity and functionality
curl -X GET http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/keyframe/search -H "Content-Type: application/json" -d '{"query": "test", "top_k": 1}'

# Test search performance with timing
time curl -X POST http://localhost:8000/api/v1/keyframe/search -H "Content-Type: application/json" -d '{"query": "person walking", "top_k": 5}'

# Test with extended timeout for large datasets
curl -X POST http://localhost:8000/api/v1/keyframe/search -H "Content-Type: application/json" -d '{"query": "test", "top_k": 10}' --max-time 120

# Check container resource usage
docker stats rag_app_ctn --no-stream
docker stats milvus-standalone --no-stream

# Check GUI accessibility
curl -X GET http://localhost:8501/
# Or open http://localhost:8501 in browser

# Monitor real-time logs (use Ctrl+C to stop)
docker logs rag_app_ctn -f

# Check model loading progress
docker logs rag_app_ctn | grep -E "(BEiT3|model|startup)"

# Verify migration data count
docker logs app_migration_ctn | grep "entities"

# Check current embedding files
ls -la migration_data/

# Monitor migration progress during re-run
docker logs app_migration_ctn -f
```

---

## Error Summary Reference

| Error Type                        | Key Symptoms                                                      | Primary Cause                        | Quick Fix                                 |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------------ | ----------------------------------------- |
| **Migration Script Not Found**    | `exec /app/migration/app_migration.sh: no such file or directory` | Windows CRLF line endings            | `dos2unix migration/app_migration.sh`     |
| **Migration Data File Not Found** | `FileNotFoundError: combined_30082025-*.npy`                      | Hardcoded filenames in script        | Update filenames in `app_migration.sh`    |
| **GUI Connection Error**          | `Connection Error: HTTPConnectionPool(host='app', port=8000)`     | App container not running/restarting | Fix line endings + restart containers     |
| **Docker Permission Errors**      | `chown: changing ownership: Operation not permitted`              | Windows filesystem incompatibility   | Comment out chown commands                |
| **App Container Restart Loop**    | Container status shows "Restarting"                               | Multiple issues: CRLF + permissions  | Apply all Windows fixes                   |
| **Model Loading Timeout**         | GUI connects but search fails                                     | BEiT-3 model still loading           | Wait 60+ seconds for model initialization |
| **GUI Search Read Timeout**       | `Read timed out. (read timeout=30)`                               | 30-second timeout too short          | Increase timeout to 120 seconds in GUI    |
| **Large Dataset Performance**     | `Operation timed out after 60+ seconds`                           | Large embedding dataset slow         | Reduce dataset size or optimize config    |
| **Embedding Management**          | Questions about adding/removing embeddings                        | Process not documented               | Follow embedding management workflow      |

### Most Common Issue Chains:

**Windows Setup Issues:**

1. **CRLF line endings** → Container won't start
2. **Permission errors** → Container restart loop
3. **GUI connection fails** → Can't access search API
4. **Model loading time** → Search appears broken initially

**Performance Issues (Large Datasets):**

1. **Large embeddings added** → Search becomes slow
2. **GUI timeout too short** → Read timeout errors
3. **Milvus performance degrades** → API timeouts
4. **User reduces dataset** → Performance restored

**Embedding Management Flow:**

1. **Add new embeddings** → Update migration script
2. **Remove migration lock** → Re-run migration
3. **Old embeddings replaced** → System uses new data
4. **Performance may vary** → Monitor and optimize

### Success Indicators:

✅ `docker ps` shows all containers "Up" and "healthy"  
✅ `curl http://localhost:8000/health` returns `{"status": "healthy"}`  
✅ `curl http://localhost:8000/api/v1/keyframe/search` returns search results  
✅ GUI at `http://localhost:8501` performs searches without errors
