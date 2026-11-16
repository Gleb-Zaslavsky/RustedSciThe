//! Bounded, process-safe filesystem lifecycle for worker-owned AOT entries.

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_SCHEMA_VERSION: u32 = 1;
const ENTRIES_DIR: &str = "entries";
const LEASES_DIR: &str = ".leases";

#[derive(Debug, thiserror::Error)]
pub enum WorkerCacheError {
    #[error("worker AOT cache I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("worker AOT cache key must be a 64-character lowercase hex digest")]
    InvalidKey,
    #[error("worker AOT cache entry manifest is invalid at {path}: {reason}")]
    InvalidManifest { path: PathBuf, reason: String },
    #[error("worker AOT cache manifest key mismatch at {path}")]
    KeyMismatch { path: PathBuf },
    #[error("worker AOT cache size limit must be greater than zero")]
    InvalidSizeLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvictionReport {
    pub removed_entries: usize,
    pub removed_bytes: u64,
    pub over_limit_bytes: u64,
    pub locked_entries_skipped: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    schema_version: u32,
    cache_key: String,
    provenance: serde_json::Value,
    created_unix_ms: u128,
    last_access_unix_ms: u128,
    completed_jobs: u64,
    last_job_status: Option<String>,
}

/// A leased cache entry. The lease prevents another process from evicting the
/// directory while a compiler or dynamically loaded artifact is using it.
pub struct WorkerArtifactCacheEntry {
    cache_root: PathBuf,
    entries_root: PathBuf,
    entry_path: PathBuf,
    cache_key: String,
    lease: Option<File>,
    was_present: bool,
    max_bytes: u64,
}

impl WorkerArtifactCacheEntry {
    pub fn acquire(
        cache_root: &Path,
        cache_key: &str,
        provenance: serde_json::Value,
        max_bytes: u64,
    ) -> Result<Self, WorkerCacheError> {
        if max_bytes == 0 {
            return Err(WorkerCacheError::InvalidSizeLimit);
        }
        if cache_key.len() != 64
            || !cache_key
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(WorkerCacheError::InvalidKey);
        }
        let entries_root = cache_root.join(ENTRIES_DIR);
        let leases_root = entries_root.join(LEASES_DIR);
        fs::create_dir_all(&leases_root)?;
        let entries_root = fs::canonicalize(entries_root)?;
        let entry_path = entries_root.join(cache_key);
        let lease_path = leases_root.join(format!("{cache_key}.lock"));
        let lease = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lease_path)?;
        FileExt::lock_exclusive(&lease)?;

        let was_present = entry_path.exists();
        if was_present {
            let metadata = fs::symlink_metadata(&entry_path)?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                return Err(WorkerCacheError::InvalidManifest {
                    path: entry_path,
                    reason: "cache entry must be a real directory, not a symlink or file".into(),
                });
            }
            let canonical_entry = fs::canonicalize(&entry_path)?;
            if !canonical_entry.starts_with(&entries_root) {
                return Err(WorkerCacheError::InvalidManifest {
                    path: entry_path,
                    reason: "cache entry resolves outside the worker-owned entries root".into(),
                });
            }
        }
        fs::create_dir_all(&entry_path)?;
        let manifest_path = entry_path.join("rst-worker-cache.json");
        let now = now_unix_ms();
        let mut manifest = if manifest_path.exists() {
            let bytes = fs::read(&manifest_path)?;
            let manifest: CacheManifest = serde_json::from_slice(&bytes).map_err(|error| {
                WorkerCacheError::InvalidManifest {
                    path: manifest_path.clone(),
                    reason: error.to_string(),
                }
            })?;
            if manifest.schema_version != CACHE_SCHEMA_VERSION || manifest.cache_key != cache_key {
                return Err(WorkerCacheError::KeyMismatch {
                    path: manifest_path,
                });
            }
            manifest
        } else {
            CacheManifest {
                schema_version: CACHE_SCHEMA_VERSION,
                cache_key: cache_key.to_owned(),
                provenance: provenance.clone(),
                created_unix_ms: now,
                last_access_unix_ms: now,
                completed_jobs: 0,
                last_job_status: None,
            }
        };
        if manifest.provenance != provenance {
            return Err(WorkerCacheError::KeyMismatch {
                path: manifest_path,
            });
        }
        manifest.last_access_unix_ms = now;
        write_manifest(&manifest_path, &manifest)?;

        Ok(Self {
            cache_root: cache_root.to_path_buf(),
            entries_root,
            entry_path,
            cache_key: cache_key.to_owned(),
            lease: Some(lease),
            was_present,
            max_bytes,
        })
    }

    pub fn path(&self) -> &Path {
        &self.entry_path
    }

    /// True means this identity directory existed before this request; it is
    /// not a claim that every solver-level generated artifact was a hit.
    pub fn entry_was_present(&self) -> bool {
        self.was_present
    }

    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }

    pub fn finish(mut self, job_status: &str) -> Result<EvictionReport, WorkerCacheError> {
        let manifest_path = self.entry_path.join("rst-worker-cache.json");
        let bytes = fs::read(&manifest_path)?;
        let mut manifest: CacheManifest =
            serde_json::from_slice(&bytes).map_err(|error| WorkerCacheError::InvalidManifest {
                path: manifest_path.clone(),
                reason: error.to_string(),
            })?;
        manifest.last_access_unix_ms = now_unix_ms();
        manifest.completed_jobs = manifest.completed_jobs.saturating_add(1);
        manifest.last_job_status = Some(job_status.to_owned());
        write_manifest(&manifest_path, &manifest)?;
        if let Some(lease) = self.lease.take() {
            FileExt::unlock(&lease)?;
            drop(lease);
        }
        evict_to_limit(&self.cache_root, &self.entries_root, self.max_bytes)
    }
}

fn evict_to_limit(
    cache_root: &Path,
    entries_root: &Path,
    max_bytes: u64,
) -> Result<EvictionReport, WorkerCacheError> {
    let lock_path = cache_root.join(".eviction.lock");
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    FileExt::lock_exclusive(&lock)?;

    let leases_root = entries_root.join(LEASES_DIR);
    let mut entries = Vec::new();
    let mut total_bytes = 0u64;
    for item in fs::read_dir(entries_root)? {
        let item = item?;
        let file_type = item.file_type()?;
        if !file_type.is_dir() || file_type.is_symlink() || item.file_name() == LEASES_DIR {
            continue;
        }
        let path = item.path();
        let canonical = match fs::canonicalize(&path) {
            Ok(path) if path.starts_with(entries_root) => path,
            _ => continue,
        };
        let size = directory_size_no_follow(&canonical)?;
        total_bytes = total_bytes.saturating_add(size);
        let manifest_path = canonical.join("rst-worker-cache.json");
        let last_access = fs::read(&manifest_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CacheManifest>(&bytes).ok())
            .map_or(0, |manifest| manifest.last_access_unix_ms);
        entries.push((last_access, canonical, size));
    }
    entries.sort_by_key(|entry| entry.0);

    let mut report = EvictionReport::default();
    for (_, path, size) in entries {
        if total_bytes <= max_bytes {
            break;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let lease_path = leases_root.join(format!("{name}.lock"));
        let lease = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lease_path)?;
        if FileExt::try_lock_exclusive(&lease).is_err() {
            report.locked_entries_skipped += 1;
            continue;
        }
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                total_bytes = total_bytes.saturating_sub(size);
                report.removed_entries += 1;
                report.removed_bytes = report.removed_bytes.saturating_add(size);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(WorkerCacheError::Io(error)),
        }
        FileExt::unlock(&lease)?;
    }
    report.over_limit_bytes = total_bytes.saturating_sub(max_bytes);
    FileExt::unlock(&lock)?;
    Ok(report)
}

fn write_manifest(path: &Path, manifest: &CacheManifest) -> Result<(), WorkerCacheError> {
    let temporary = path.with_extension("json.tmp");
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)?;
    serde_json::to_writer(&mut file, manifest)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    file.flush()?;
    file.sync_all()?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}

fn directory_size_no_follow(root: &Path) -> io::Result<u64> {
    let mut total = 0u64;
    for item in fs::read_dir(root)? {
        let item = item?;
        let kind = item.file_type()?;
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            total = total.saturating_add(directory_size_no_follow(&item.path())?);
        } else if kind.is_file() {
            total = total.saturating_add(item.metadata()?.len());
        }
    }
    Ok(total)
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::WorkerArtifactCacheEntry;
    use serde_json::json;
    use std::fs;

    #[test]
    fn worker_cache_evicts_least_recently_used_entries_to_configured_limit() {
        let root = tempfile::tempdir().unwrap();
        let first_key = "a".repeat(64);
        let second_key = "b".repeat(64);
        let first = WorkerArtifactCacheEntry::acquire(
            root.path(),
            &first_key,
            json!({"model":"first"}),
            4096,
        )
        .unwrap();
        fs::write(first.path().join("artifact.bin"), vec![1u8; 3000]).unwrap();
        first.finish("succeeded").unwrap();

        let second = WorkerArtifactCacheEntry::acquire(
            root.path(),
            &second_key,
            json!({"model":"second"}),
            4096,
        )
        .unwrap();
        fs::write(second.path().join("artifact.bin"), vec![2u8; 3000]).unwrap();
        let eviction = second.finish("succeeded").unwrap();
        assert_eq!(eviction.removed_entries, 1);
        assert!(!root.path().join("entries").join(first_key).exists());
        assert!(root.path().join("entries").join(second_key).exists());
    }

    #[test]
    fn existing_cache_key_requires_matching_provenance_manifest() {
        let root = tempfile::tempdir().unwrap();
        let key = "c".repeat(64);
        let entry =
            WorkerArtifactCacheEntry::acquire(root.path(), &key, json!({"model":"same"}), 1024)
                .unwrap();
        entry.finish("succeeded").unwrap();
        let entry =
            WorkerArtifactCacheEntry::acquire(root.path(), &key, json!({"model":"changed"}), 1024);
        assert!(entry.is_err());
    }
}
