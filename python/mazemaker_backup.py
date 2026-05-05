#!/usr/bin/env python3
"""
mazemaker_backup.py - Backup and recovery for Mazemaker database.
Protects against corruption from snapshot updates, concurrent access, etc.
"""
import os, sqlite3, time, glob, shutil
from pathlib import Path


class NeuralMemoryBackup:
    """Backup and recovery system for Mazemaker database."""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or str(Path.home() / ".mazemaker" / "engine" / "memory.db")
        self.backup_dir = Path.home() / ".mazemaker" / "engine" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = 10  # Keep last N backups
    
    def backup(self):
        """Create a safe backup using SQLite's online backup API."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"memory_{ts}.db"

        src = None
        dst = None
        try:
            src = sqlite3.connect(self.db_path)
            dst = sqlite3.connect(str(backup_path))
            with dst:
                src.backup(dst)
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            # Always release file descriptors / WAL locks, even if backup() raised.
            if src is not None:
                try: src.close()
                except Exception: pass
            if dst is not None:
                try: dst.close()
                except Exception: pass

        # Verify (separate try so a verification failure doesn't leak the
        # newly-created backup file's connection)
        conn = None
        try:
            conn = sqlite3.connect(str(backup_path))
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            if conn is not None:
                try: conn.close()
                except Exception: pass

        # Clean old backups
        self._clean_old_backups()

        return {"status": "ok", "path": str(backup_path), "memories": count,
                "size_kb": os.path.getsize(backup_path) / 1024}

    def restore(self, backup_path=None):
        """Restore from backup. Uses latest if none specified."""
        if backup_path is None:
            backups = sorted(glob.glob(str(self.backup_dir / "memory_*.db")))
            if not backups:
                return {"status": "error", "error": "No backups found"}
            backup_path = backups[-1]

        conn = None
        try:
            # Verify backup is valid
            conn = sqlite3.connect(backup_path)
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            if conn is not None:
                try: conn.close()
                except Exception: pass

        try:
            # Backup current before restore
            self.backup()

            # Copy backup to main DB
            shutil.copy2(backup_path, self.db_path)

            return {"status": "ok", "restored_from": backup_path, "memories": count}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def verify(self, path=None):
        """Verify database integrity."""
        path = path or self.db_path
        conn = None
        try:
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA integrity_check")
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            return {"status": "ok", "memories": count}
        except Exception as e:
            return {"status": "corrupted", "error": str(e)}
        finally:
            if conn is not None:
                try: conn.close()
                except Exception: pass

    def list_backups(self):
        """List all available backups."""
        backups = sorted(glob.glob(str(self.backup_dir / "memory_*.db")))
        result = []
        for b in backups:
            conn = None
            try:
                conn = sqlite3.connect(b)
                count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                result.append({
                    "path": b,
                    "memories": count,
                    "size_kb": os.path.getsize(b) / 1024,
                    "mtime": time.ctime(os.path.getmtime(b))
                })
            except Exception:
                # Narrowed from bare `except:` so KeyboardInterrupt/SystemExit
                # propagate. Corrupted DBs (sqlite3.DatabaseError, OSError) are
                # still caught and reported as `corrupted: True`.
                result.append({"path": b, "corrupted": True})
            finally:
                if conn is not None:
                    try: conn.close()
                    except Exception: pass
        return result
    
    def _clean_old_backups(self):
        """Keep only max_backups most recent backups."""
        backups = sorted(glob.glob(str(self.backup_dir / "memory_*.db")))
        while len(backups) > self.max_backups:
            old = backups.pop(0)
            os.remove(old)


if __name__ == "__main__":
    import sys
    bm = NeuralMemoryBackup()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "backup":
            print(json.dumps(bm.backup(), indent=2))
        elif cmd == "restore":
            path = sys.argv[2] if len(sys.argv) > 2 else None
            print(json.dumps(bm.restore(path), indent=2))
        elif cmd == "verify":
            print(json.dumps(bm.verify(), indent=2))
        elif cmd == "list":
            print(json.dumps(bm.list_backups(), indent=2))
        else:
            print(f"Usage: {sys.argv[0]} [backup|restore|verify|list]")
    else:
        # Default: backup
        import json
        print(json.dumps(bm.backup(), indent=2))
