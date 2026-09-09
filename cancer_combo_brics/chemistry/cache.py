"""Persistent and in-memory caching for functional group fragmentations."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Dict, List, Optional
from cancer_combo_brics.chemistry.functional_group_fragments import extract_functional_group_fragments


class FunctionalGroupCache:
    """Thread-safe persistent SQLite and in-memory cache for functional-group fragments."""

    def __init__(
        self,
        db_path: Optional[str] = "./data/fg_cache.sqlite",
        radius: int = 1,
        version_tag: str = "fg_v1",
    ):
        self.db_path = db_path
        self.radius = radius
        self.version_tag = version_tag
        self._mem_cache: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

        if self.db_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            self._init_db()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()


    def _init_db(self) -> None:
        conn = None
        try:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fg_cache_v1 (
                        smiles TEXT PRIMARY KEY,
                        radius INTEGER,
                        fragments TEXT
                    )
                    """
                )
        finally:
            if conn is not None:
                conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        if not self.db_path:
            raise RuntimeError("Database path is not set.")
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _clean_smiles(self, smiles: str) -> str:
        if not smiles or not isinstance(smiles, str) or not smiles.strip() or str(smiles).strip().lower() == "nan":
            return "C"
        return smiles.strip()

    def get(self, smiles: str) -> Optional[List[str]]:
        """Retrieve fragments from memory or SQLite cache."""
        smiles = self._clean_smiles(smiles)
        with self._lock:
            if smiles in self._mem_cache:
                return self._mem_cache[smiles]

        if not self.db_path or not os.path.exists(self.db_path):
            return None

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fragments FROM fg_cache_v1 WHERE smiles = ? AND radius = ?",
                (smiles, self.radius),
            )
            row = cursor.fetchone()
            if row:
                frags = json.loads(row[0])
                with self._lock:
                    self._mem_cache[smiles] = frags
                return frags
        except Exception:
            return None
        finally:
            if conn is not None:
                conn.close()

        return None

    def set(self, smiles: str, fragments: List[str]) -> None:
        """Store fragments into memory and SQLite cache."""
        smiles = self._clean_smiles(smiles)
        with self._lock:
            self._mem_cache[smiles] = fragments

        if not self.db_path:
            return

        conn = None
        try:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fg_cache_v1 (smiles, radius, fragments)
                    VALUES (?, ?, ?)
                    """,
                    (smiles, self.radius, json.dumps(fragments)),
                )
        except Exception:
            pass
        finally:
            if conn is not None:
                conn.close()

    def get_or_decompose(self, smiles: str) -> List[str]:
        """Fetch from cache or extract functional groups and cache."""
        smiles = self._clean_smiles(smiles)
        cached = self.get(smiles)
        if cached is not None:
            return cached

        frags = extract_functional_group_fragments(smiles, radius=self.radius)
        self.set(smiles, frags)
        return frags


    def get_many(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        """Batch retrieve fragments."""
        result: Dict[str, List[str]] = {}
        missing: List[str] = []

        with self._lock:
            for s in smiles_list:
                if s in self._mem_cache:
                    result[s] = self._mem_cache[s]
                else:
                    missing.append(s)

        if missing and self.db_path and os.path.exists(self.db_path):
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                chunk_size = 500
                for i in range(0, len(missing), chunk_size):
                    chunk = missing[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    params = list(chunk) + [self.radius]
                    cursor.execute(
                        f"SELECT smiles, fragments FROM fg_cache_v1 WHERE smiles IN ({placeholders}) AND radius = ?",
                        params,
                    )
                    for s_val, f_val in cursor.fetchall():
                        frags = json.loads(f_val)
                        result[s_val] = frags
                        with self._lock:
                            self._mem_cache[s_val] = frags
            except Exception:
                pass
            finally:
                if conn is not None:
                    conn.close()

        return result

    def preload_dataset_smiles(self, all_smiles: List[str]) -> int:
        """Pre-extracts and populates cache for a list of SMILES."""
        unique_smiles = list(set(s for s in all_smiles if s and isinstance(s, str)))
        cached_dict = self.get_many(unique_smiles)
        needed = [s for s in unique_smiles if s not in cached_dict]

        newly_cached = 0
        if self.db_path:
            conn = None
            try:
                conn = self._get_connection()
                with conn:
                    for s in needed:
                        try:
                            frags = extract_functional_group_fragments(s, radius=self.radius)
                            with self._lock:
                                self._mem_cache[s] = frags
                            conn.execute(
                                "INSERT OR REPLACE INTO fg_cache_v1 (smiles, radius, fragments) VALUES (?, ?, ?)",
                                (s, self.radius, json.dumps(frags)),
                            )
                            newly_cached += 1
                        except Exception:
                            continue
            except Exception:
                pass
            finally:
                if conn is not None:
                    conn.close()
        else:
            for s in needed:
                try:
                    frags = extract_functional_group_fragments(s, radius=self.radius)
                    with self._lock:
                        self._mem_cache[s] = frags
                    newly_cached += 1
                except Exception:
                    continue

        return newly_cached


# Alias for backward compatibility if referenced
BRICSCache = FunctionalGroupCache
