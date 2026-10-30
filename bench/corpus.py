#!/usr/bin/env python3
"""Large-corpus retrieval bench for ams.

Ingests a markdown tree (default: the sql-docs corpus), then measures retrieval
through a live MCP server:

  * heading queries  - the chunk's own heading as the query; checks how often the
                       source chunk/file is found (chunk rank@1, file rank@1/5, MRR)
  * paraphrase queries - hand-written questions checked against the expected
                       source path substring
  * negative queries - unrelated questions; any hit is a false positive
  * latency          - median/p95 per search round trip

Usage:
  python bench/corpus.py [--corpus DIR] [--db PATH] [--sample N]

With --db pointing at an existing database, the import step is skipped.
"""
import argparse
import json
import os
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMS = os.path.join(ROOT, "target", "debug", "ams")
AGENT = "docs"
PROJECT = "sql-docs"
LIMIT = 10

PARAPHRASE = [
    ("how do I restore a database to a point in time", "point-in-time"),
    ("automatic backups cannot be deleted for seven days", "immutability"),
    ("replicate a database to another region and fail over", "geo-replication"),
    ("share resources between databases to save money", "elastic pool"),
    ("monitor query performance and execution plans", "query store"),
    ("authenticate users with Microsoft Entra ID", "microsoft entra"),
    ("build an append-only tamper-evident table", "ledger"),
    ("scale a database beyond 100 TB with a distributed architecture", "hyperscale"),
    ("migrate a database to a managed instance", "managed instance"),
    ("set up auditing and threat detection", "auditing"),
]

NEGATIVE = [
    "banana bread recipe with walnuts",
    "how to train for a marathon",
    "best hiking trails in norway",
    "python decorators explained",
    "weather forecast for lisbon",
]


def run(cmd, env, **kwargs):
    return subprocess.run(cmd, env=env, text=True, capture_output=True, **kwargs)


class Mcp:
    def __init__(self, env):
        self.p = subprocess.Popen(
            [AMS, "mcp"], env=env, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._send({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"}})
        self.p.stdout.readline()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.n = 0

    def _send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def search(self, query):
        self.n += 1
        start = time.perf_counter()
        self._send({"jsonrpc": "2.0", "id": self.n, "method": "tools/call",
                    "params": {"name": "search",
                               "arguments": {"query": query, "agent": AGENT,
                                             "project": PROJECT, "limit": LIMIT}}})
        line = self.p.stdout.readline()
        elapsed = time.perf_counter() - start
        text = json.loads(line)["result"]["content"][0]["text"]
        ids = [int(m) for m in re.findall(r"^\[(\d+)\]", text, re.MULTILINE)]
        return ids, elapsed

    def close(self):
        self.p.stdin.close()
        self.p.kill()
        self.p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=r"C:\Users\ataid\projects\sql-docs")
    ap.add_argument("--db", help="reuse an existing database (skips import)")
    ap.add_argument("--sample", type=int, default=100)
    args = ap.parse_args()

    db_path = args.db
    if not db_path:
        workdir = tempfile.mkdtemp(prefix="ams_corpus_")
        db_path = os.path.join(workdir, "memory.db")
        env = dict(os.environ, AMS_DB=db_path, AMS_CURATE_DISABLED="1")
        print(f"importing {args.corpus} (one-off, can take minutes)...", flush=True)
        out = run([AMS, "import", "--agent", AGENT, "--project", PROJECT,
                   "--kind", "fact", "--dir", args.corpus], env)
        if out.returncode != 0:
            raise SystemExit(out.stderr)
        print(out.stdout.strip(), flush=True)

    env = dict(os.environ, AMS_DB=db_path, AMS_CURATE_DISABLED="1")
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT m.id, m.content, r.source_session FROM memories m "
        "JOIN raw_memories r ON r.id = m.raw_id "
        "WHERE m.agent = ? AND m.project = ? AND m.status != 'archived'",
        (AGENT, PROJECT),
    ).fetchall()
    id_to_path = {i: p for i, _, p in rows}
    id_to_content = {i: c.lower() for i, c, _ in rows}
    print(f"corpus: {len(rows)} memories", flush=True)

    # Heading queries: the chunk's first line is its heading (the chunker
    # repeats it). Keep only headings that are unique in the corpus and look
    # like titles - generic repeated headings ("Related content", tab labels)
    # have no single correct answer and would poison the metric.
    firsts = [c.strip().splitlines()[0].strip() for _, c, _ in rows]
    seen = {}
    for f in firsts:
        seen[f] = seen.get(f, 0) + 1
    GENERIC = {
        "related content", "next steps", "see also", "feedback", "syntax",
        "arguments", "examples", "remarks", "permissions", "prerequisites",
        "limitations", "additional resources", "overview", "summary",
    }
    candidates = []
    for (i, content, path), first in zip(rows, firsts):
        words = first.split()
        if (
            15 <= len(first) <= 120
            and 2 <= len(words) <= 12
            and seen[first] == 1
            and first.lower() not in GENERIC
            and not any(ch in first for ch in "[]|>#*`")
        ):
            candidates.append((first, i, path))
    rng = random.Random(42)
    heading_queries = rng.sample(candidates, min(args.sample, len(candidates)))
    print(f"unique headings: {len(candidates)}", flush=True)

    mcp = Mcp(env)
    chunk_at1 = chunk_at5 = 0
    file_at1 = file_at5 = 0
    rr_chunk = rr_file = 0.0
    misses = []
    latencies = []

    for query, expected_id, expected_path in heading_queries:
        ids, elapsed = mcp.search(query)
        latencies.append(elapsed)
        rank = ids.index(expected_id) + 1 if expected_id in ids else None
        file_rank = next(
            (n + 1 for n, i in enumerate(ids) if id_to_path.get(i) == expected_path), None
        )
        chunk_at1 += rank == 1
        chunk_at5 += bool(rank and rank <= 5)
        file_at1 += file_rank == 1
        file_at5 += bool(file_rank and file_rank <= 5)
        rr_chunk += 1.0 / rank if rank else 0.0
        rr_file += 1.0 / file_rank if file_rank else 0.0
        if not (file_rank and file_rank <= 5):
            misses.append((query, expected_path, [id_to_path.get(i, "?") for i in ids[:3]]))

    n = len(heading_queries)
    print("\nheading queries (correct source file must appear in top-5):")
    print(f"  chunk rank@1 {chunk_at1 / n:5.1%}   chunk rank@5 {chunk_at5 / n:5.1%}   chunk MRR {rr_chunk / n:.3f}")
    print(f"  file  rank@1 {file_at1 / n:5.1%}   file  rank@5 {file_at5 / n:5.1%}   file  MRR {rr_file / n:.3f}")
    for q, exp, got in misses[:8]:
        print(f"  MISS  {q[:60]!r}\n        expected {os.path.basename(exp)}; got {[os.path.basename(g) for g in got]}")

    print("\nparaphrase queries (expected phrase in top-5 chunk content):")
    para_hits = 0
    for query, expected in PARAPHRASE:
        ids, elapsed = mcp.search(query)
        latencies.append(elapsed)
        texts = [id_to_content.get(i, "") for i in ids[:5]]
        ok = any(expected in t for t in texts)
        para_hits += ok
        if not ok:
            preview = [t.strip().splitlines()[0][:40] for t in texts[:2]]
            print(f"  MISS {query[:64]:<64} -> {preview}")
        else:
            print(f"  ok   {query[:64]}")
    print(f"  {para_hits}/{len(PARAPHRASE)} paraphrase queries found")

    print("\nnegative queries (any hit is a false positive):")
    fp = 0
    for query in NEGATIVE:
        ids, elapsed = mcp.search(query)
        latencies.append(elapsed)
        fp += bool(ids)
        print(f"  {'FALSE POSITIVE' if ids else 'ok            '} {query:<44} hits={len(ids)}")

    mcp.close()
    order = sorted(latencies)
    p50 = statistics.median(order) * 1000
    p95 = order[min(len(order) - 1, int(len(order) * 0.95))] * 1000
    print(f"\nlatency per MCP search: p50 {p50:.0f} ms, p95 {p95:.0f} ms "
          f"({len(order)} queries, server warm)")
    print(f"database: {db_path}")
    failed = (n - file_at5) + (len(PARAPHRASE) - para_hits) + fp
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
