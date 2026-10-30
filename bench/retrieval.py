#!/usr/bin/env python3
"""Retrieval quality bench for ams.

Seeds a throwaway database with realistic agent memories, runs queries whose
expected answer is known, and reports where (or whether) the expected memory
was found.

Usage:  python bench/retrieval.py [path-to-ams]
Default binary: ./target/debug/ams
"""
import json
import os
import re
import subprocess
import sys
import tempfile

AMS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "target", "debug", "ams",
)

# (kind, agent, project, content)
SEED = [
    ("troubleshoot", "claude", "ams",
     "Symptom: git push returns 403 Permission to ataidecarlos/ams.git denied. "
     "Cause: the GITHUB_TOKEN env var holds a fine-grained PAT with read-only Contents "
     "and shadows the working keyring OAuth credential. "
     "Fix: push with env -u GITHUB_TOKEN. Verification: git push succeeds."),
    ("fact", "claude", "ams",
     "ams uses sqlite-vec vec0 virtual tables with cosine distance and 384-dimensional "
     "BGE-small embeddings."),
    ("troubleshoot", "claude", "ams",
     "Symptom: cargo build fails in sqlite-vec with fatal error C1083: "
     "sqlite-vec-diskann.c missing. Cause: the 0.1.10-alpha.4 crate package omits a file "
     'on Windows. Fix: pin sqlite-vec = "0.1.9". Verification: cargo test passes.'),
    ("correction", "frontend-agent", "webapp",
     "Correction: I used default exports for React components. "
     "Correct behavior: named exports only. "
     'Quote: "we never use default exports, fix that". '
     "Context: src/components."),
    ("troubleshoot", "frontend-agent", "webapp",
     "Symptom: Vite dev server returns 500 after adding a path alias. "
     "Cause: tsconfig paths are not mirrored in vite.config.ts resolve.alias. "
     "Fix: add the alias under resolve.alias. Verification: dev server serves the module."),
    ("fact", "claude", "ams",
     "The CI pipeline runs cargo test on Windows arm64 and Linux x64; "
     "Windows jobs use the bundled SQLite build."),
    ("procedure", "claude", "ams",
     "Releasing ams: bump version in Cargo.toml, run cargo test, build --release, "
     "tag the commit, push the tag."),
    ("fact", "claude", "ams",
     "User prefers terse answers and no emoji in code comments."),
    ("troubleshoot", "claude", "ams",
     "Symptom: cargo test hangs on Windows in fastembed. "
     "Cause: the model download on first run blocks on the cache lock. "
     "Fix: pre-warm the model with ams reembed. Verification: tests finish."),
    ("fact", "claude", "ams",
     "API keys live in daemon environment variables, never in the repository."),
    ("troubleshoot", "claude", "ams",
     "Symptom: SQLite reports database is locked under concurrent writes. "
     "Cause: WAL writer contention between the curator transaction and remember calls. "
     "Fix: raise busy_timeout to 5000ms."),
    ("fact", "frontend-agent", "webapp",
     "The frontend uses pnpm workspaces with a shared ui package."),
    ("troubleshoot", "ops", "infra",
     "Symptom: Docker build fails with no space left on device. "
     "Cause: dangling images accumulate on the builder. "
     "Fix: docker system prune -af. Verification: build completes."),
    ("correction", "claude", "ams",
     "Correction: do not amend commits that are already pushed. "
     'Correct behavior: create a new commit instead. Quote: "never amend after push".'),
]

# (query, expected seed index (0-based), note). None = expect no results.
QUERIES = [
    ("github push 403 denied", "claude", 0, "lexical + exact error"),
    ("keychain credentials ignored by git", "claude", 0, "semantic, few shared words"),
    ("token in environment hides oauth login", "claude", 0, "semantic paraphrase"),
    ("sqlite-vec compile error missing file windows", "claude", 2, "lexical + semantic"),
    ("cargo build fails with C1083", "claude", 2, "lexical code"),
    ("sqlite3 vector extension registration bug", "claude", 2, "semantic cross-vocab"),
    ("react components export style", "frontend-agent", 3, "correction, cross-agent"),
    ("dev server 500 path alias", "frontend-agent", 4, "troubleshoot"),
    ("pnpm workspace ui package", "frontend-agent", 11, "fact"),
    ("release process", "claude", 6, "procedure"),
    ("database locked errors", "claude", 10, "troubleshoot"),
    ("docker disk full", "ops", 12, "troubleshoot, other project"),
    ("terse answers emoji comments", "claude", 7, "preference"),
    ("banana bread recipe", "claude", None, "negative control"),
    ("how do I write a haiku poem", "claude", None, "neg: one shared filler word"),
    ("stock market trends this quarter", "claude", None, "neg: semantic noise ceiling"),
]

TOP_K = 3


def run(args, env):
    p = subprocess.run([AMS] + args, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"command failed: {args}\n{p.stderr}")
    return p.stdout


def main():
    workdir = tempfile.mkdtemp(prefix="ams_bench_")
    env = dict(os.environ)
    env["AMS_DB"] = os.path.join(workdir, "memory.db")
    env["AMS_CURATE_DISABLED"] = "1"

    ids = []
    for kind, agent, project, content in SEED:
        out = run(["remember", "--agent", agent, "--project", project,
                   "--kind", kind, "--content", content], env)
        ids.append(json.loads(out.strip().splitlines()[-1])["memory_id"])

    # Reinforcement case: the release procedure gets three independent upvotes.
    for i in range(3):
        run(["vote", str(ids[6]), "--agent", "claude", "--vote", "up",
             "--session", f"bench-{i}"], env)

    failures = 0
    print(f"{'query':<48} {'expected':>8} {'result':<54}")
    print("-" * 112)
    for query, agent, expected, note in QUERIES:
        out = run(["search", query, "--agent", agent, "--limit", str(TOP_K)], env)
        hits = [int(m) for m in re.findall(r"^\[(\d+)\]", out, re.MULTILINE)]

        if expected is None:
            ok = not hits
            result = "no hits" if ok else f"FALSE POSITIVE: {hits}"
            exp_label = "-"
        else:
            target = ids[expected]
            ok = target in hits
            rank = hits.index(target) + 1 if ok else None
            result = (f"rank {rank} {hits}" if ok else f"MISS {hits}")
            exp_label = str(target)
        failures += 0 if ok else 1
        mark = "ok  " if ok else "FAIL"
        print(f"{query:<48} {exp_label:>8} {mark} {result:<47} ({note})")

    print("-" * 112)
    total = len(QUERIES)
    print(f"{total - failures}/{total} queries passed; db {env['AMS_DB']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
