# ams — central memory for AI agents

A single SQLite-backed memory store shared by multiple AI agents. Agents append raw
memories; a scheduled curator (Claude Haiku) merges, deduplicates, and promotes them into
canonical memories; retrieval is hybrid full-text (FTS5) + semantic (sqlite-vec)
search fused with Reciprocal Rank Fusion.

Designed for two write triggers: **after troubleshooting an unexpected behavior to a
working fix**, and **when a human corrects the agent's behavior**. The `remember` skill
in `skills/remember/SKILL.md` teaches agents those triggers.

## Quick start

```bash
cargo build --release

# one-time: create the database
./target/release/ams init

# register the MCP server with Claude Code (stores the API key for curation)
claude mcp add ams \
  --env AMS_ANTHROPIC_API_KEY=sk-ant-... \
  -- ./target/release/ams mcp

# copy the skills so agents know when to read and write memory
mkdir -p ~/.claude/skills/remember ~/.claude/skills/recall
cp skills/remember/SKILL.md ~/.claude/skills/remember/
cp skills/recall/SKILL.md ~/.claude/skills/recall/
```

On first use the embedding model (BAAI/bge-small-en-v1.5, ~130 MB) downloads into
`%LOCALAPPDATA%\ams\models` (Windows) or `~/.cache/ams/models`. Without the model, search
degrades to full-text only; without the API key, memories stay provisional and curation
is disabled — both remain fully usable.

## MCP tools

| tool | purpose |
|---|---|
| `remember` | append a raw memory; a searchable provisional record exists immediately |
| `search` | hybrid search, scoped to what the agent may see |
| `get` | full memory by id |
| `vote` | confirm (`up`) or flag (`down`) a memory you actually used |
| `digest` | curated context: project-specific when `project` is given, else all project digests |
| `history` | version history of one memory |
| `stats` | store counts |

Every call carries `agent` and `project` (honor system, not authentication). `project` is
a categorization label — memories are not isolated per project; pass it as a filter to
`search` when relevant. `scope` on `remember` is stored as `own`; promotion to `shared` is
a curation decision.

Agents are told how to use the store two ways: the MCP `initialize` response carries an
`instructions` string with the session-start flow, and the `remember`/`recall` skills
describe the write and read triggers in detail.

## How it works

### Two-stage write

1. **Append (fast, always succeeds).** `remember` writes a raw record to
   `raw_memories` (append-only ground truth) and a provisional row to `memories`,
   embedded and searchable within the same call. No LLM is involved, so latency is
   one embedding pass.
2. **Curate (scheduled).** A curator processes up to 20 pending raw records per run. It
   sees each provisional memory plus its nearest existing neighbors, and returns JSON
   operations: `update` (merge, reword, promote to shared, adjust confidence/expiry) and
   `archive` (duplicate or superseded). Each agent's digest is rewritten in the same
   pass.

Watermarks: raw rows are marked consumed only when the run completes. A failed API call
or unparseable response advances nothing and the same batch is retried.

Curation is enabled when `AMS_ANTHROPIC_API_KEY` is set. It runs in the MCP process on a
background thread, every `AMS_CURATE_INTERVAL_SECS` (default 7200) or as soon as 20
records are pending. A database-backed lease prevents two processes curating at once;
failed runs back off exponentially (capped at one hour) and never stop the thread.
`ams curate` runs a pass manually. The curator may only touch memories present in the
batch it was shown — ops on any other id are dropped.

The unique `(agent, content_hash)` index over non-archived memories makes duplicate
writes impossible even under concurrent sessions; a losing writer is answered as a
duplicate (and confirms the existing memory).

### Search

Three ranked signals fused with RRF (`k=60`):

1. **Lexical** — FTS5 `porter unicode61`, bm25 with tags weighted 4× over content. Every
   query token is quoted, so arbitrary agent input is always valid FTS syntax. If the
   ANDed query misses, an OR-relaxed retry runs, but a hit must contain at least two
   distinct content tokens (or all of them, for short queries) — otherwise a query like
   "write a haiku" matches a memory because it contains "write".
2. **Semantic** — sqlite-vec `vec0` KNN over BGE-small-en-v1.5 vectors (cosine, 384
   dims). Documents are embedded as-is and queries carry BGE v1.5's retrieval
   instruction, per the model's documented recipe. Candidates are gated twice: an
   absolute floor that scales with corpus size (`0.55 + 0.04·log10(n/10)`, capped at
   0.78, because the best cosine of an unrelated query grows with corpus size) and ≥
   85% of the query's own top score. `AMS_DEBUG_SCORES=1` prints raw scores, the
   distribution stats and the active floor.
3. **Recency** — candidates found by signals 1–2, ordered by creation time. Recency only
   reorders relevant memories; it never injects unrelated fresh ones.

Visibility: an agent sees its own memories plus all `shared` ones. Archived memories are
excluded. Returned hits bump `access_count`/`last_access_at`.

### Feedback loop

Memory is reinforced by real use, not by retrieval:

- An agent that applies a memory and verifies the result votes it up (`vote` tool). One
  vote per agent per session; the session defaults to the current MCP process, so
  separate sessions accumulate even when the host provides no session id.
- An agent that finds a memory wrong votes it down and writes a `correction` memory.
  The curator archives the superseded memory and sums its votes into the survivor.
- A `remember` call whose content is byte-identical to an existing memory is treated as
  validation: the duplicate is rejected and the existing memory is upvoted (the same
  signal Copilot Memory calls "upvote instead of storing a duplicate").
- Fused scores are multiplied by `1 + 0.3 × usefulness`, `usefulness =
  (up − down) / (up + down + 2)`, clamped to 0.7–1.3. Votes nudge, never dominate.
- Memories with ≥ 3 votes and a negative balance are hidden from retrieval; the curator
  archives them.

### Expiry

New memories expire after `AMS_MEMORY_TTL_DAYS` (default 90) — except `procedure`
memories, which never expire. Only an upvote resets the timer; retrieval does not. The
curator archives expired memories each run (recoverable via `ams history`/`restore`, and
`restore` clears the stale expiry and supersession link).

### Digests

One digest per `(agent, project)`, ≤ 2200 characters each, regenerated by the curator
whenever that agent/project has new records or had memories expire. `digest` with a
project returns that one; without, all of the agent's project digests.

### Memory tiers

- **Provisional** — captured, searchable, not yet curated.
- **Canonical** — curated; the agent digest is built from these.
- **Archived** — superseded duplicates/corrections; retained for audit, hidden from search.

Every content change writes a `memory_versions` row. `ams history <id>` shows them;
`ams restore <version_id>` restores one (also versioned).

### Schema

`raw_memories`, `memories` (+ FTS5 external-content table and triggers), `memories_vec`
(vec0), `memory_versions`, `digest`, `consumers` (watermarks), `locks`, `meta`.
Migrations are keyed on `PRAGMA user_version`. A model or text-recipe change is detected
via `meta.vector_model`; run `ams reembed` to rebuild all vectors.

## CLI

```
ams mcp                                 run the MCP server (stdio)
ams init                                create the db, print its path
ams remember --agent A --project P --kind K --content C [--scope own|shared]
ams search "query" --agent A [--project P] [--limit N]
ams digest --agent A
ams curate                              one curation pass
ams stats
ams history <memory_id>
ams restore <version_id>
ams reembed                             rebuild vectors after a model change
ams import --agent A --project P --dir D [--kind fact] [--chunk-chars 1200]
                                        ingest a markdown tree as canonical memories
```

## Configuration

| env var | default | meaning |
|---|---|---|
| `AMS_DB` | `<config dir>/ams/memory.db` | database path |
| `AMS_ANTHROPIC_API_KEY` | unset | enables curation |
| `AMS_CURATOR_MODEL` | `claude-haiku-5-5` | curation model |
| `AMS_ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | API base URL |
| `AMS_CURATE_BATCH` | `20` | max raw records per curation run |
| `AMS_CURATE_INTERVAL_SECS` | `7200` | curation interval |
| `AMS_CURATE_DISABLED` | unset | set to disable the background curator |
| `AMS_MEMORY_TTL_DAYS` | `90` | expiry for new memories without reinforcement |
| `AMS_MODEL_CACHE` | platform cache dir | embedding model cache |

Results show the source file (`<- database/foo.md`) when the memory came from an
imported corpus.

## Testing retrieval

```bash
cargo build
python bench/retrieval.py        # 14 seeded memories, 16 queries, ~40 s
python bench/corpus.py           # ingest sql-docs and run 115 queries through MCP
python bench/corpus.py --db PATH # reuse an existing corpus database
```

`retrieval.py` is the fast smoke test. `corpus.py` ingests a real markdown corpus
(default `C:\Users\ataid\projects\sql-docs`, 351 files, ~7k memories) and reports:

- heading queries: chunk rank@1/5 and file rank@1/5/MRR (unique headings only)
- paraphrase queries: whether the expected phrase appears in the top-5 chunk content
- negative queries: false positives
- latency per MCP search round trip

Last measured on sql-docs: chunk rank@1 67%, chunk rank@5 92%, file rank@5 95%,
paraphrase 10/10, negatives 0/5 false positives, p50 63 ms / p95 108 ms.

## Limits and caveats

- Writes are capped at 2000 chars so the 512-token embedding window covers the whole
  record. Longer input is truncated with an ellipsis.
- The semantic floor is a fitted heuristic (`0.55 + 0.04·log10(n/10)`), calibrated on a
  14-record corpus and the 7056-record sql-docs corpus. It is not a learned calibration;
  re-measure with `AMS_DEBUG_SCORES=1` when adding very different data.
- Agent identity is self-reported. A misconfigured agent can pollute `shared`; the
  correction/archive path (`supersedes_id`, `memory_versions`, `ams restore`) is the
  cleanup route.
- `porter unicode61` stems English. Non-English memories and code identifiers are
  matched less well by FTS; the embedding side still works.
- `raw_memories` is append-only and never pruned: `memories.raw_id` references it, so
  retention needs a schema change. Expect slow growth of a few hundred bytes per write.
- Duplicate raw rows from superseded memories stay in `raw_memories` for audit; only
  non-archived memory rows are deduplicated.
- Stored vectors are tied to the embedding model. Changing `MODEL_NAME` or bumping
  `TEXT_VERSION` in `src/embeddings.rs` requires `ams reembed`.
- The curator requires network access to the Anthropic API. Everything else is local.

## Design provenance

- Retrieval: hybrid FTS5 + embeddings + RRF, model-version stamping, graceful FTS-only
  degradation — from `md_jumper`.
- Write safety: weak/strong write split, deferred curation, versioned state, digest
  tier — from `NousResearch/hermes-agent`.
- Curation lifecycle: raw journal → scheduled consolidation, success-only watermarks,
  recurrence-based skill promotion, correction supersession — from `HKUDS/nanobot`.
