---
name: recall
description: Load relevant AMS memories at the start of a session, and search them whenever a problem sounds familiar or before troubleshooting. Use when the user mentions "we fixed this before", a recurring error, or a known quirk of this project.
---

# Recall (AMS memory)

AMS is the shared memory store. Reading it is cheap; rediscovering a known problem is
not.

## At session start

Call the MCP tool `digest` with your agent name and the project:

```
digest(agent: "claude-code", project: "ams")
```

The digest is the curated, reinforced subset of what is known: durable facts, active
corrections, and useful procedures for this project. Load it before the first task. If it
is empty, continue normally — memory grows as agents write.

## When a problem sounds familiar

Before investigating, call `search`:

```
search(query: "cargo build fails sqlite-vec missing file", agent: "claude-code", project: "ams")
```

Results are ranked by hybrid full-text + semantic match, then adjusted by how often other
sessions confirmed each memory. Each hit shows an id, kind, project, agent and whether it
was voted on. Use `get(id)` to read the full record.

Trust but verify: a memory describes what was true when it was written. If it no longer
matches the code or the system, do not silently ignore it — vote it down and write a
correction (see the `remember` skill).

## After using a memory

- If the memory's guidance worked and you verified the result: `vote` it up.
- If it was wrong or outdated: `vote` it down, then write a `correction` memory.

Retrieval alone is not use. Do not vote on memories you did not apply.
