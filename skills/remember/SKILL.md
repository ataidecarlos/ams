---
name: remember
description: Write to the AMS agent memory after troubleshooting an unexpected behavior to a working fix, or when a human corrects your behavior. Use also when the user says "remember this", "note that", or asks why you did something you were previously corrected on.
---

# Remember (AMS memory)

AMS is a shared memory store for AI agents, exposed as MCP tools. This skill tells you
**when** and **how** to write memories. A memory written now saves a future agent (often
future-you) the whole investigation.

## When to write a memory

Write one when either of these happens:

1. **Unexpected behavior you had to troubleshoot.** Something behaved differently from
   what you or the user expected, you investigated, and you found a working fix. Record
   the fix, not the wandering. If it took more than one failed attempt, it is worth a
   memory.
2. **A human corrected you.** The user told you that an approach, assumption, output
   style, or behavior was wrong. Record the correction verbatim. Human corrections
   outrank everything you inferred yourself.

Do not write a memory for:

- Routine, expected work ("ran the tests, they passed").
- Facts already in the repository (README, docs, CLAUDE.md) — those are read, not
  remembered.
- Guesses you have not verified. If you could not confirm the fix worked, say so
  explicitly in the memory or do not write it.
- Secrets, tokens, passwords, personal data.

## How to write it

Call the MCP tool `remember` with:

| field | value |
|---|---|
| `agent` | Your agent name, e.g. `claude-code` |
| `project` | Repository/project name, e.g. `ams` or `acme/backend` |
| `kind` | `troubleshoot` or `correction` (see below) |
| `content` | The structured record, ≤ 2000 chars (see templates) |
| `scope` | `own` (default). The curator may promote project-independent facts to `shared`. |
| `source_session` | Optional session id, if the host provides one |

Keep **one fact per call**. If you learned three independent things, write three
memories. Small records embed fully and match precisely; a wall of text matches nothing.

### Template — troubleshoot

```
Symptom: <what went wrong, observable>
Cause: <the actual root cause, if you found it>
Fix: <the exact change or command that made it work>
Verification: <how you confirmed it works>
Env: <versions/platform only if they matter>
```

### Template — correction

```
Correction: <what you did that was wrong>
Correct behavior: <what to do instead>
Quote: "<the human's words, verbatim>"
Context: <when this applies>
```

## Before writing

1. **Search first.** Call `search` with a few words from the symptom or correction. If an
   equivalent memory exists, do not duplicate it — a duplicate is detected automatically
   and rejected, but searching first is cheaper and catches paraphrases.
2. Write with the vocabulary a future agent would search for: error text, file names,
   tool names, commands. Avoid "the issue", "it", "the thing".

## Examples

Troubleshoot:

```
remember(
  agent: "claude-code",
  project: "ams",
  kind: "troubleshoot",
  content: "Symptom: cargo build fails in sqlite-vec with C1083 sqlite-vec-diskann.c missing.\nCause: the 0.1.10-alpha.4 crate package omits a file; 0.1.9 builds cleanly.\nFix: pin sqlite-vec = \"0.1.9\" in Cargo.toml.\nVerification: cargo test passes, vec0 tables initialize."
)
```

Correction:

```
remember(
  agent: "claude-code",
  project: "frontend",
  kind: "correction",
  content: "Correction: I used default exports for React components in this repo.\nCorrect behavior: use named exports everywhere.\nQuote: \"we never use default exports, fix that\"\nContext: all files under src/components/."
)
```

## After a memory helped you

Memory only stays useful if agents report back. Two signals:

1. **It worked.** Once you actually applied a memory and verified the result, call the
   MCP tool `vote` with `vote: "up"`. This confirms the memory and resets its expiry
   timer. Passing a `session` id limits you to one vote per memory per session.
2. **It was wrong or outdated.** Call `vote` with `vote: "down"`, then write a
   `correction` memory that supersedes it. Never downvote a memory you did not use.

Do not vote just because a search returned something, or because you wrote it. Use is
the signal, retrieval is not.

## Reading memory back

- `search` — hybrid lexical + semantic search; use it when a task resembles something you
  or another agent may have hit before.
- `digest` — the curated summary for your agent, useful at the start of a session.
- `get` — full record by id, once search or digest surfaced it.

Do not rely on memory as a source of truth. Verify against the code and the live system;
memories describe what was true when written. If a memory turns out to be wrong, tell the
user and write a correction memory that supersedes it.
