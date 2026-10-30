use std::io::{BufRead, Write};
use std::path::PathBuf;

use rusqlite::Connection;
use serde_json::{Value, json};

use crate::{curator, db, embeddings, search};

/// Write-time cap. Records are structured and small; a longer entry gets
/// truncated so the embedding covers the whole thing (no tail loss).
pub const MAX_CONTENT_CHARS: usize = 2000;

fn req_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing string argument `{key}`"))
}

fn opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str()).map(str::trim).filter(|s| !s.is_empty())
}

/// Session identity for one MCP process. Hosts rarely hand agents a session
/// id, so without this every vote from an agent would collide on "" and the
/// reuse signal would saturate at one upvote forever.
static SESSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn default_session() -> &'static str {
    SESSION.get_or_init(db::new_session_id)
}

// ---------------------------------------------------------------- tools

/// A write that duplicates an existing memory: keep the raw record for the
/// audit trail and treat the rediscovery as validation, the same signal
/// Copilot Memory calls "upvote instead of storing a duplicate".
fn record_duplicate(
    conn: &Connection,
    existing_id: i64,
    agent: &str,
    project: &str,
    kind: &str,
    scope: &str,
    content: &str,
    session: &str,
) -> Result<String, String> {
    let raw = db::insert_raw(conn, agent, project, kind, scope, content, session)
        .map_err(|e| e.to_string())?;
    db::mark_raw_consumed(conn, &[raw]).map_err(|e| e.to_string())?;
    let counts = db::cast_vote(conn, existing_id, agent, session, 1, "duplicate write").ok();
    Ok(json!({
        "memory_id": existing_id,
        "duplicate": true,
        "status": "duplicate",
        "upvoted": counts.is_some(),
        "upvotes": counts.map(|c| c.up),
        "message": "an identical memory already exists; it was confirmed instead"
    })
    .to_string())
}

pub fn tool_remember(conn: &Connection, args: &Value) -> Result<String, String> {
    let agent = req_str(args, "agent")?;
    let project = req_str(args, "project")?;
    let kind = req_str(args, "kind")?;
    if !curator::is_kind(kind) {
        return Err(format!(
            "invalid kind `{kind}`; expected troubleshoot|correction|fact|procedure"
        ));
    }
    let requested_shared = match opt_str(args, "scope") {
        Some("shared") => true,
        Some("own") | None => false,
        Some(other) => return Err(format!("invalid scope `{other}`; expected own|shared")),
    };
    // Agents may request shared, but promotion stays a curation decision so one
    // misconfigured agent cannot plant facts every agent sees.
    let scope = "own";
    let session = opt_str(args, "source_session").unwrap_or("");
    let content = req_str(args, "content")?;
    let content = db::truncate_chars(content, MAX_CONTENT_CHARS);
    let hash = db::content_hash(&content);

    if let Some(id) = db::find_duplicate(conn, agent, &hash).map_err(|e| e.to_string())? {
        return record_duplicate(conn, id, agent, project, kind, scope, &content, session);
    }

    // Embedding runs outside the write transaction: ONNX is slow, and the
    // lock must not cover it.
    let vector = embeddings::embed_documents(std::slice::from_ref(&content))
        .and_then(|mut v| v.pop());

    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    let raw = db::insert_raw(&tx, agent, project, kind, scope, &content, session)
        .map_err(|e| e.to_string())?;
    // Fresh memories expire unless an upvote reinforces them; procedures are
    // exempt because a verified procedure is the most durable kind of memory.
    let expires_at = if kind == "procedure" {
        None
    } else {
        Some(db::now() + db::ttl_secs())
    };
    // The unique (agent, content_hash) index catches duplicates that slipped
    // past the pre-check under concurrency.
    let id = match db::insert_memory(
        &tx, raw, agent, project, kind, scope, &content, "", "provisional", 0.6, expires_at,
    ) {
        Ok(id) => {
            if let Some(v) = &vector {
                db::set_vector(&tx, id, v).map_err(|e| e.to_string())?;
            }
            tx.commit().map_err(|e| e.to_string())?;
            id
        }
        Err(e) if db::is_unique_violation(&e) => {
            drop(tx); // rolls back
            let Some(existing) = db::find_duplicate(conn, agent, &hash).map_err(|e| e.to_string())?
            else {
                return Err(e.to_string());
            };
            return record_duplicate(conn, existing, agent, project, kind, scope, &content, session);
        }
        Err(e) => return Err(e.to_string()),
    };

    Ok(json!({
        "memory_id": id,
        "duplicate": false,
        "status": "provisional",
        "embedded": vector.is_some(),
        "scope": scope,
        "note": requested_shared.then_some("requested shared is stored as own; curation decides promotion"),
        "message": "stored; visible to search immediately"
    })
    .to_string())
}

pub fn tool_search(conn: &Connection, args: &Value) -> Result<String, String> {
    let agent = req_str(args, "agent")?;
    let query = req_str(args, "query")?;
    let project = opt_str(args, "project");
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .clamp(1, 50) as usize;
    let hits = search::search(conn, agent, project, query, limit)?;
    if hits.is_empty() {
        return Ok("no matching memories".into());
    }
    Ok(search::format_hits(&hits))
}

pub fn tool_vote(conn: &Connection, args: &Value) -> Result<String, String> {
    let memory_id = args
        .get("memory_id")
        .and_then(|v| v.as_i64())
        .ok_or("missing integer argument `memory_id`")?;
    let agent = req_str(args, "agent")?;
    let vote = match req_str(args, "vote")? {
        "up" => 1,
        "down" => -1,
        other => return Err(format!("invalid vote `{other}`; expected up|down")),
    };
    let session = opt_str(args, "session")
        .map(str::to_string)
        .unwrap_or_else(|| default_session().to_string());
    let note = opt_str(args, "note").unwrap_or("");
    let counts = db::cast_vote(conn, memory_id, agent, &session, vote, note)?;
    Ok(json!({
        "memory_id": memory_id,
        "upvotes": counts.up,
        "downvotes": counts.down,
        "usefulness": db::usefulness(counts.up, counts.down),
    })
    .to_string())
}

pub fn tool_get(conn: &Connection, args: &Value) -> Result<String, String> {
    let id = args
        .get("id")
        .and_then(|v| v.as_i64())
        .ok_or("missing integer argument `id`")?;
    let m = db::get_memory(conn, id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("memory {id} not found"))?;
    Ok(json!({
        "id": m.id,
        "agent": m.agent,
        "project": m.project,
        "kind": m.kind,
        "scope": m.scope,
        "status": m.status,
        "confidence": m.confidence,
        "tags": m.tags,
        "content": m.content,
        "supersedes_id": m.supersedes_id,
        "superseded_by": m.superseded_by,
        "version": m.version,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "expires_at": m.expires_at,
        "access_count": m.access_count,
        "last_access_at": m.last_access_at,
        "upvotes": m.upvotes,
        "downvotes": m.downvotes,
        "last_confirmed_at": m.last_confirmed_at,
    })
    .to_string())
}

pub fn tool_digest(conn: &Connection, args: &Value) -> Result<String, String> {
    let agent = req_str(args, "agent")?;
    match opt_str(args, "project") {
        Some(project) => match db::get_digest(conn, agent, project).map_err(|e| e.to_string())? {
            Some(d) if !d.trim().is_empty() => Ok(d),
            _ => Ok(format!("(no digest yet for `{agent}` in `{project}`)")),
        },
        None => {
            let rows = db::digests_for_agent(conn, agent).map_err(|e| e.to_string())?;
            if rows.is_empty() {
                return Ok(format!("(no digest yet for `{agent}`)"));
            }
            let mut out = String::new();
            for (project, content) in rows {
                let name = if project.is_empty() { "(no project)" } else { &project };
                out.push_str(&format!("## {name}\n{content}\n\n"));
            }
            Ok(out.trim_end().to_string())
        }
    }
}

pub fn tool_stats(conn: &Connection, _args: &Value) -> Result<String, String> {
    let s = db::stats(conn).map_err(|e| e.to_string())?;
    Ok(json!({
        "raw_total": s.raw_total,
        "raw_pending": s.raw_pending,
        "memories": s.memories,
        "provisional": s.provisional,
        "canonical": s.canonical,
        "archived": s.archived,
        "agents": s.agents,
        "votes": s.votes,
    })
    .to_string())
}

pub fn tool_history(conn: &Connection, args: &Value) -> Result<String, String> {
    let memory_id = args
        .get("memory_id")
        .and_then(|v| v.as_i64())
        .ok_or("missing integer argument `memory_id`")?;
    let versions = db::versions(conn, memory_id).map_err(|e| e.to_string())?;
    let rows: Vec<Value> = versions
        .iter()
        .map(|v| {
            json!({
                "version_id": v.id,
                "memory_id": v.memory_id,
                "version": v.version,
                "op": v.op,
                "at": v.at,
                "content": v.content,
            })
        })
        .collect();
    Ok(serde_json::to_string_pretty(&rows).unwrap_or_default())
}

fn tool_list() -> Value {
    json!([
        {
            "name": "remember",
            "description": "Store a memory. Call after you troubleshoot an unexpected behavior to a working fix, or when a human corrects your behavior. Keep one fact per call.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Your agent name"},
                    "project": {"type": "string", "description": "Repository or project name"},
                    "kind": {"type": "string", "enum": ["troubleshoot", "correction", "fact", "procedure"]},
                    "content": {"type": "string", "description": "Max ~2000 chars. troubleshoot: symptom, cause, fix, verification. correction: what was wrong, what is correct, quote the human verbatim."},
                    "scope": {"type": "string", "enum": ["own", "shared"], "default": "own", "description": "Stored as own; curation decides promotion to shared."},
                    "source_session": {"type": "string", "description": "Optional session identifier for provenance"}
                },
                "required": ["agent", "project", "kind", "content"]
            }
        },
        {
            "name": "search",
            "description": "Hybrid (full-text + semantic) search over memories this agent can see. Returns ranked ids, kinds, projects and snippets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "agent": {"type": "string", "description": "Your agent name"},
                    "project": {"type": "string", "description": "Optional project filter"},
                    "limit": {"type": "integer", "default": 8, "description": "Max results (1-50)"}
                },
                "required": ["query", "agent"]
            }
        },
        {
            "name": "get",
            "description": "Fetch one memory in full by id.",
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"]
            }
        },
        {
            "name": "vote",
            "description": "Confirm a memory you actually used and verified (up), or flag one you found wrong or outdated (down), then write a correction memory for wrong ones. One vote per agent per session; an upvote resets the memory's expiry timer.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "integer"},
                    "agent": {"type": "string", "description": "Your agent name"},
                    "vote": {"type": "string", "enum": ["up", "down"]},
                    "session": {"type": "string", "description": "Session identifier; defaults to this MCP process. One vote per agent and session"},
                    "note": {"type": "string", "description": "Optional short reason"}
                },
                "required": ["memory_id", "agent", "vote"]
            }
        },
        {
            "name": "digest",
            "description": "Return the curated memory digest: project-specific when `project` is given, otherwise every project digest for this agent. Load it at the start of a session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "project": {"type": "string", "description": "Optional project name"}
                },
                "required": ["agent"]
            }
        },
        {
            "name": "history",
            "description": "Version history of one memory, including content at each version.",
            "inputSchema": {
                "type": "object",
                "properties": {"memory_id": {"type": "integer"}},
                "required": ["memory_id"]
            }
        },
        {
            "name": "stats",
            "description": "Memory store counts.",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ])
}

// ---------------------------------------------------------------- rpc

fn call_tool(conn: &Connection, msg: &Value) -> Result<Value, (i32, String)> {
    let name = msg
        .pointer("/params/name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let args = msg
        .pointer("/params/arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let out = match name {
        "remember" => tool_remember(conn, &args),
        "search" => tool_search(conn, &args),
        "get" => tool_get(conn, &args),
        "vote" => tool_vote(conn, &args),
        "digest" => tool_digest(conn, &args),
        "history" => tool_history(conn, &args),
        "stats" => tool_stats(conn, &args),
        _ => return Err((-32602, format!("unknown tool: {name}"))),
    };
    match out {
        Ok(text) => Ok(json!({"content": [{"type": "text", "text": text}]})),
        Err(e) => Ok(json!({
            "content": [{"type": "text", "text": e}],
            "isError": true
        })),
    }
}

fn handle(conn: &Connection, msg: &Value) -> Option<Value> {
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = msg.get("id").cloned().unwrap_or(Value::Null);
    let has_id = !id.is_null();

    let result: Result<Value, (i32, String)> = match method {
        "initialize" => Ok(json!({
            "protocolVersion": msg
                .pointer("/params/protocolVersion")
                .and_then(|v| v.as_str())
                .unwrap_or("2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ams", "version": env!("CARGO_PKG_VERSION")},
            "instructions": "AMS is a shared memory for AI agents. At the start of a session, call `digest` with your agent name and project to load what is already known. Before troubleshooting a familiar-sounding problem, call `search`. After a fix that came from a memory actually works, vote it up; if a memory was wrong or outdated, vote it down and write a correction. Call `remember` after troubleshooting an unexpected behavior to a working fix, or when a human corrects your behavior."
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tool_list()})),
        "tools/call" => call_tool(conn, msg),
        m if m.starts_with("notifications/") => Ok(Value::Null),
        _ => Err((-32601, format!("method not found: {method}"))),
    };

    if !has_id {
        return None;
    }
    Some(match result {
        Ok(v) if v.is_null() => json!({"jsonrpc": "2.0", "id": id, "result": {}}),
        Ok(v) => json!({"jsonrpc": "2.0", "id": id, "result": v}),
        Err((code, message)) => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
        }
    })
}

/// Run the MCP server over stdio (newline-delimited JSON-RPC), with background
/// curation when `AMS_ANTHROPIC_API_KEY` is set.
pub fn serve(db_path: PathBuf) -> Result<(), String> {
    let conn = db::open_db_at(&db_path).map_err(|e| format!("cannot open database: {e}"))?;

    std::thread::spawn(embeddings::warm);
    if let Some(cfg) = curator::CuratorConfig::from_env()
        && std::env::var_os("AMS_CURATE_DISABLED").is_none()
    {
        curator::spawn(db_path.clone(), cfg);
    }

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle(&conn, &msg),
            Err(e) => Some(json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {"code": -32700, "message": format!("parse error: {e}")}
            })),
        };
        if let Some(resp) = response {
            writeln!(stdout, "{}", resp).map_err(|e| e.to_string())?;
            stdout.flush().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
