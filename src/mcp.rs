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

// ---------------------------------------------------------------- tools

pub fn tool_remember(conn: &Connection, args: &Value) -> Result<String, String> {
    let agent = req_str(args, "agent")?;
    let project = req_str(args, "project")?;
    let kind = req_str(args, "kind")?;
    if !curator::is_kind(kind) {
        return Err(format!(
            "invalid kind `{kind}`; expected troubleshoot|correction|fact|procedure"
        ));
    }
    let scope = opt_str(args, "scope").unwrap_or("own");
    if !curator::is_scope(scope) {
        return Err(format!("invalid scope `{scope}`; expected own|shared"));
    }
    let session = opt_str(args, "source_session").unwrap_or("");
    let content = req_str(args, "content")?;
    let content = db::truncate_chars(content, MAX_CONTENT_CHARS);

    // Exact duplicate: keep the raw record for the audit trail, but do not
    // create a second searchable memory.
    if let Some(id) = db::find_duplicate(conn, agent, &db::content_hash(&content))
        .map_err(|e| e.to_string())?
    {
        let raw = db::insert_raw(conn, agent, project, kind, scope, &content, session)
            .map_err(|e| e.to_string())?;
        db::mark_raw_consumed(conn, &[raw]).map_err(|e| e.to_string())?;
        return Ok(json!({
            "memory_id": id,
            "duplicate": true,
            "status": "duplicate",
            "message": "an identical memory already exists"
        })
        .to_string());
    }

    // Embedding runs outside the write transaction: ONNX is slow, and the
    // lock must not cover it.
    let vector = embeddings::embed_documents(std::slice::from_ref(&content))
        .and_then(|mut v| v.pop());

    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    let raw = db::insert_raw(&tx, agent, project, kind, scope, &content, session)
        .map_err(|e| e.to_string())?;
    let id = db::insert_memory(
        &tx, raw, agent, project, kind, scope, &content, "", "provisional", 0.6, None,
    )
    .map_err(|e| e.to_string())?;
    if let Some(v) = &vector {
        db::set_vector(&tx, id, v).map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;

    Ok(json!({
        "memory_id": id,
        "duplicate": false,
        "status": "provisional",
        "embedded": vector.is_some(),
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
    })
    .to_string())
}

pub fn tool_digest(conn: &Connection, args: &Value) -> Result<String, String> {
    let agent = req_str(args, "agent")?;
    match db::get_digest(conn, agent).map_err(|e| e.to_string())? {
        Some(d) if !d.trim().is_empty() => Ok(d),
        _ => Ok(format!("(no digest yet for `{agent}`)")),
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
                    "scope": {"type": "string", "enum": ["own", "shared"], "default": "own", "description": "own = visible to this agent; shared = visible to all agents. Curation may promote facts to shared."},
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
            "name": "digest",
            "description": "Return this agent's curated memory digest (capped, cache-stable). Useful at the start of a session.",
            "inputSchema": {
                "type": "object",
                "properties": {"agent": {"type": "string"}},
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
            "serverInfo": {"name": "ams", "version": env!("CARGO_PKG_VERSION")}
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
