mod curator;
mod db;
mod embeddings;
mod mcp;
mod search;

use clap::{Parser, Subcommand};
use rusqlite::Connection;

#[derive(Parser)]
#[command(name = "ams", about = "Central memory for AI agents (FTS5 + embeddings, curated)")]
struct Cli {
    #[command(subcommand)]
    command: Sub,
}

#[derive(Subcommand)]
enum Sub {
    /// Run the MCP server over stdio (what agent hosts connect to)
    Mcp,
    /// Create the database if missing and print its path
    Init,
    /// Store a memory (same semantics as the MCP `remember` tool)
    Remember {
        #[arg(long)]
        agent: String,
        #[arg(long)]
        project: String,
        /// troubleshoot | correction | fact | procedure
        #[arg(long)]
        kind: String,
        #[arg(long)]
        content: String,
        #[arg(long, default_value = "own")]
        scope: String,
        #[arg(long, default_value = "")]
        source_session: String,
    },
    /// Hybrid search over memories
    Search {
        query: String,
        #[arg(long)]
        agent: String,
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
    },
    /// Print an agent's curated digest
    Digest {
        #[arg(long)]
        agent: String,
    },
    /// Run one curation pass (needs AMS_ANTHROPIC_API_KEY)
    Curate,
    /// Memory store counts
    Stats,
    /// Version history of one memory
    History {
        memory_id: i64,
    },
    /// Restore a memory to a stored version
    Restore {
        version_id: i64,
    },
    /// Rebuild every vector with the current embedding model (after a model
    /// or text-recipe change)
    Reembed,
}

fn main() {
    let cli = Cli::parse();
    if let Sub::Mcp = cli.command {
        if let Err(e) = mcp::serve(db::config_db_path()) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }

    let conn = match db::open_db() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: cannot open database: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = run(&conn, cli.command) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(conn: &Connection, cmd: Sub) -> Result<(), String> {
    match cmd {
        Sub::Mcp => unreachable!(),
        Sub::Init => {
            println!("{}", db::config_db_path().display());
        }
        Sub::Remember {
            agent,
            project,
            kind,
            content,
            scope,
            source_session,
        } => {
            let args = serde_json::json!({
                "agent": agent,
                "project": project,
                "kind": kind,
                "content": content,
                "scope": scope,
                "source_session": source_session,
            });
            println!("{}", mcp::tool_remember(conn, &args)?);
        }
        Sub::Search {
            query,
            agent,
            project,
            limit,
        } => {
            let hits = search::search(conn, &agent, project.as_deref(), &query, limit)?;
            if hits.is_empty() {
                println!("no matching memories");
            } else {
                print!("{}", search::format_hits(&hits));
            }
        }
        Sub::Digest { agent } => match db::get_digest(conn, &agent).map_err(|e| e.to_string())? {
            Some(d) if !d.trim().is_empty() => println!("{d}"),
            _ => println!("(no digest yet for `{agent}`)"),
        },
        Sub::Curate => {
            let cfg = curator::CuratorConfig::from_env()
                .ok_or("AMS_ANTHROPIC_API_KEY is not set; curation needs an API key")?;
            let r = curator::run_once(conn, &cfg)?;
            if r.skipped {
                println!("skipped: another process holds the curation lock");
            } else if r.batch == 0 {
                println!("nothing to curate");
            } else {
                println!(
                    "curated {} record(s): {} updated, {} archived, {} promoted, {} digest(s)",
                    r.batch, r.updated, r.archived, r.promoted, r.digests
                );
            }
        }
        Sub::Stats => {
            let s = db::stats(conn).map_err(|e| e.to_string())?;
            println!("raw records: {} ({} pending curation)", s.raw_total, s.raw_pending);
            println!(
                "memories: {} ({} provisional, {} canonical, {} archived)",
                s.memories, s.provisional, s.canonical, s.archived
            );
            println!("agents: {}", s.agents);
        }
        Sub::History { memory_id } => {
            let versions = db::versions(conn, memory_id).map_err(|e| e.to_string())?;
            if versions.is_empty() {
                println!("no versions for memory {memory_id}");
            }
            for v in versions {
                println!("v{}  version_id={}  {}  at={}", v.version, v.id, v.op, v.at);
                println!("{}", v.content);
                println!("---");
            }
        }
        Sub::Restore { version_id } => {
            let id = db::restore_version(conn, version_id).map_err(|e| e.to_string())?;
            println!("restored memory {id} to version {version_id}");
        }
        Sub::Reembed => {
            let mems = db::active_memories(conn).map_err(|e| e.to_string())?;
            if mems.is_empty() {
                println!("no memories to embed");
                return Ok(());
            }
            let mut done = 0;
            for chunk in mems.chunks(16) {
                let texts: Vec<String> = chunk.iter().map(|(_, c)| c.clone()).collect();
                let Some(vectors) = embeddings::embed_documents(&texts) else {
                    return Err("embedding model unavailable".into());
                };
                for ((id, _), v) in chunk.iter().zip(vectors) {
                    db::set_vector(conn, *id, &v).map_err(|e| e.to_string())?;
                    done += 1;
                }
            }
            db::set_meta(conn, "vector_model", &embeddings::model_id())
                .map_err(|e| e.to_string())?;
            println!("re-embedded {done} memor{}", if done == 1 { "y" } else { "ies" });
        }
    }
    Ok(())
}
