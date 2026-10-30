use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::{db, embeddings};

#[derive(Debug, Default)]
pub struct ImportSummary {
    pub files: usize,
    pub chunks: usize,
    pub inserted: usize,
    pub duplicates: usize,
    pub skipped: usize,
}

/// Ingest a markdown tree as canonical memories, one memory per heading-sized
/// chunk. Used for bootstrapping a corpus (and for the retrieval bench);
/// agent-captured memories should go through `remember` instead.
pub fn import_dir(
    conn: &Connection,
    agent: &str,
    project: &str,
    kind: &str,
    dir: &Path,
    chunk_chars: usize,
) -> Result<ImportSummary, String> {
    if !embeddings::available() {
        return Err("embedding model unavailable; import needs vectors".into());
    }
    let files = collect_markdown(dir);
    let mut summary = ImportSummary {
        files: files.len(),
        ..Default::default()
    };
    let mut pending: Vec<(String, String)> = Vec::new();
    for path in files {
        let Ok(bytes) = std::fs::read(&path) else {
            summary.skipped += 1;
            continue;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            summary.skipped += 1;
            continue;
        };
        let source = path.display().to_string();
        for chunk in chunk_markdown(&text, chunk_chars) {
            summary.chunks += 1;
            pending.push((chunk, source.clone()));
            if pending.len() >= 32 {
                flush(conn, agent, project, kind, &mut pending, &mut summary)?;
            }
        }
    }
    flush(conn, agent, project, kind, &mut pending, &mut summary)?;
    Ok(summary)
}

fn flush(
    conn: &Connection,
    agent: &str,
    project: &str,
    kind: &str,
    pending: &mut Vec<(String, String)>,
    summary: &mut ImportSummary,
) -> Result<(), String> {
    if pending.is_empty() {
        return Ok(());
    }
    let texts: Vec<String> = pending.iter().map(|(c, _)| c.clone()).collect();
    let vectors =
        embeddings::embed_documents(&texts).ok_or("embedding model unavailable during import")?;
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    for ((content, source), vector) in pending.iter().zip(vectors) {
        let raw = db::insert_raw(&tx, agent, project, kind, "own", content, source)
            .map_err(|e| e.to_string())?;
        match db::insert_memory(
            &tx, raw, agent, project, kind, "own", content, "", "canonical", 0.7, None,
        ) {
            Ok(id) => {
                db::set_vector(&tx, id, &vector).map_err(|e| e.to_string())?;
                db::mark_raw_consumed(&tx, &[raw]).map_err(|e| e.to_string())?;
                summary.inserted += 1;
            }
            Err(e) if db::is_unique_violation(&e) => {
                // Identical boilerplate across files: keep the first copy only.
                db::mark_raw_consumed(&tx, &[raw]).map_err(|e| e.to_string())?;
                summary.duplicates += 1;
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    tx.commit().map_err(|e| e.to_string())?;
    pending.clear();
    Ok(())
}

fn collect_markdown(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("md"))
            {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn strip_frontmatter(text: &str) -> &str {
    let rest = match text.strip_prefix("---") {
        Some(r) if r.starts_with('\n') || r.starts_with("\r\n") => r,
        _ => return text,
    };
    match rest.find("\n---") {
        Some(end) => rest[end + 4..].trim_start_matches(['\r', '\n']),
        None => text,
    }
}

fn char_truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Split markdown into chunks of at most `cap` characters, keyed on headings.
/// Every chunk repeats the current heading so it stays retrievable on its own.
pub fn chunk_markdown(text: &str, cap: usize) -> Vec<String> {
    let cap = cap.max(200);
    let text = strip_frontmatter(text);
    let mut out: Vec<String> = Vec::new();
    let mut current_heading = String::new();
    let mut buf = String::new();
    let mut paragraph = String::new();

    fn flush(buf: &mut String, out: &mut Vec<String>) {
        let t = buf.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
        buf.clear();
    }

    fn push_paragraph(
        paragraph: &str,
        heading: &str,
        buf: &mut String,
        out: &mut Vec<String>,
        cap: usize,
    ) {
        let trimmed = paragraph.trim();
        if trimmed.is_empty() {
            return;
        }
        let add_len = trimmed.chars().count() + 2;
        if !buf.is_empty() && buf.chars().count() + add_len > cap {
            flush(buf, out);
            if !heading.is_empty() {
                buf.push_str(heading);
                buf.push('\n');
            }
        }
        let room = cap.saturating_sub(buf.chars().count()).max(50);
        buf.push_str(char_truncate(trimmed, room));
        buf.push_str("\n\n");
    }

    let mut handle = |paragraph: &mut String,
                      current_heading: &mut String,
                      buf: &mut String,
                      out: &mut Vec<String>| {
        let trimmed = paragraph.trim();
        if trimmed.is_empty() {
            paragraph.clear();
            return;
        }
        let is_heading =
            trimmed.starts_with('#') && trimmed.trim_start_matches('#').starts_with(' ');
        if is_heading {
            flush(buf, out);
            // A paragraph may hold a heading plus its unseparated body text.
            let mut lines = trimmed.lines();
            let first = lines.next().unwrap_or("");
            *current_heading = first.trim_start_matches('#').trim().to_string();
            buf.push_str(current_heading);
            buf.push('\n');
            let rest: Vec<&str> = lines.collect();
            if !rest.is_empty() {
                push_paragraph(&rest.join("\n"), current_heading, buf, out, cap);
            }
        } else {
            push_paragraph(trimmed, current_heading, buf, out, cap);
        }
        paragraph.clear();
    };

    for line in text.lines() {
        if line.trim().is_empty() {
            handle(&mut paragraph, &mut current_heading, &mut buf, &mut out);
        } else {
            paragraph.push_str(line);
            paragraph.push('\n');
        }
    }
    handle(&mut paragraph, &mut current_heading, &mut buf, &mut out);
    flush(&mut buf, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_frontmatter() {
        let text = "---\ntitle: X\nms.date: 01/01\n---\n\n# Heading\nBody";
        assert!(chunk_markdown(text, 500)[0].starts_with("Heading"));
    }

    #[test]
    fn chunks_respect_cap_and_repeat_heading() {
        let long = "word ".repeat(200);
        let text = format!("# Topic\n\n{long}\n\n## Sub\n\nSecond paragraph.");
        let chunks = chunk_markdown(&text, 300);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.chars().count() <= 300 + 100, "chunk too large: {}", c.chars().count());
        }
        assert!(chunks.iter().all(|c| !c.trim().is_empty()));
    }

    #[test]
    fn small_doc_splits_intro_from_heading() {
        let text = "intro text before headings\n\n# Title\nthe body text";
        let chunks = chunk_markdown(text, 1000);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].starts_with("intro"));
        assert!(chunks[1].starts_with("Title"));
        assert!(chunks[1].contains("the body text"));
    }
}
