use std::collections::HashMap;

use rusqlite::Connection;

use crate::db;
use crate::embeddings;

/// Absolute similarity floor for a semantic hit. Measured on the bench corpus:
/// unrelated queries top out around 0.50, genuine cross-vocabulary matches
/// start around 0.64. Re-calibrate with `AMS_DEBUG_SCORES=1` on real data.
const MIN_SEMANTIC_SCORE: f32 = 0.55;
/// Candidates must also be within this fraction of the query's own best score,
/// which adapts to how sharply this query separates its matches.
const RELATIVE_FLOOR: f32 = 0.85;

/// Build an FTS5 MATCH expression from a raw query. Every token is quoted, so
/// arbitrary input (FTS operators, punctuation, code ids) is always valid
/// syntax. A trailing `*` on a token becomes a prefix search. Tokens AND by
/// default; `or` joins them with OR instead.
pub fn fts_match(query: &str, or: bool) -> String {
    let parts: Vec<String> = query
        .split_whitespace()
        .map(|tok| {
            let (tok, star) = match tok.strip_suffix('*') {
                Some(t) if !t.is_empty() => (t, true),
                _ => (tok, false),
            };
            let quoted = format!("\"{}\"", tok.replace('"', "\"\""));
            if star { format!("{}*", quoted) } else { quoted }
        })
        .collect();
    parts.join(if or { " OR " } else { " " })
}

/// Keep semantic candidates that clear both the absolute floor and 85% of the
/// best score for this query. `scores` is expected sorted descending.
///
/// Calibrated on the bench corpus (`bench/retrieval.py`): unrelated queries top
/// out near 0.50, genuine cross-vocabulary matches start near 0.64. Re-measure
/// with `AMS_DEBUG_SCORES=1` when real memory data changes the distribution.
fn semantic_gate(scores: &mut Vec<(i64, f32)>) {
    scores.retain(|(_, s)| *s >= MIN_SEMANTIC_SCORE);
    let Some(top) = scores.first().map(|(_, s)| *s) else {
        return;
    };
    let cut = top * RELATIVE_FLOOR;
    scores.retain(|(_, s)| *s >= cut);
}

/// Vote signal applied to the fused score. Bounded (0.7–1.3) so evidence
/// nudges the fusion order without letting one upvote dominate it.
fn vote_multiplier(up: i64, down: i64) -> f64 {
    let u = db::usefulness(up, down) as f64;
    (1.0 + 0.3 * u).clamp(0.7, 1.3)
}

/// Reciprocal Rank Fusion: score = Σ 1/(k + rank). `k` = 60 is the constant
/// from the RRF paper.
pub fn rrf_merge(lists: &[Vec<i64>], k: f64, limit: usize) -> Vec<(i64, f64)> {
    let mut scores: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + (rank + 1) as f64);
        }
    }
    let mut merged: Vec<(i64, f64)> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged.truncate(limit);
    merged
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub memory: db::Memory,
    pub snippet: String,
    pub semantic_only: bool,
    pub score: f64,
}

/// Hybrid retrieval over memories the agent may see: lexical (FTS5) and
/// semantic (sqlite-vec KNN) ranked lists fused with RRF, plus a recency list
/// over the candidates both signals already found.
pub fn search(
    conn: &Connection,
    agent: &str,
    project: Option<&str>,
    query: &str,
    limit: usize,
) -> Result<Vec<Hit>, String> {
    let qv = embeddings::embed_query(query);
    search_with_vector(conn, agent, project, query, limit, qv)
}

/// Same as [`search`] with an explicit query vector (or `None` for
/// lexical-only). Split out so tests never load the embedding model.
pub fn search_with_vector(
    conn: &Connection,
    agent: &str,
    project: Option<&str>,
    query: &str,
    limit: usize,
    query_vector: Option<Vec<f32>>,
) -> Result<Vec<Hit>, String> {
    let limit = limit.max(1);
    let pool = (limit * 4).max(20);

    // Signal 1: lexical. Fall back to an OR-relaxed query when the ANDed one
    // misses, the way hermes-agent's retrieval ladder does.
    let and_expr = fts_match(query, false);
    if and_expr.is_empty() {
        return Err("empty search query".into());
    }
    let mut lexical = db::fts_search(conn, &and_expr, agent, project, pool as i64)
        .map_err(|e| format!("search failed: {e}"))?;
    if lexical.is_empty() && query.split_whitespace().count() > 1 {
        let or_expr = fts_match(query, true);
        let candidates = db::fts_search(conn, &or_expr, agent, project, pool as i64)
            .map_err(|e| format!("search failed: {e}"))?;
        // An OR match on one common word ("do", "write") is noise: require at
        // least two distinct content tokens in the hit, unless the query only
        // has one.
        let tokens = content_tokens(query);
        let needed = tokens.len().min(2);
        lexical = candidates
            .into_iter()
            .filter(|(id, _)| {
                if needed == 0 {
                    return false;
                }
                db::get_memory(conn, *id)
                    .ok()
                    .flatten()
                    .map(|m| count_token_hits(&m.content, &tokens) >= needed)
                    .unwrap_or(false)
            })
            .collect();
    }

    // Signal 2: semantic KNN. Absent model => silently skip; FTS still works.
    let mut semantic: Vec<(i64, f32)> = match query_vector {
        Some(qv) => {
            let k = (pool * 2).max(40) as i64;
            db::knn(conn, &qv, agent, project, k)
                .map_err(|e| format!("semantic search failed: {e}"))?
                .into_iter()
                // cosine distance -> similarity
                .map(|(id, d)| (id, 1.0 - d))
                .collect()
        }
        None => Vec::new(),
    };
    semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Calibration aid: AMS_DEBUG_SCORES=1 prints the raw cosine scores that
    // the gate is about to accept or reject.
    if std::env::var_os("AMS_DEBUG_SCORES").is_some() {
        use std::fmt::Write as _;
        let mut dbg = String::from("ams: semantic scores:");
        for (id, s) in semantic.iter().take(10) {
            let _ = write!(dbg, " {}:{:.3}", id, s);
        }
        eprintln!("{dbg}");
    }
    semantic_gate(&mut semantic);
    semantic.truncate(pool);

    let lexical_ids: Vec<i64> = lexical.iter().map(|(id, _)| *id).collect();
    let semantic_ids: Vec<i64> = semantic.iter().map(|(id, _)| *id).collect();
    let semantic_ids_set: std::collections::HashSet<i64> = semantic_ids.iter().copied().collect();

    // Signal 3: recency, restricted to candidates the other signals found, so
    // it only reorders relevant memories instead of injecting fresh noise.
    let union: Vec<i64> = lexical_ids
        .iter()
        .chain(semantic_ids.iter())
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let created = db::created_at_map(conn, &union).map_err(|e| e.to_string())?;
    let mut recency: Vec<i64> = union.clone();
    recency.sort_by_key(|id| std::cmp::Reverse(created.get(id).copied().unwrap_or(0)));

    let mut lists: Vec<Vec<i64>> = Vec::new();
    if !lexical_ids.is_empty() {
        lists.push(lexical_ids);
    }
    if !semantic_ids.is_empty() {
        lists.push(semantic_ids);
    }
    if !recency.is_empty() {
        lists.push(recency);
    }
    if lists.is_empty() {
        return Ok(Vec::new());
    }

    // Over-fetch before the vote multiplier so a strongly reinforced memory
    // just below the cutoff can still surface.
    let merged = rrf_merge(&lists, 60.0, limit.saturating_mul(2).max(limit + 1));
    let snippets: HashMap<i64, String> = lexical.into_iter().collect();

    let mut hits = Vec::with_capacity(merged.len());
    for (id, score) in merged {
        let Some(memory) = db::get_memory(conn, id).map_err(|e| e.to_string())? else {
            continue;
        };
        let semantic_only = !snippets.contains_key(&id) && semantic_ids_set.contains(&id);
        let snippet = match snippets.get(&id) {
            Some(s) => s.trim().to_string(),
            None => first_line(&memory.content),
        };
        let score = score * vote_multiplier(memory.upvotes, memory.downvotes);
        hits.push(Hit {
            memory,
            snippet,
            semantic_only,
            score,
        });
    }
    // Votes can reorder memories that fused close together.
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(limit);

    let ids: Vec<i64> = hits.iter().map(|h| h.memory.id).collect();
    let _ = db::bump_access(conn, &ids);
    Ok(hits)
}

fn first_line(s: &str) -> String {
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    let mut out: String = line.chars().take(200).collect();
    if line.chars().count() > 200 {
        out.push('…');
    }
    out
}

/// Query tokens long enough to carry meaning, lowercased and deduplicated.
/// Used to keep the OR fallback from matching on filler words.
fn content_tokens(query: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    query
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|t| t.chars().count() >= 3 && seen.insert(t.clone()))
        .collect()
}

/// Rough stem-tolerant check: a token counts as present when the lowercased
/// content contains it. Good enough for gating the OR fallback; FTS5 does the
/// real matching.
fn count_token_hits(content: &str, tokens: &[String]) -> usize {
    let text = content.to_lowercase();
    tokens.iter().filter(|t| text.contains(t.as_str())).count()
}

/// Human-readable hit list used by both the CLI and the MCP `search` tool.
pub fn format_hits(hits: &[Hit]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for h in hits {
        let m = &h.memory;
        let tag = if h.semantic_only { " [semantic]" } else { "" };
        let project = if m.project.is_empty() { "-" } else { &m.project };
        let scope = if m.scope == "shared" {
            " shared"
        } else if m.status == "provisional" {
            " provisional"
        } else {
            ""
        };
        let vote_tag = if m.upvotes > 0 || m.downvotes > 0 {
            format!(" votes ↑{}↓{}", m.upvotes, m.downvotes)
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "[{}] ({}, {}{}) {}{}{} score={:.4}",
            m.id, m.kind, project, scope, m.agent, tag, vote_tag, h.score
        );
        let _ = writeln!(out, "    {}", h.snippet);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_tokens_are_quoted() {
        assert_eq!(fts_match("hello world", false), "\"hello\" \"world\"");
        assert_eq!(fts_match("selfhost*", false), "\"selfhost\"*");
        assert_eq!(fts_match("a:b-c_d", false), "\"a:b-c_d\"");
        assert_eq!(fts_match("say \"hi\"", false), "\"say\" \"\"\"hi\"\"\"");
        assert_eq!(fts_match("a b", true), "\"a\" OR \"b\"");
        assert_eq!(fts_match("   ", false), "");
    }

    #[test]
    fn rrf_rewards_documents_in_both_lists() {
        let lexical = vec![1, 2, 3];
        let semantic = vec![3, 4];
        let merged = rrf_merge(&[lexical, semantic], 60.0, 10);
        assert_eq!(merged[0].0, 3);
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn votes_adjust_but_do_not_dominate() {
        assert_eq!(vote_multiplier(0, 0), 1.0);
        assert!(vote_multiplier(3, 0) > 1.0);
        assert!(vote_multiplier(0, 3) < 1.0);
        assert!(vote_multiplier(100, 0) <= 1.3);
        assert!(vote_multiplier(0, 100) >= 0.7);
    }

    #[test]
    fn votes_reorder_within_limit_after_overfetch() {
        let dir = std::env::temp_dir().join(format!("ams_search_votes_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.db");
        let _ = std::fs::remove_file(&path);
        let conn = crate::db::open_db_at(&path).unwrap();

        let insert = |text: &str| {
            let raw = crate::db::insert_raw(&conn, "a", "", "fact", "own", text, "").unwrap();
            crate::db::insert_memory(
                &conn, raw, "a", "", "fact", "own", text, "", "canonical", 0.6, None,
            )
            .unwrap()
        };
        let best = insert("kubernetes kubernetes kubernetes rollout guide");
        let second = insert("kubernetes kubernetes notes");
        let voted = insert("kubernetes brief mention");
        for (i, agent) in ["a", "b", "c", "d", "e"].iter().enumerate() {
            crate::db::cast_vote(&conn, voted, agent, &format!("s{i}"), 1, "").unwrap();
        }

        let hits = search_with_vector(&conn, "a", None, "kubernetes", 2, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory.id, voted, "vote boost should lift it into the top 2");
        assert!(hits.iter().any(|h| h.memory.id == best || h.memory.id == second));
    }

    #[test]
    fn content_tokens_ignore_filler() {
        assert_eq!(content_tokens("how do I write a haiku"), vec!["how", "write", "haiku"]);
        assert!(content_tokens("a b c").is_empty());
        assert_eq!(content_tokens("Docker docker"), vec!["docker"]);
    }

    #[test]
    fn token_hits_count_distinct_matches() {
        let tokens = content_tokens("docker disk full");
        assert_eq!(count_token_hits("docker build fails, no space on disk", &tokens), 2);
        assert_eq!(count_token_hits("unrelated text", &tokens), 0);
    }

    #[test]
    fn semantic_gate_keeps_only_strong_matches() {
        let mut v = vec![(1, 0.74), (2, 0.70), (3, 0.55), (4, 0.40)];
        semantic_gate(&mut v);
        // 0.70 clears both floors; 0.55 fails the relative floor (0.74*0.85=0.629);
        // 0.40 fails the absolute floor.
        assert_eq!(v.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![1, 2]);

        let mut none = vec![(9, 0.45)];
        semantic_gate(&mut none);
        assert!(none.is_empty(), "noise-only query returns nothing");
    }
}
