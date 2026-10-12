//! Synthetic gRPC load generator for the SASY policy engine.
//!
//! Reads a replay JSONL produced by ``SASY_SERVER_LATENCY_LOG_FILE``
//! (one record per action with session_id, fn_name, args,
//! current_node_ids, plus ts and per-RPC timing/decision), then
//! issues the same ``CheckToolCall`` RPCs concurrently against a
//! running server and records each call's wall-clock RTT to a fresh
//! JSONL.
//!
//! Pure tonic client — no Python, no GIL, so the load generator
//! itself doesn't contaminate measurement past whatever ceiling the
//! local network stack imposes.
//!
//! Records in the *output* log share the SDK's
//! ``SASY_LATENCY_LOG_FILE`` shape (``ts``, ``fn_name``,
//! ``authorized``, ``rtt_us``, ``session_id``), so a reducer written
//! for that log reads these records too. Server-side timing fields
//! (``server_total_us``, ``server_eval_us``, ``server_sync_us``) are
//! always zero: ``ToolCallResponse`` does not carry
//! ``PerformanceTiming``. Authoritative server-side timing is in the
//! matching ``SASY_SERVER_LATENCY_LOG_FILE`` row on the server.
//!
//! A versioned JSON summary sidecar (``<out>.summary.json``) is
//! written after the timed pass with monotonic elapsed time,
//! expected/attempted/successful/failed call counts, and run
//! parameters. Python reducers use the sidecar's ``elapsed_s`` for
//! per-trial throughput instead of the legacy timestamp-span
//! approximation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinSet;
use tonic::transport::{Channel, Endpoint};
use tonic::Request;
use tracing::{info, warn};

use sasy_auth::tls::TlsConfig;
use sasy_common::policy_engine::policy_engine_client::PolicyEngineClient;
use sasy_common::policy_engine::{action::ActionType, Action, EndSessionRequest};
use sasy_common::reference_monitor::rm_proxy_client::RmProxyClient;
use sasy_common::reference_monitor::ToolCallRequest;

/// One row from the replay JSONL. Mirrors the server-side
/// ``ActionRecord`` plus a few optional fields the loader accepts
/// for forward compatibility (extra keys are ignored by serde).
#[derive(Debug, Clone, Deserialize)]
struct ReplayRow {
    #[serde(default)]
    session_id: String,
    fn_name: String,
    #[serde(default)]
    args: String,
    #[serde(default)]
    current_node_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct OutputRecord<'a> {
    ts: f64,
    fn_name: &'a str,
    authorized: bool,
    rtt_us: f64,
    session_id: &'a str,
    server_total_us: u64,
    server_eval_us: u64,
    server_sync_us: u64,
}

/// Versioned JSON summary written as ``<out>.summary.json`` after the
/// timed pass completes. Python reducers read ``elapsed_s`` for
/// authoritative per-trial throughput instead of the legacy
/// first-last timestamp span approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunSummary {
    version: u32,
    mode: String,
    concurrency: usize,
    repeat: usize,
    expected_calls: usize,
    attempted_calls: usize,
    successful_calls: usize,
    failed_calls: usize,
    elapsed_s: f64,
}

#[derive(Parser, Debug)]
#[command(
    about = "Replay-driven load generator for the SASY policy engine.",
    long_about = None,
)]
struct Args {
    /// Server endpoint (e.g. https://localhost:10089).
    #[arg(long, default_value = "https://localhost:10089")]
    url: String,

    /// CA certificate path. Required for mTLS clusters.
    #[arg(long)]
    ca: Option<PathBuf>,

    /// Client cert (mTLS).
    #[arg(long)]
    cert: Option<PathBuf>,

    /// Client private key (mTLS).
    #[arg(long)]
    key: Option<PathBuf>,

    /// API key sent in ``x-api-key`` metadata. Optional; only
    /// needed if the server's auth chain expects it.
    #[arg(long)]
    api_key: Option<String>,

    /// Path to the replay JSONL (server-side latency log).
    #[arg(long)]
    replay: PathBuf,

    /// Output JSONL with per-call timings.
    #[arg(long)]
    out: PathBuf,

    /// Max concurrent in-flight sessions (session mode) or RPCs
    /// (flat mode). Must be at least 1.
    #[arg(long, default_value_t = 16)]
    concurrency: usize,

    /// Number of channels to round-robin across. Each channel is one
    /// HTTP/2 connection; with too few channels the per-channel I/O
    /// task serializes outgoing frames per worker. Empirically, fixing
    /// at 64 (matching SASY_CHANNEL_POOL_SIZE on the SDK side) makes
    /// per-call latency independent of channel count up to c=32.
    #[arg(long, default_value_t = 64)]
    channels: usize,

    /// Limit on rows replayed (head N); 0 = all.
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Repeat the replay N times. In session mode, each recorded
    /// session runs N complete passes through the bounded ready
    /// queue — finish one pass, release the active slot, re-enqueue
    /// if repetitions remain. In flat mode, the global row stream
    /// cycles N times. Must be at least 1.
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// Skip rows whose ``fn_name`` matches this set
    /// (comma-separated). Useful for excluding HTTP / message
    /// actions when only tool calls are interesting.
    #[arg(long, default_value = "")]
    skip_fns: String,

    /// Replay scheduling model.
    ///
    /// * ``session`` (default) — group rows by session_id and drive
    ///   each session serially. A semaphore caps active sessions to
    ///   ``--concurrency``. Within a session, calls go out in their
    ///   original order with at most one in-flight at a time. With
    ///   ``--repeat N > 1``, each session runs N complete passes
    ///   through a bounded ready queue: finish one pass, release the
    ///   active slot, re-enqueue at the back if repetitions remain.
    ///   At most one pass per session ID is active at any time.
    ///   Request payloads are shared (not copied per repeat).
    /// * ``flat`` — one global semaphore caps total in-flight
    ///   RPCs to ``--concurrency`` regardless of which
    ///   session each belongs to. Useful for stress-testing the
    ///   server above realistic concurrency.
    #[arg(long, value_enum, default_value_t = ScheduleMode::Session)]
    mode: ScheduleMode,

    /// Send ``EndSession`` for every session_id touched after the
    /// timed pass completes. Drops the per-session evaluator on the
    /// server (graph state is preserved). Default on so successive
    /// loadgen invocations against a long-lived server don't see
    /// each other's stale evaluators in the SessionEvaluatorMap or
    /// broadcast subscriber list.
    #[arg(long, default_value_t = true,
        action = clap::ArgAction::Set)]
    cleanup_sessions: bool,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ScheduleMode {
    Session,
    Flat,
}

fn validate_args(repeat: usize, concurrency: usize) -> Result<()> {
    if repeat == 0 {
        return Err(anyhow!("--repeat must be at least 1"));
    }
    if concurrency == 0 {
        return Err(anyhow!("--concurrency must be at least 1"));
    }
    Ok(())
}

fn checked_total_calls(rows: usize, repeat: usize) -> Result<usize> {
    rows.checked_mul(repeat).ok_or_else(|| {
        anyhow!(
            "total call count overflows: {} rows × {} repeats",
            rows,
            repeat,
        )
    })
}

fn build_endpoint(args: &Args) -> Result<Endpoint> {
    let mut ep = Endpoint::from_shared(args.url.clone())
        .with_context(|| format!("invalid url: {}", args.url))?
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(15));

    let tls_cfg = TlsConfig {
        cert_path: args.cert.as_ref().map(|p| p.to_string_lossy().to_string()),
        key_path: args.key.as_ref().map(|p| p.to_string_lossy().to_string()),
        ca_path: args.ca.as_ref().map(|p| p.to_string_lossy().to_string()),
    };
    if tls_cfg.is_configured() {
        let client_tls = tls_cfg
            .load_client_config()
            .map_err(|e| anyhow!("tls config: {e:?}"))?;
        ep = ep
            .tls_config(client_tls)
            .with_context(|| "tls config build")?;
    }
    Ok(ep)
}

async fn build_pool(args: &Args) -> Result<Vec<RmProxyClient<Channel>>> {
    let ep = build_endpoint(args)?;
    let mut clients = Vec::with_capacity(args.channels);
    for _ in 0..args.channels.max(1) {
        let ch = ep
            .clone()
            .connect()
            .await
            .with_context(|| format!("connect to {}", args.url))?;
        clients.push(RmProxyClient::new(ch));
    }
    Ok(clients)
}

async fn load_replay(path: &std::path::Path) -> Result<Vec<ReplayRow>> {
    let file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("open replay {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut rows = Vec::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<ReplayRow>(trimmed) {
            Ok(r) => rows.push(r),
            Err(e) => warn!("skipping unparseable replay line: {e}"),
        }
    }
    Ok(rows)
}

/// Drive the bounded session-mode queue. Maintains the production
/// invariants: stable initial order, exactly ``repeat`` passes per
/// session, strict within-pass call order, bounded active work, and
/// no overlapping copies of the same session ID.
async fn drive_session_queue<F, Fut>(
    order: Vec<String>,
    repeat: usize,
    concurrency: usize,
    spawn_pass: F,
) -> Result<()>
where
    F: Fn(String, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let queue: Arc<std::sync::Mutex<VecDeque<(String, usize)>>> = Arc::new(std::sync::Mutex::new(
        order.iter().map(|s| (s.clone(), 0usize)).collect(),
    ));
    let sem = Arc::new(Semaphore::new(concurrency));
    let mut join_set = JoinSet::new();

    loop {
        while let Some(res) = join_set.try_join_next() {
            res.context("session worker panicked or was cancelled")??;
        }

        let item = queue.lock().unwrap().pop_front();
        let Some((sid, pass)) = item else {
            if join_set.is_empty() {
                break;
            }
            if let Some(res) = join_set.join_next().await {
                res.context("session worker panicked or was cancelled")??;
            }
            continue;
        };

        let permit = sem.clone().acquire_owned().await.unwrap();
        let queue_ref = queue.clone();
        let fut = spawn_pass(sid.clone(), pass);

        join_set.spawn(async move {
            let _permit = permit;
            let result = fut.await;
            if pass + 1 < repeat {
                queue_ref.lock().unwrap().push_back((sid, pass + 1));
            }
            result
        });
    }

    while let Some(res) = join_set.join_next().await {
        res.context("session worker panicked or was cancelled")??;
    }

    Ok(())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // rustls 0.23 requires the process-level CryptoProvider to be
    // installed before any TLS handshake. Mirror sasy-binary's
    // setup so the loadgen + server agree on the provider.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    validate_args(args.repeat, args.concurrency)?;

    let skip_fns: HashSet<String> = args
        .skip_fns
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    info!(
        "loading replay from {} (concurrency={}, channels={}, repeat={})",
        args.replay.display(),
        args.concurrency,
        args.channels,
        args.repeat,
    );
    let mut rows = load_replay(&args.replay).await?;
    if args.limit > 0 && rows.len() > args.limit {
        rows.truncate(args.limit);
    }
    if !skip_fns.is_empty() {
        rows.retain(|r| !skip_fns.contains(&r.fn_name));
    }
    info!("replay rows after filter: {}", rows.len());
    if rows.is_empty() {
        return Err(anyhow!("no rows to replay"));
    }

    let total_calls = checked_total_calls(rows.len(), args.repeat)?;

    let clients = build_pool(&args).await?;
    info!("connected {} channel(s) to {}", clients.len(), args.url);

    // Warmup: fire one call per *distinct session* in the replay
    // before timing starts. The server's per-session evaluator
    // architecture spawns one Soufflé subprocess per session on
    // first call and bootstraps it from RocksDB — that cost
    // (~10-20ms per session) is a real production cost on cold
    // starts but contaminates steady-state throughput numbers
    // when bundled into the timed window. The microbench reports
    // the steady-state cost; per-session bootstrap is its own
    // story (and small relative to the LLM-dominated end-to-end
    // path that pays it once per conversation).
    //
    // Also covers the per-channel HTTP/2 SETTINGS exchange.
    let warmup_rows: Vec<&ReplayRow> = {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut out = Vec::new();
        for row in &rows {
            if seen.insert(row.session_id.as_str()) {
                out.push(row);
            }
        }
        out
    };
    info!("warming {} session(s) before timed run", warmup_rows.len());
    {
        let mut handles = Vec::with_capacity(warmup_rows.len());
        for (i, row) in warmup_rows.into_iter().enumerate() {
            let mut client = clients[i % clients.len()].clone();
            let api_key = args.api_key.clone();
            let row = row.clone();
            handles.push(tokio::spawn(async move {
                let req = build_request(&row);
                let mut r = Request::new(req);
                if let Some(ref k) = api_key {
                    if let Ok(v) = k.parse() {
                        r.metadata_mut().insert(sasy_common::headers::API_KEY, v);
                    }
                }
                if let Err(e) = client.check_tool_call(r).await {
                    warn!(
                        "warmup call failed (continuing) session={} fn={}: {}",
                        row.session_id, row.fn_name, e,
                    );
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
    }

    let out_file = tokio::fs::File::create(&args.out)
        .await
        .with_context(|| format!("create output {}", args.out.display()))?;
    let out = Arc::new(Mutex::new(out_file));
    let sem = Arc::new(Semaphore::new(args.concurrency));
    let api_key = args.api_key.clone();
    let successful = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));

    // Collect unique session IDs before dispatching (needed for
    // cleanup and for fn-count logging).
    let unique_sessions: HashSet<String> = rows.iter().map(|r| r.session_id.clone()).collect();

    let mut counts: HashMap<String, u64> = HashMap::new();
    for row in rows.iter() {
        *counts.entry(row.fn_name.clone()).or_default() += 1;
    }

    info!(
        "issuing {} calls in {:?} mode (concurrency={})",
        total_calls, args.mode, args.concurrency,
    );

    // Authoritative monotonic timer: starts before timed dispatch,
    // stops after all timed requests AND output logging complete.
    // Excludes warmup, session cleanup and subprocess startup.
    let start = Instant::now();

    match args.mode {
        ScheduleMode::Flat => {
            let rows = Arc::new(rows);
            let mut join_set = JoinSet::new();
            for i in 0..total_calls {
                while let Some(res) = join_set.try_join_next() {
                    res.context("flat-mode worker panicked or was cancelled")?;
                }

                let idx = i % rows.len();
                let rows = rows.clone();
                let client = clients[i % clients.len()].clone();
                let permit = sem.clone().acquire_owned().await.unwrap();
                let out = out.clone();
                let api_key = api_key.clone();
                let successful = successful.clone();
                let failed = failed.clone();
                join_set.spawn(async move {
                    let _p = permit;
                    if issue_one(&rows[idx], client, api_key.as_deref(), out).await {
                        successful.fetch_add(1, Ordering::Relaxed);
                    } else {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            while let Some(res) = join_set.join_next().await {
                res.context("flat-mode worker panicked or was cancelled")?;
            }
        }
        ScheduleMode::Session => {
            let mut by_session_vec: HashMap<String, Vec<ReplayRow>> = HashMap::new();
            let mut order: Vec<String> = Vec::new();
            for row in rows.into_iter() {
                if !by_session_vec.contains_key(&row.session_id) {
                    order.push(row.session_id.clone());
                }
                by_session_vec
                    .entry(row.session_id.clone())
                    .or_default()
                    .push(row);
            }
            let by_session: Arc<HashMap<String, Arc<Vec<ReplayRow>>>> = Arc::new(
                by_session_vec
                    .into_iter()
                    .map(|(k, v)| (k, Arc::new(v)))
                    .collect(),
            );

            info!(
                "session mode: {} distinct sessions, max active = {}, repeat = {}",
                order.len(),
                args.concurrency,
                args.repeat,
            );

            let clients = Arc::new(clients);
            let channel_idx = Arc::new(AtomicUsize::new(0));
            {
                let out = out.clone();
                let api_key = api_key.clone();
                let successful = successful.clone();
                let failed = failed.clone();
                drive_session_queue(order, args.repeat, args.concurrency, move |sid, _pass| {
                    let session_rows = by_session[&sid].clone();
                    let idx = channel_idx.fetch_add(1, Ordering::Relaxed);
                    let client = clients[idx % clients.len()].clone();
                    let out = out.clone();
                    let api_key = api_key.clone();
                    let successful = successful.clone();
                    let failed = failed.clone();
                    async move {
                        let mut c = client;
                        for row in session_rows.iter() {
                            if issue_one(row, c.clone(), api_key.as_deref(), out.clone()).await {
                                successful.fetch_add(1, Ordering::Relaxed);
                            } else {
                                failed.fetch_add(1, Ordering::Relaxed);
                            }
                            let _ = &mut c;
                        }
                        Ok(())
                    }
                })
                .await?;
            }
        }
    }

    // Flush output BEFORE stopping the timed-pass clock so the
    // elapsed time covers all writes.
    {
        let mut f = out.lock().await;
        f.flush().await?;
    }
    let elapsed = start.elapsed();

    let succ = successful.load(Ordering::Relaxed);
    let fail = failed.load(Ordering::Relaxed);
    let attempted = succ + fail;

    let calls_per_sec = succ as f64 / elapsed.as_secs_f64();
    info!(
        "done: {} calls in {:.2}s = {:.1} calls/s ({} succeeded, {} failed)",
        attempted,
        elapsed.as_secs_f64(),
        calls_per_sec,
        succ,
        fail,
    );
    let mut top: Vec<_> = counts.into_iter().collect();
    top.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (k, v) in top.iter().take(10) {
        info!("  {:>6} {}", v, k);
    }

    // Write versioned summary sidecar adjacent to the output JSONL.
    let summary = RunSummary {
        version: 1,
        mode: format!("{:?}", args.mode).to_lowercase(),
        concurrency: args.concurrency,
        repeat: args.repeat,
        expected_calls: total_calls,
        attempted_calls: attempted,
        successful_calls: succ,
        failed_calls: fail,
        elapsed_s: elapsed.as_secs_f64(),
    };
    let summary_path = {
        let mut name = args.out.as_os_str().to_owned();
        name.push(".summary.json");
        PathBuf::from(name)
    };
    let summary_json =
        serde_json::to_string_pretty(&summary).with_context(|| "serialize run summary")?;
    tokio::fs::write(&summary_path, summary_json)
        .await
        .with_context(|| format!("write summary {}", summary_path.display()))?;
    info!("summary: {}", summary_path.display());

    if args.cleanup_sessions {
        info!(
            "cleanup: ending {} session(s) so the next invocation starts \
             with a clean SessionEvaluatorMap",
            unique_sessions.len(),
        );
        let ep = build_endpoint(&args)?;
        let mut pe_clients: Vec<PolicyEngineClient<Channel>> =
            Vec::with_capacity(args.channels.clamp(1, 8));
        for _ in 0..args.channels.clamp(1, 8) {
            let ch = ep
                .clone()
                .connect()
                .await
                .with_context(|| "cleanup: connect")?;
            pe_clients.push(PolicyEngineClient::new(ch));
        }
        let api_key = args.api_key.clone();
        let mut handles = Vec::with_capacity(unique_sessions.len());
        for (i, sid) in unique_sessions.into_iter().enumerate() {
            let mut c = pe_clients[i % pe_clients.len()].clone();
            let api_key = api_key.clone();
            handles.push(tokio::spawn(async move {
                let mut req = Request::new(EndSessionRequest {
                    session_id: sid.clone(),
                });
                if let Some(ref k) = api_key {
                    if let Ok(v) = k.parse() {
                        req.metadata_mut().insert(sasy_common::headers::API_KEY, v);
                    }
                }
                if let Err(e) = c.end_session(req).await {
                    warn!("end_session failed for {sid}: {e}");
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
    }

    if fail > 0 {
        return Err(anyhow!(
            "{fail} of {attempted} calls failed during the timed pass"
        ));
    }
    if attempted != total_calls {
        return Err(anyhow!(
            "incomplete pass: {attempted} of {total_calls} calls attempted"
        ));
    }

    Ok(())
}

/// Serialize one output record and write it as a JSONL line.
async fn write_output_record<W: AsyncWrite + Unpin>(
    rec: &OutputRecord<'_>,
    writer: &mut W,
) -> std::io::Result<()> {
    let mut buf = serde_json::to_vec(rec)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    buf.push(b'\n');
    writer.write_all(&buf).await
}

/// Issue a single ``CheckToolCall`` RPC and write the result to the
/// output JSONL. Returns ``true`` only when the RPC succeeded AND the
/// response record was serialized and fully written.
async fn issue_one(
    row: &ReplayRow,
    mut client: RmProxyClient<Channel>,
    api_key: Option<&str>,
    out: Arc<Mutex<tokio::fs::File>>,
) -> bool {
    let req = build_request(row);
    let mut r = Request::new(req);
    if let Some(k) = api_key {
        if let Ok(v) = k.parse() {
            r.metadata_mut().insert(sasy_common::headers::API_KEY, v);
        }
    }
    let t0 = Instant::now();
    let res = client.check_tool_call(r).await;
    let rtt_us = t0.elapsed().as_micros() as f64;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    match res {
        Ok(resp) => {
            let inner = resp.into_inner();
            let rec = OutputRecord {
                ts,
                fn_name: &row.fn_name,
                authorized: inner.authorized,
                rtt_us,
                session_id: &row.session_id,
                server_total_us: 0,
                server_eval_us: 0,
                server_sync_us: 0,
            };
            let mut f = out.lock().await;
            if let Err(e) = write_output_record(&rec, &mut *f).await {
                warn!(
                    "record write failed fn={} session={}: {}",
                    row.fn_name, row.session_id, e,
                );
                return false;
            }
            true
        }
        Err(e) => {
            warn!(
                "call failed fn={} session={}: {}",
                row.fn_name, row.session_id, e,
            );
            false
        }
    }
}

fn build_request(row: &ReplayRow) -> ToolCallRequest {
    ToolCallRequest {
        fn_name: row.fn_name.clone(),
        args: row.args.clone(),
        input_node_ids: row.current_node_ids.clone(),
        session_id: Some(row.session_id.clone()).filter(|s| !s.is_empty()),
        entity: None,
        metadata: vec![],
    }
}

// Make the Action / ActionType imports used so they're not dead code
// when callers want them in tests.
#[allow(dead_code)]
fn _action_type_ref(action: &Action) -> Option<&ActionType> {
    action.action_type.as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_zero_repeat() {
        assert!(validate_args(0, 4).is_err());
    }

    #[test]
    fn reject_zero_concurrency() {
        assert!(validate_args(1, 0).is_err());
    }

    #[test]
    fn accept_valid_args() {
        assert!(validate_args(1, 1).is_ok());
        assert!(validate_args(10, 32).is_ok());
    }

    #[test]
    fn checked_multiply_overflow() {
        assert!(checked_total_calls(usize::MAX, 2).is_err());
        assert!(checked_total_calls(usize::MAX / 2 + 1, 2).is_err());
    }

    #[test]
    fn checked_multiply_normal() {
        assert_eq!(checked_total_calls(100, 5).unwrap(), 500);
        assert_eq!(checked_total_calls(1, 1).unwrap(), 1);
        assert_eq!(checked_total_calls(0, 100).unwrap(), 0);
    }

    #[test]
    fn summary_serialization_roundtrip() {
        let summary = RunSummary {
            version: 1,
            mode: "session".into(),
            concurrency: 4,
            repeat: 3,
            expected_calls: 120,
            attempted_calls: 120,
            successful_calls: 118,
            failed_calls: 2,
            elapsed_s: 1.234,
        };
        let json = serde_json::to_string_pretty(&summary).unwrap();
        let rt: RunSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(rt.version, 1);
        assert_eq!(rt.failed_calls, 2);
        assert_eq!(rt.successful_calls, 118);
        assert!((rt.elapsed_s - 1.234).abs() < 1e-6);
    }

    #[test]
    fn summary_path_convention() {
        let out = PathBuf::from("trial.jsonl");
        let mut name = out.as_os_str().to_owned();
        name.push(".summary.json");
        let p = PathBuf::from(name);
        assert_eq!(p.to_str().unwrap(), "trial.jsonl.summary.json");
    }

    /// Verify all scheduler invariants via the production
    /// ``drive_session_queue`` with unequal session lengths.
    #[tokio::test]
    async fn session_repeat_bounded_queue_unequal_lengths() {
        let by_session: Arc<HashMap<String, Arc<Vec<String>>>> = Arc::new(
            [
                (
                    "s1".into(),
                    Arc::new(vec!["a".into(), "b".into(), "c".into()]),
                ),
                ("s2".into(), Arc::new(vec!["d".into()])),
                ("s3".into(), Arc::new(vec!["e".into(), "f".into()])),
            ]
            .into_iter()
            .collect(),
        );
        let order = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];
        let repeat = 3usize;
        let concurrency = 2usize;

        let events: Arc<std::sync::Mutex<Vec<(String, usize, String)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        {
            let events = events.clone();
            let by_session = by_session.clone();
            drive_session_queue(order.clone(), repeat, concurrency, move |sid, pass| {
                let events = events.clone();
                let session_reqs = by_session[&sid].clone();
                async move {
                    events
                        .lock()
                        .unwrap()
                        .push((sid.clone(), pass, "start".into()));
                    for (i, _) in session_reqs.iter().enumerate() {
                        events
                            .lock()
                            .unwrap()
                            .push((sid.clone(), pass, format!("call:{i}")));
                        tokio::task::yield_now().await;
                    }
                    events
                        .lock()
                        .unwrap()
                        .push((sid.clone(), pass, "end".into()));
                    Ok(())
                }
            })
            .await
            .unwrap();
        }

        let events = events.lock().unwrap();

        // 1. Exact pass count per session.
        let mut pass_counts: HashMap<&str, usize> = HashMap::new();
        for (sid, _, typ) in events.iter() {
            if typ == "start" {
                *pass_counts.entry(sid.as_str()).or_default() += 1;
            }
        }
        for sid in &order {
            assert_eq!(
                pass_counts[sid.as_str()],
                repeat,
                "session {sid} should have {repeat} passes"
            );
        }

        // 2. No simultaneous passes for any session.
        for sid in &order {
            let mut depth = 0i32;
            for (s, _, typ) in events.iter() {
                if s == sid {
                    if typ == "start" {
                        depth += 1;
                        assert!(depth <= 1, "overlapping passes for session {sid}");
                    } else if typ == "end" {
                        depth -= 1;
                    }
                }
            }
            assert_eq!(depth, 0, "unmatched start/end for session {sid}");
        }

        // 3. Correct total call count.
        let total_calls: usize = events
            .iter()
            .filter(|(_, _, t)| t.starts_with("call:"))
            .count();
        let expected: usize = by_session.values().map(|v| v.len()).sum::<usize>() * repeat;
        assert_eq!(total_calls, expected);

        // 4. Request order preserved within each pass of each session.
        for sid in &order {
            let session_len = by_session[sid].len();
            for p in 0..repeat {
                let indices: Vec<usize> = events
                    .iter()
                    .filter(|(s, pass, t)| s == sid && *pass == p && t.starts_with("call:"))
                    .map(|(_, _, t)| t.strip_prefix("call:").unwrap().parse::<usize>().unwrap())
                    .collect();
                let expected: Vec<usize> = (0..session_len).collect();
                assert_eq!(indices, expected, "wrong request order for {sid} pass {p}");
            }
        }
    }

    /// Single repeat should produce exactly one pass per session.
    #[tokio::test]
    async fn session_repeat_one_is_single_pass() {
        let by_session: Arc<HashMap<String, Arc<Vec<String>>>> = Arc::new(
            [
                ("s1".into(), Arc::new(vec!["a".into(), "b".into()])),
                ("s2".into(), Arc::new(vec!["c".into()])),
            ]
            .into_iter()
            .collect(),
        );
        let order = vec!["s1".to_string(), "s2".to_string()];

        let events: Arc<std::sync::Mutex<Vec<(String, usize, String)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        {
            let events = events.clone();
            let by_session = by_session.clone();
            drive_session_queue(order, 1, 1, move |sid, pass| {
                let events = events.clone();
                let reqs = by_session[&sid].clone();
                async move {
                    events
                        .lock()
                        .unwrap()
                        .push((sid.clone(), pass, "start".into()));
                    for (i, _) in reqs.iter().enumerate() {
                        events
                            .lock()
                            .unwrap()
                            .push((sid.clone(), pass, format!("call:{i}")));
                    }
                    events
                        .lock()
                        .unwrap()
                        .push((sid.clone(), pass, "end".into()));
                    Ok(())
                }
            })
            .await
            .unwrap();
        }

        let events = events.lock().unwrap();
        let starts: usize = events.iter().filter(|(_, _, t)| t == "start").count();
        assert_eq!(starts, 2);
        let calls: usize = events
            .iter()
            .filter(|(_, _, t)| t.starts_with("call:"))
            .count();
        assert_eq!(calls, 3); // 2 + 1
    }

    /// Write-error on the production record-writing helper propagates.
    #[tokio::test]
    async fn write_record_propagates_io_error() {
        struct FailWriter;

        impl tokio::io::AsyncWrite for FailWriter {
            fn poll_write(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
                _buf: &[u8],
            ) -> std::task::Poll<std::io::Result<usize>> {
                std::task::Poll::Ready(Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "synthetic write error",
                )))
            }
            fn poll_flush(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<std::io::Result<()>> {
                std::task::Poll::Ready(Ok(()))
            }
            fn poll_shutdown(
                self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<std::io::Result<()>> {
                std::task::Poll::Ready(Ok(()))
            }
        }

        let rec = OutputRecord {
            ts: 1.0,
            fn_name: "test_fn",
            authorized: true,
            rtt_us: 100.0,
            session_id: "s1",
            server_total_us: 0,
            server_eval_us: 0,
            server_sync_us: 0,
        };
        let mut writer = FailWriter;
        let result = write_output_record(&rec, &mut writer).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::BrokenPipe);
    }

    /// Scheduler propagates a worker panic as an error.
    #[tokio::test]
    async fn scheduler_propagates_worker_panic() {
        let order = vec!["s1".to_string()];
        let result = drive_session_queue(order, 1, 1, |_sid, _pass| async {
            panic!("synthetic panic in worker");
        })
        .await;
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("panic") || msg.contains("cancelled"),
            "error should mention panic: {msg}"
        );
    }

    /// Scheduler propagates a worker error without losing it.
    #[tokio::test]
    async fn scheduler_propagates_worker_error() {
        let order = vec!["s1".to_string(), "s2".to_string()];
        let result = drive_session_queue(order, 1, 2, |sid, _pass| async move {
            if sid == "s1" {
                Err(anyhow!("synthetic error for {sid}"))
            } else {
                Ok(())
            }
        })
        .await;
        assert!(result.is_err());
    }
}
