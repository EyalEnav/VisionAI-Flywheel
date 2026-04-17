import { useState, useEffect, useRef, useCallback } from "react";

const RENDER_API = "/api";

async function apiFetch(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const r = await fetch(RENDER_API + path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

function useToast() {
  const [toasts, setToasts] = useState([]);
  const show = useCallback((message, type = "info") => {
    const id = Date.now();
    setToasts(t => [...t, { id, message, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 5000);
  }, []);
  return { toasts, show };
}

function Toast({ toasts }) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map(t => (
        <div key={t.id} className={`px-4 py-3 rounded-xl shadow-2xl border text-sm font-medium flex items-center gap-2
          ${t.type === "success" ? "bg-green-900 border-green-600 text-green-200" :
            t.type === "error"   ? "bg-red-900 border-red-600 text-red-200" :
                                   "bg-blue-900 border-blue-600 text-blue-200"}`}>
          {t.type === "success" ? "✅" : t.type === "error" ? "❌" : "ℹ️"} {t.message}
        </div>
      ))}
    </div>
  );
}

const STEPS = [
  { key: "motion",  label: "Kimodo",  emoji: "🦴", end: 20  },
  { key: "render",  label: "SOMA",    emoji: "🎬", end: 65  },
  { key: "cosmos",  label: "Cosmos",  emoji: "🪐", end: 95  },
  { key: "vss",     label: "VSS",     emoji: "🔍", end: 100 },
];

function PipelineBar({ job }) {
  if (!job) return null;
  const p = job.progress || 0;
  const isDone  = job.status === "done";
  const isError = job.status === "error";
  return (
    <div className="space-y-3">
      <div className="flex justify-between text-xs text-gray-400 font-mono">
        <span>{p}%</span>
        <span className={isDone ? "text-green-400" : isError ? "text-red-400" : "text-yellow-300 animate-pulse"}>
          {isDone ? "✅ Complete" : isError ? `❌ ${job.error || "Error"}` : "⟳ Running…"}
        </span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
        <div className={`h-3 rounded-full transition-all duration-700 ${isError ? "bg-red-500" : isDone ? "bg-green-500" : "bg-blue-500 animate-pulse"}`}
          style={{ width: `${Math.max(p, isError ? 100 : 0)}%` }} />
      </div>
      <div className="flex items-center gap-1">
        {STEPS.map((s, i) => {
          const done   = isDone || p >= s.end;
          const active = !done && !isError && p < s.end && (i === 0 || p >= STEPS[i-1].end);
          return (
            <div key={s.key} className="flex items-center flex-1 min-w-0">
              <div className={`flex flex-col items-center flex-1 ${!done && !active ? "opacity-40" : ""}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm border-2
                  ${done ? "bg-green-700 border-green-500" : active ? "bg-blue-700 border-blue-400 animate-pulse" : "bg-gray-800 border-gray-600"}`}>
                  {done ? "✓" : s.emoji}
                </div>
                <span className="text-xs text-gray-500 mt-1 truncate">{s.label}</span>
              </div>
              {i < STEPS.length - 1 && <div className={`h-px w-4 mb-4 ${p >= s.end ? "bg-green-600" : "bg-gray-700"}`} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function LogBox({ lines, title = "Logs" }) {
  const ref = useRef();
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines]);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</span>
        <span className="text-xs text-gray-600 font-mono">{lines?.length || 0} lines</span>
      </div>
      <div ref={ref}
        className="bg-gray-950 border border-gray-700 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs space-y-0.5">
        {lines?.length ? lines.map((l, i) => (
          <div key={i} className={
            l.includes("Error") || l.includes("error") || l.includes("FAILED") ? "text-red-400" :
            l.includes("✅") || l.includes("done") || l.includes("complete") ? "text-green-400" :
            l.includes("⟳") || l.includes("Running") || l.includes("Starting") ? "text-yellow-300" :
            l.includes("🦴") ? "text-orange-400" :
            l.includes("🎬") ? "text-blue-400" :
            l.includes("🪐") ? "text-purple-400" :
            l.includes("🔍") ? "text-cyan-400" :
            "text-gray-400"
          }>{l}</div>
        )) : (
          <div className="text-gray-700 italic">Waiting for logs…</div>
        )}
      </div>
    </div>
  );
}

function VideoCard({ src, label }) {
  const [ok, setOk] = useState(false);
  if (!src) return null;
  return (
    <div className="space-y-1">
      <p className="text-xs text-gray-500 font-mono">{label}</p>
      <video controls loop autoPlay muted
        className="w-full rounded-lg border border-gray-700 bg-gray-950 max-h-72"
        onCanPlay={() => setOk(true)}
        src={src} />
      {!ok && <p className="text-xs text-gray-600 text-center">Loading video…</p>}
    </div>
  );
}

// ─── VSS Panel ────────────────────────────────────────────────────────────────
function VSSPanel({ jobId, preferCosmos }) {
  const [query, setQuery]       = useState("Describe this surveillance footage in detail. Identify any suspicious activity, people, vehicles, or events.");
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const [useCosmosVid, setUseCosmos] = useState(preferCosmos);

  const sendToVSS = async () => {
    if (!jobId || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const fd = new FormData();
      // job_id suffix: "" = SOMA render, "_cosmos" = Cosmos output
      fd.append("job_id", useCosmosVid ? jobId + "_cosmos" : jobId);
      fd.append("prompt", query);
      fd.append("backend", "vss");
      const r = await fetch(RENDER_API + "/analyze", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
      setResult(data.description || data.result || JSON.stringify(data));
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  return (
    <div className="bg-gray-900 border border-cyan-800 rounded-xl p-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-cyan-400 text-lg">🔍</span>
        <span className="text-sm font-semibold text-cyan-300">Send to VSS</span>
      </div>

      {/* Video source selector */}
      <div className="flex gap-3 text-xs">
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="radio" name={`vsrc_${jobId}`} checked={!useCosmosVid} onChange={() => setUseCosmos(false)} className="accent-cyan-500" />
          <span className="text-gray-300">🎬 SOMA Render</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="radio" name={`vsrc_${jobId}`} checked={useCosmosVid} onChange={() => setUseCosmos(true)} className="accent-cyan-500" />
          <span className="text-gray-300">🪐 Cosmos Output</span>
        </label>
      </div>

      {/* Query textarea */}
      <textarea
        value={query}
        onChange={e => setQuery(e.target.value)}
        rows={2}
        className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2.5 text-xs text-white resize-none focus:outline-none focus:border-cyan-600"
        placeholder="Ask VSS anything about this video…"
      />

      {/* Send button */}
      <button
        onClick={sendToVSS}
        disabled={loading}
        className={`w-full py-2 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2
          ${loading
            ? "bg-cyan-900 text-cyan-600 cursor-not-allowed"
            : "bg-cyan-700 hover:bg-cyan-600 text-white"}`}>
        {loading
          ? <><span className="animate-spin">⟳</span> Analyzing…</>
          : "🔍 Analyze with VSS"}
      </button>

      {/* Result */}
      {result && (
        <div className="bg-gray-800 border border-green-700 rounded-lg p-3 space-y-1">
          <p className="text-xs text-green-400 font-semibold">✅ VSS Response</p>
          <p className="text-sm text-gray-200 leading-relaxed">{result}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-950 border border-red-700 rounded-lg p-3">
          <p className="text-xs text-red-400">❌ {error}</p>
        </div>
      )}
    </div>
  );
}

// ─── Generate Tab ─────────────────────────────────────────────────────────────
function GenerateTab({ visible }) {
  const [prompt, setPrompt]   = useState("person pushing through a crowd and falling on a city street");
  const [cosmos, setCosmos]       = useState(true);
  const [edgeWeight, setEdgeWeight] = useState(0.85);
  const [visWeight, setVisWeight]   = useState(0.45);
  const [vssAuto, setVssAuto]       = useState(true);
  const [jobId, setJobId]     = useState(null);
  const [job, setJob]         = useState(null);
  const [running, setRunning] = useState(false);
  const [pollErr, setPollErr] = useState(0);
  const { toasts, show }      = useToast();
  const pollRef               = useRef(null);

  const stopPoll = () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };

  const pollJob = useCallback((id) => {
    stopPoll();
    pollRef.current = setInterval(async () => {
      try {
        const j = await apiFetch(`/jobs/${id}`);
        setJob(j);
        setPollErr(0);
        if (j.status === "done" || j.status === "error") {
          setRunning(false);
          if (j.status === "error") {
            stopPoll();
            show(`Error: ${j.error || "unknown"}`, "error");
          } else {
            show("SOMA ready! 🎬 Cosmos Transfer running…", "success");
            if (j.cosmos_status === "done" || j.cosmos_status === "error") {
              stopPoll();
            }
          }
        }
      } catch(e) {
        setPollErr(n => {
          const next = n + 1;
          if (next >= 5) {
            stopPoll();
            setRunning(false);
            setJob(prev => prev ? { ...prev, status: "error", error: "Lost connection to server" } : prev);
            show("Lost connection to render-api", "error");
          }
          return next;
        });
      }
    }, 1500);
  }, [show]);

  const generate = async () => {
    if (!prompt.trim() || running) return;
    setRunning(true); setJob(null); setJobId(null); setPollErr(0);
    try {
      const fd = new FormData();
      fd.append("prompt", prompt);
      fd.append("texture_mode", cosmos ? "cosmos" : "skeleton");
      fd.append("cosmos_prompt", prompt);
      fd.append("cosmos_edge_weight", edgeWeight);
      fd.append("cosmos_vis_weight", visWeight);
      const resp = await fetch("/api/generate", { method: "POST", body: fd });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      const r = await resp.json();
      if (r.error) { show(r.error, "error"); setRunning(false); return; }
      setJobId(r.job_id);
      setJob({ status: "queued", progress: 0, log: [`Job started: ${r.job_id}`] });
      pollJob(r.job_id);
      show("Pipeline started! 🚀", "info");
    } catch(e) {
      show("Failed to start: " + e.message, "error");
      setRunning(false);
    }
  };

  useEffect(() => () => stopPoll(), []);

  if (!visible) return null;
  return (
    <div className="space-y-5 p-5">
      <Toast toasts={toasts} />

      {/* Prompt */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Scene Prompt</label>
        <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-sm text-white resize-none focus:outline-none focus:border-blue-500"
          placeholder="Describe the motion/scene…" />
      </div>

      {/* Options */}
      <div className="flex gap-6 flex-wrap">
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={cosmos} onChange={e => setCosmos(e.target.checked)} className="w-4 h-4 accent-blue-500" />
          <span className="text-sm text-gray-300">🪐 Cosmos Transfer (Sim2Real)</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={vssAuto} onChange={e => setVssAuto(e.target.checked)} className="w-4 h-4 accent-blue-500" />
          <span className="text-sm text-gray-300">🔍 Auto VSS annotation</span>
        </label>
      </div>

      {/* Cosmos Transfer params */}
      {cosmos && (
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 space-y-3">
          <p className="text-xs font-semibold text-purple-400 uppercase tracking-wider">🪐 Cosmos Transfer Params</p>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-gray-400">
                <span>Edge Weight</span>
                <span className="text-purple-300 font-mono">{edgeWeight.toFixed(2)}</span>
              </div>
              <input type="range" min="0" max="1" step="0.05"
                value={edgeWeight} onChange={e => setEdgeWeight(parseFloat(e.target.value))}
                className="w-full accent-purple-500" />
              <p className="text-xs text-gray-600">Structure preservation (geometry)</p>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-gray-400">
                <span>Vis Weight</span>
                <span className="text-purple-300 font-mono">{visWeight.toFixed(2)}</span>
              </div>
              <input type="range" min="0" max="1" step="0.05"
                value={visWeight} onChange={e => setVisWeight(parseFloat(e.target.value))}
                className="w-full accent-purple-500" />
              <p className="text-xs text-gray-600">Visual/color guidance (0 = edge only)</p>
            </div>
          </div>
          <p className="text-xs text-gray-600">Sweet spot: edge=0.85 + vis=0.45 ✦ Edge only: vis=0</p>
        </div>
      )}

      {/* Pipeline chips */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="px-2 py-1 rounded text-xs border bg-orange-900/40 border-orange-700 text-orange-300 font-mono">🦴 Kimodo</span>
        <span className="text-gray-600">→</span>
        <span className="px-2 py-1 rounded text-xs border bg-blue-900/40 border-blue-700 text-blue-300 font-mono">🎬 SOMA</span>
        {cosmos && <><span className="text-gray-600">→</span>
          <span className="px-2 py-1 rounded text-xs border bg-purple-900/40 border-purple-700 text-purple-300 font-mono">🪐 Cosmos</span></>}
        {vssAuto && <><span className="text-gray-600">→</span>
          <span className="px-2 py-1 rounded text-xs border bg-cyan-900/40 border-cyan-700 text-cyan-300 font-mono">🔍 VSS</span></>}
      </div>

      {/* Generate button */}
      <button onClick={generate} disabled={running}
        className={`w-full py-3 rounded-xl font-bold text-base transition-all flex items-center justify-center gap-2
          ${running ? "bg-gray-700 text-gray-500 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/40"}`}>
        {running ? <><span className="animate-spin">⟳</span> Generating…</> : "🚀 Generate Scene"}
      </button>

      {/* Progress */}
      {job && (
        <div className="bg-gray-900 border border-gray-700 rounded-xl p-4 space-y-4">
          <PipelineBar job={job} />
          <LogBox lines={job.log} />
        </div>
      )}

      {/* Result videos */}
      {job?.status === "done" && jobId && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <VideoCard src={`${RENDER_API}/render/video/${jobId}`} label="🎬 SOMA Render" />
            <VideoCard src={`${RENDER_API}/render/video/${jobId}_cosmos`} label="🪐 Cosmos Output" />
          </div>
          <VSSPanel jobId={jobId} preferCosmos={cosmos} />
        </div>
      )}
    </div>
  );
}

// ─── Preview Tab ──────────────────────────────────────────────────────────────
function PreviewTab({ visible }) {
  const [jobs, setJobs]       = useState([]);
  const [loading, setLoad]    = useState(false);
  const [selected, setSel]    = useState(null);
  const [showVSS, setShowVSS] = useState(false);

  const load = async () => {
    setLoad(true);
    try { setJobs((await apiFetch("/jobs")).jobs || []); } catch(e) {}
    setLoad(false);
  };

  useEffect(() => { if (visible) load(); }, [visible]);

  // Reset VSS panel when switching clips
  useEffect(() => { setShowVSS(false); }, [selected]);

  if (!visible) return null;
  return (
    <div className="p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-white">Generated Clips</h2>
        <button onClick={load} className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-sm text-gray-300 transition">🔄 Refresh</button>
      </div>
      {loading && <p className="text-gray-500 text-sm">Loading…</p>}

      {selected && (
        <div className="bg-gray-900 border border-blue-700 rounded-xl p-4 space-y-4">
          {/* Header */}
          <div className="flex justify-between items-center">
            <span className="text-sm font-mono text-blue-300">{selected.job_id}</span>
            <button onClick={() => setSel(null)} className="text-gray-500 hover:text-white text-xs">✕ close</button>
          </div>

          {/* Videos */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <VideoCard src={`${RENDER_API}/render/video/${selected.job_id}`} label="🎬 SOMA Render" />
            <VideoCard src={`${RENDER_API}/render/video/${selected.job_id}_cosmos`} label="🪐 Cosmos" />
          </div>

          {/* Existing VSS annotation (from auto-run) */}
          {selected.vss_description && !showVSS && (
            <div className="bg-gray-800 rounded-lg p-3">
              <p className="text-xs text-green-400 mb-1">🔍 VSS (auto)</p>
              <p className="text-sm text-gray-200">{selected.vss_description}</p>
            </div>
          )}

          {/* Action row */}
          <div className="flex gap-2">
            <button
              onClick={() => setShowVSS(v => !v)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all
                ${showVSS
                  ? "bg-cyan-800 text-cyan-200 border border-cyan-600"
                  : "bg-cyan-700 hover:bg-cyan-600 text-white"}`}>
              🔍 {showVSS ? "Hide VSS Panel" : "Send to VSS"}
            </button>
            {selected.prompt && (
              <span className="text-xs text-gray-500 italic self-center truncate max-w-xs">"{selected.prompt}"</span>
            )}
          </div>

          {/* VSS Panel (collapsible) */}
          {showVSS && (
            <VSSPanel
              jobId={selected.job_id}
              preferCosmos={selected.cosmos_status === "done"}
            />
          )}
        </div>
      )}

      {/* Clip grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {jobs.map(j => (
          <div key={j.job_id} onClick={() => setSel(j)}
            className={`bg-gray-900 border rounded-xl p-3 cursor-pointer transition-all group
              ${selected?.job_id === j.job_id ? "border-blue-500" : "border-gray-700 hover:border-blue-600"}`}>
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-mono text-gray-400 truncate">{j.job_id?.slice(0,16)}…</span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-mono
                ${j.status === "done" ? "bg-green-900 text-green-300" :
                  j.status === "error" ? "bg-red-900 text-red-300" : "bg-yellow-900 text-yellow-300 animate-pulse"}`}>{j.status}</span>
            </div>
            {j.prompt && <p className="text-xs text-gray-500 truncate italic mb-2">"{j.prompt}"</p>}
            <video muted loop className="w-full rounded-lg max-h-36 bg-gray-950 border border-gray-800 group-hover:border-blue-700 transition"
              src={`${RENDER_API}/render/video/${j.job_id}${j.cosmos_status === "done" ? "_cosmos" : ""}`}
              onMouseEnter={e => e.target.play()}
              onMouseLeave={e => { e.target.pause(); e.target.currentTime = 0; }} />
          </div>
        ))}
        {!loading && jobs.length === 0 && (
          <p className="text-gray-600 text-sm col-span-2 text-center py-8">No clips yet — go generate one!</p>
        )}
      </div>
    </div>
  );
}

// ─── Monitor Tab ─────────────────────────────────────────────────────────────
function MonitorTab({ visible }) {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    if (!visible) return;
    const load = async () => {
      try { setStatus(await apiFetch("/status")); } catch(e) {}
    };
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [visible]);

  if (!visible) return null;
  const services = status?.services || {};
  const gpus     = status?.gpus || [];

  return (
    <div className="p-5 space-y-5">
      <h2 className="text-lg font-bold text-white">System Monitor</h2>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {Object.entries(services).map(([name, info]) => (
          <div key={name} className={`rounded-xl border p-3 space-y-1
            ${info.healthy ? "bg-green-900/20 border-green-700" : "bg-red-900/20 border-red-700"}`}>
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-white capitalize">{name}</span>
              <span className={`text-xs font-mono ${info.healthy ? "text-green-400" : "text-red-400"}`}>
                {info.healthy ? "● UP" : "● DOWN"}
              </span>
            </div>
            <p className="text-xs text-gray-500">{info.status || "—"}</p>
          </div>
        ))}
      </div>

      {gpus.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">GPU Usage</h3>
          {gpus.map((g, i) => {
            const used  = parseInt(g.mem_used)  || 0;
            const total = parseInt(g.mem_total) || 1;
            const pct   = Math.round((used / total) * 100);
            return (
              <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl p-3 space-y-2">
                <div className="flex justify-between text-xs text-gray-300">
                  <span>GPU {i}: {g.name}</span>
                  <span className="font-mono">{g.mem_used} / {g.mem_total}</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div className={`h-2 rounded-full transition-all ${pct > 85 ? "bg-red-500" : pct > 60 ? "bg-yellow-500" : "bg-purple-500"}`}
                    style={{ width: `${pct}%` }} />
                </div>
                <div className="text-xs text-gray-500 font-mono">Util: {g.util} | {pct}% VRAM</div>
              </div>
            );
          })}
        </div>
      )}

      {!status && <p className="text-gray-600 text-sm">Connecting to render-api…</p>}
    </div>
  );
}

// ─── Dataset Tab ─────────────────────────────────────────────────────────────
function DatasetTab({ visible }) {
  const [rows, setRows]    = useState([]);
  const [loading, setLoad] = useState(false);

  const load = async () => {
    setLoad(true);
    try {
      const r = await apiFetch("/annotations");
      setRows(r.annotations || []);
    } catch(e) {}
    setLoad(false);
  };

  useEffect(() => { if (visible) load(); }, [visible]);

  if (!visible) return null;
  return (
    <div className="p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-white">VLM Dataset</h2>
        <button onClick={load} className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-sm text-gray-300 transition">🔄 Refresh</button>
      </div>
      {loading && <p className="text-gray-500 text-sm">Loading…</p>}
      {rows.length > 0 ? (
        <div className="overflow-x-auto rounded-xl border border-gray-700">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-800 text-xs text-gray-400 uppercase">
              <tr>
                <th className="px-3 py-2">File</th>
                <th className="px-3 py-2">VSS Response</th>
                <th className="px-3 py-2">User Annotation</th>
                <th className="px-3 py-2">Tags</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {rows.map((r, i) => (
                <tr key={i} className="hover:bg-gray-800/50 transition">
                  <td className="px-3 py-2 font-mono text-xs text-blue-300 max-w-32 truncate">{r.video_filename}</td>
                  <td className="px-3 py-2 text-gray-400 text-xs max-w-48 truncate">{r.vss_response}</td>
                  <td className="px-3 py-2 text-gray-200 text-xs max-w-48 truncate">{r.user_annotation}</td>
                  <td className="px-3 py-2 text-xs text-purple-300">{Array.isArray(r.tags) ? r.tags.join(", ") : r.tags}</td>
                  <td className="px-3 py-2"><span className="px-2 py-0.5 rounded-full text-xs bg-green-900 text-green-300">{r.tuning_status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !loading && <p className="text-gray-600 text-sm text-center py-8">No annotations yet.</p>
      )}
    </div>
  );
}

// ─── App Shell ────────────────────────────────────────────────────────────────
const TABS = [
  { id: "generate", label: "🚀 Generate" },
  { id: "preview",  label: "🎬 Preview"  },
  { id: "monitor",  label: "📊 Monitor"  },
  { id: "dataset",  label: "🗂 Dataset"  },
];

export default function App() {
  const [tab, setTab] = useState("generate");
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-lg">🎯</div>
        <div>
          <h1 className="text-base font-bold tracking-tight">VisionAI Flywheel</h1>
          <p className="text-xs text-gray-500">Synthetic surveillance dataset pipeline</p>
        </div>
      </div>
      <div className="flex border-b border-gray-800 px-4 pt-2 gap-1">
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-all
              ${tab === t.id ? "bg-gray-800 text-white border border-b-0 border-gray-700" : "text-gray-500 hover:text-gray-300"}`}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="max-w-4xl mx-auto">
        <GenerateTab visible={tab === "generate"} />
        <PreviewTab  visible={tab === "preview"}  />
        <MonitorTab  visible={tab === "monitor"}  />
        <DatasetTab  visible={tab === "dataset"}  />
      </div>
    </div>
  );
}
