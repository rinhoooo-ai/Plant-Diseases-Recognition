// frontend/src/App.jsx
import { useState, useRef, useCallback } from "react";

function ConfidenceBar({ value, label, rank }) {
  const pct = Math.round(value * 100);
  const barColors = [
    "bg-emerald-500", "bg-blue-500", "bg-violet-500",
    "bg-amber-400", "bg-rose-400"
  ];
  const color = barColors[rank] || "bg-gray-400";
  return (
    <div className="mb-2">
      <div className="flex justify-between items-center mb-0.5">
        <span className="text-xs text-gray-700 font-medium truncate max-w-[220px]" title={label}>
          {rank + 1}. {label}
        </span>
        <span className="text-xs font-bold text-gray-800 ml-2">{pct}%</span>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2">
        <div
          className={`${color} h-2 rounded-full transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function ReasoningTrace({ text }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-4 border border-indigo-200 rounded-xl bg-indigo-50 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-indigo-800 hover:bg-indigo-100 transition"
      >
        <span className="flex items-center gap-2">
          <span>🔬</span> VLM Symptom Analysis
        </span>
        <span className={`transform transition-transform ${open ? "rotate-180" : ""}`}>▾</span>
      </button>
      {open && (
        <div className="px-4 pb-4">
          <p className="text-sm text-indigo-900 leading-relaxed whitespace-pre-wrap">{text}</p>
        </div>
      )}
    </div>
  );
}

function UploadZone({ onFile, preview }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  }, [onFile]);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-200
        flex flex-col items-center justify-center
        ${drag ? "border-green-400 bg-green-50 scale-[1.01]" : "border-gray-300 bg-gray-50 hover:border-green-400 hover:bg-green-50"}
        ${preview ? "h-64" : "h-52"}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      {preview ? (
        <img src={preview} alt="Preview" className="max-h-60 max-w-full rounded-xl object-contain shadow" />
      ) : (
        <>
          <span className="text-4xl mb-2">🌿</span>
          <p className="text-sm font-medium text-gray-600">
            Drag & drop hoặc <span className="text-green-600 underline">chọn ảnh lá cây</span>
          </p>
          <p className="text-xs text-gray-400 mt-1">JPG, PNG, WebP · tối đa 10MB</p>
        </>
      )}
    </div>
  );
}

function ResultCard({ result }) {
  const { disease, confidence, reasoning_trace, top_candidates, latency_ms } = result;
  const pct = Math.round(confidence * 100);
  const isHealthy = /healthy/i.test(disease);
  const totalScore = top_candidates.reduce((s, c) => s + c.score, 0) || 1;
  const candidatesWithConf = top_candidates.map(c => ({ ...c, conf: c.score / totalScore }));

  return (
    <div className="mt-6 bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
      <div className={`px-6 py-5 ${isHealthy ? "bg-green-600" : "bg-red-600"}`}>
        <div className="flex items-start justify-between">
          <div>
            <p className="text-white/80 text-xs font-medium uppercase tracking-wider mb-1">Kết quả chẩn đoán</p>
            <h2 className="text-white text-xl font-bold leading-tight">{disease}</h2>
          </div>
          <div className="text-right">
            <div className="text-white/80 text-xs mb-0.5">Confidence</div>
            <div className="text-white text-2xl font-black">{pct}%</div>
          </div>
        </div>
        <div className="mt-3 w-full bg-white/20 rounded-full h-1.5">
          <div className="bg-white rounded-full h-1.5 transition-all duration-1000" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <div className="px-6 py-5">
        <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Top 5 Candidates</h3>
        {candidatesWithConf.map((c, i) => (
          <ConfidenceBar key={c.disease} label={c.disease} value={c.conf} rank={i} />
        ))}
        <ReasoningTrace text={reasoning_trace} />
        <p className="text-right text-xs text-gray-400 mt-3">⏱ {latency_ms.toLocaleString()} ms</p>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [stage, setStage] = useState("");

  const handleFile = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const stages = [
      "Extracting CLIP embeddings...",
      "Qwen3-VL analyzing symptoms...",
      "FAISS dual retrieval...",
      "Finalizing prediction...",
    ];
    let si = 0;
    setStage(stages[0]);
    const stageTimer = setInterval(() => {
      si = Math.min(si + 1, stages.length - 1);
      setStage(stages[si]);
    }, 3000);

    try {
      const form = new FormData();
      form.append("file", file);

      // Gọi Vercel proxy thay vì backend trực tiếp
      const res = await fetch("/api/predict", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Lỗi không xác định" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      clearInterval(stageTimer);
      setLoading(false);
      setStage("");
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-emerald-50">
      <header className="bg-white/80 backdrop-blur border-b border-green-100 sticky top-0 z-10">
        <div className="max-w-lg mx-auto px-4 py-4 flex items-center gap-3">
          <span className="text-2xl">🌱</span>
          <div>
            <h1 className="text-lg font-bold text-gray-900 leading-none">Plant Disease AI</h1>
            <p className="text-xs text-gray-500">CLIP + Qwen3-VL + FAISS</p>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto px-4 py-6">
        <UploadZone onFile={handleFile} preview={preview} />

        <div className="flex gap-3 mt-4">
          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className={`
              flex-1 py-3 rounded-xl font-semibold text-sm transition-all duration-200
              ${file && !loading
                ? "bg-green-600 hover:bg-green-700 text-white shadow-md hover:shadow-lg active:scale-[0.98]"
                : "bg-gray-200 text-gray-400 cursor-not-allowed"}
            `}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                Analyzing...
              </span>
            ) : "🔍 Analyze Disease"}
          </button>

          {(preview || result) && (
            <button
              onClick={handleReset}
              className="px-4 py-3 rounded-xl border border-gray-300 text-gray-600 text-sm hover:bg-gray-50 transition"
            >
              Reset
            </button>
          )}
        </div>

        {loading && stage && (
          <div className="mt-4 bg-indigo-50 border border-indigo-200 rounded-xl px-4 py-3">
            <div className="flex items-center gap-3">
              <div className="flex gap-1">
                {[0,1,2].map(i => (
                  <span key={i} className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
              <p className="text-sm text-indigo-700 font-medium">{stage}</p>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-xl px-4 py-3">
            <p className="text-sm text-red-700">❌ {error}</p>
          </div>
        )}

        {result && <ResultCard result={result} />}

        {!result && !loading && (
          <div className="mt-8 grid grid-cols-3 gap-3">
            {[
              { icon: "🖼️", title: "CLIP ViT-L/14", desc: "Image embedding" },
              { icon: "🤖", title: "Qwen3-VL", desc: "Symptom verbalization" },
              { icon: "⚡", title: "FAISS Dual", desc: "38-class retrieval" },
            ].map(card => (
              <div key={card.title} className="bg-white rounded-xl border border-gray-100 p-3 text-center shadow-sm">
                <div className="text-2xl mb-1">{card.icon}</div>
                <div className="text-xs font-bold text-gray-800">{card.title}</div>
                <div className="text-xs text-gray-500">{card.desc}</div>
