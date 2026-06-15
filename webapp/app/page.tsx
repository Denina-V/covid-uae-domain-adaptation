"use client";

import { useState } from "react";
import Image from "next/image";

const EDA_PANELS = [
  {
    src: "/panel1_raw_vs_smooth.png",
    title: "Raw vs Smoothed Signal",
    insight: "Raw case counts are noisy. 7-day smoothing is non-negotiable before any modelling.",
  },
  {
    src: "/panel2_missingness_heatmap.png",
    title: "Missing Value Pattern (UAE)",
    insight: "UAE's R-value is missing ~39% of the time in clusters — not random noise, a structural labelling problem.",
  },
  {
    src: "/panel3_missing_by_country.png",
    title: "Missing Rate by Country",
    insight: "UAE has dramatically higher missingness than Germany, India, or Singapore across all key features.",
  },
  {
    src: "/panel4_domain_shift.png",
    title: "Domain Shift Across Countries",
    insight: "Same disease, four completely different temporal signatures. A model trained on Germany fails on UAE.",
  },
  {
    src: "/panel5_r_distribution.png",
    title: "Reproduction Rate Distribution",
    insight: "UAE's R-values mostly below 1 with a long right tail — distributional quirks that matter for calibration.",
  },
];

const SUGGESTED = [
  "Why is UAE's reproduction rate harder to predict?",
  "What is semi-supervised learning and why does it help here?",
  "How much labeled data do we need for domain adaptation?",
  "What causes clustered missingness in UAE data?",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function ask(q: string) {
    setQuestion(q);
    setAnswer("");
    setSources([]);
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      if (data.error) setError(data.error);
      else {
        setAnswer(data.answer);
        setSources(data.sources);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 font-sans">

      {/* Hero */}
      <section className="max-w-4xl mx-auto px-6 pt-20 pb-12 text-center">
        <span className="inline-block px-3 py-1 text-xs font-semibold tracking-widest uppercase bg-blue-900/50 text-blue-300 rounded-full mb-6">
          Domain Adaptation · Semi-Supervised Learning · RAG
        </span>
        <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 leading-tight">
          COVID-19 UAE<br />
          <span className="text-blue-400">Data Scarcity Pipeline</span>
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mx-auto">
          Exploratory analysis of UAE COVID data revealing structural missingness and domain shift —
          then a domain adaptation model that cuts prediction error by{" "}
          <span className="text-green-400 font-semibold">54%</span> using just 10% UAE labels,
          explained by a live RAG system.
        </p>
        <div className="flex flex-wrap justify-center gap-3 mt-8 text-sm">
          {["UAE · Germany · India · Singapore", "Mar 2020 – Mar 2023", "Gradient Boosting + Claude RAG"].map((t) => (
            <span key={t} className="px-3 py-1 bg-slate-800 rounded-full text-slate-300">{t}</span>
          ))}
        </div>
      </section>

      {/* EDA */}
      <section className="max-w-5xl mx-auto px-6 pb-16">
        <h2 className="text-2xl font-bold text-white mb-2">Exploratory Analysis</h2>
        <p className="text-slate-400 mb-8">What the data actually looks like before any model touches it.</p>
        <div className="grid gap-8">
          {EDA_PANELS.map((p) => (
            <div key={p.src} className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-800">
              <div className="bg-slate-800/60 px-5 py-3 flex items-start gap-3">
                <span className="text-blue-400 mt-0.5">→</span>
                <p className="text-slate-300 text-sm">{p.insight}</p>
              </div>
              <Image
                src={p.src}
                alt={p.title}
                width={900}
                height={400}
                className="w-full h-auto"
              />
            </div>
          ))}
        </div>
      </section>

      {/* Model Results */}
      <section className="max-w-5xl mx-auto px-6 pb-16">
        <h2 className="text-2xl font-bold text-white mb-2">Domain Adaptation Results</h2>
        <p className="text-slate-400 mb-8">
          Baseline trained on Germany, India, Singapore only — then adapted with 10% UAE labeled data.
        </p>
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {[
            { label: "Baseline (source-only)", mae: "0.49", r2: "-0.03", color: "red", note: "Worse than a mean predictor" },
            { label: "Adapted (+10% UAE labels)", mae: "0.23", r2: "0.76", color: "green", note: "54% MAE improvement" },
          ].map((m) => (
            <div
              key={m.label}
              className={`bg-slate-900 rounded-2xl p-6 border ${m.color === "green" ? "border-green-700/50" : "border-red-800/40"}`}
            >
              <p className="text-sm text-slate-400 mb-3">{m.label}</p>
              <div className="flex gap-6 mb-3">
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-wide">MAE</p>
                  <p className={`text-3xl font-bold ${m.color === "green" ? "text-green-400" : "text-red-400"}`}>{m.mae}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-wide">R²</p>
                  <p className={`text-3xl font-bold ${m.color === "green" ? "text-green-400" : "text-red-400"}`}>{m.r2}</p>
                </div>
              </div>
              <p className="text-sm text-slate-400">{m.note}</p>
            </div>
          ))}
        </div>
        <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-800">
          <Image
            src="/model_comparison.png"
            alt="Model comparison scatter plots"
            width={1100}
            height={450}
            className="w-full h-auto"
          />
        </div>
      </section>

      {/* RAG Chat */}
      <section className="max-w-3xl mx-auto px-6 pb-24">
        <h2 className="text-2xl font-bold text-white mb-2">Research Q&A</h2>
        <p className="text-slate-400 mb-6">
          Ask anything about the methods, findings, or data. Powered by Claude with retrieval over a curated COVID research knowledge base.
        </p>
        <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
          <div className="flex flex-wrap gap-2 mb-5">
            {SUGGESTED.map((q) => (
              <button
                key={q}
                onClick={() => ask(q)}
                className="text-xs px-3 py-1.5 bg-slate-800 hover:bg-blue-900/50 hover:text-blue-300 text-slate-300 rounded-full transition-colors border border-slate-700 hover:border-blue-700"
              >
                {q}
              </button>
            ))}
          </div>
          <div className="flex gap-3">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && question.trim() && ask(question)}
              placeholder="Ask a question about the research..."
              className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={() => question.trim() && ask(question)}
              disabled={loading || !question.trim()}
              className="px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 rounded-xl text-sm font-semibold transition-colors"
            >
              {loading ? "..." : "Ask"}
            </button>
          </div>
          {(loading || answer || error) && (
            <div className="mt-5 pt-5 border-t border-slate-800">
              {loading && (
                <div className="flex items-center gap-2 text-slate-400 text-sm">
                  <span className="animate-pulse">●</span> Retrieving context and generating answer...
                </div>
              )}
              {error && <p className="text-red-400 text-sm">{error}</p>}
              {answer && (
                <>
                  <p className="text-slate-100 text-sm leading-relaxed mb-4">{answer}</p>
                  {sources.length > 0 && (
                    <div>
                      <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">Sources retrieved</p>
                      <ul className="space-y-1">
                        {sources.map((s) => (
                          <li key={s} className="text-xs text-blue-400 flex items-start gap-1.5">
                            <span className="mt-0.5 shrink-0">↗</span>{s}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-8 text-center text-sm text-slate-500">
        Built by{" "}
        <a href="https://github.com/Denina-V" className="text-blue-400 hover:underline">
          Denina-V
        </a>{" "}
        · Data: OWID COVID-19 · Model: Gradient Boosting + Domain Adaptation · RAG: Claude Sonnet
      </footer>
    </main>
  );
}
