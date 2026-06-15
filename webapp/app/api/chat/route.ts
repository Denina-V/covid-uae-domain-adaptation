import Anthropic from "@anthropic-ai/sdk";
import { NextRequest, NextResponse } from "next/server";

const DOCS = [
  {
    title: "Semi-Supervised Learning for Epidemiological Data",
    text: "Semi-supervised learning (SSL) leverages large amounts of unlabeled data alongside small labeled datasets. In COVID-19 epidemiology, labels such as reproduction rates are often missing or delayed, making SSL particularly valuable. SSL methods like pseudo-labeling and consistency regularization have shown 15–30% improvement over supervised baselines when labeled data is below 20% of the total.",
  },
  {
    title: "Domain Shift in Pandemic Modelling Across Countries",
    text: "Models trained on European COVID data systematically underperform on Middle Eastern and South Asian datasets due to differences in population density, reporting infrastructure, and intervention timing. This covariate shift means features like case counts have different statistical distributions across regions. Domain adaptation techniques such as feature alignment and instance re-weighting reduce this gap significantly.",
  },
  {
    title: "COVID-19 Data Quality in the UAE",
    text: "The UAE has one of the highest testing rates globally but inconsistent reporting of derived metrics like reproduction rate and ICU occupancy. Gaps in UAE's R-value data tend to cluster around public holidays and policy shifts, suggesting administrative rather than epidemiological causes. This clustered missingness violates the 'missing at random' assumption required by standard imputation methods.",
  },
  {
    title: "Retrieval-Augmented Generation for Clinical Decision Support",
    text: "RAG systems combine dense vector retrieval with large language models to answer questions grounded in a curated knowledge base. In clinical and epidemiological settings, RAG reduces hallucination by anchoring responses to verified documents. For COVID modelling, a RAG system can explain model predictions by retrieving relevant research on why specific features drive the reproduction number in a given region.",
  },
  {
    title: "Estimating the Effective Reproduction Number",
    text: "The effective reproduction number (Rt) measures how many secondary cases one infected person generates at time t. Rt < 1 indicates a shrinking epidemic; Rt > 1 indicates growth. Estimation requires smoothed case counts and assumptions about the serial interval distribution. In data-scarce settings, Rt estimates are unreliable without at least 7–14 days of consistent reporting.",
  },
  {
    title: "Transfer Learning for Low-Resource Epidemiological Settings",
    text: "Transfer learning pre-trains models on high-resource source domains and fine-tunes on low-resource targets. For COVID prediction in the UAE, models pre-trained on Germany or the UK and fine-tuned with as few as 50–100 UAE observations can match the performance of UAE-only models trained on 10x more data. The key is selecting source domains with similar epidemic dynamics.",
  },
  {
    title: "Handling Clustered Missingness in Time Series",
    text: "Clustered (non-random) missingness in time series data cannot be addressed by mean or forward-fill imputation without introducing bias. Methods like MICE (Multiple Imputation by Chained Equations) or deep learning imputation models trained on observed patterns are more appropriate. In epidemic data, missingness during outbreak peaks is particularly dangerous as it masks the most critical signal.",
  },
];

function retrieve(query: string, k = 3) {
  const qWords = new Set(query.toLowerCase().split(/\s+/));
  return DOCS.map((doc) => {
    const docWords = doc.text.toLowerCase().split(/\s+/);
    const overlap = docWords.filter((w) => qWords.has(w)).length;
    return { doc, score: overlap };
  })
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((x) => x.doc);
}

export async function POST(req: NextRequest) {
  const { question } = await req.json();
  if (!question?.trim()) {
    return NextResponse.json({ error: "No question provided" }, { status: 400 });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "ANTHROPIC_API_KEY not configured on server." },
      { status: 500 }
    );
  }

  const docs = retrieve(question, 3);
  const context = docs
    .map((d) => `[${d.title}]\n${d.text}`)
    .join("\n\n");

  const prompt = `You are a research assistant explaining COVID-19 modelling findings.
Answer the question using ONLY the research context below. Be concise (3–5 sentences max).

Research Context:
${context}

Model results from this project:
- Baseline (Germany/India/Singapore only): MAE 0.49, R² -0.03
- Adapted (+10% UAE labels): MAE 0.23, R² 0.76
- Domain adaptation improved MAE by 54%

Question: ${question}

Answer:`;

  const client = new Anthropic({ apiKey });
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 350,
    messages: [{ role: "user", content: prompt }],
  });

  const answer = (response.content[0] as { text: string }).text.trim();
  const sources = docs.map((d) => d.title);
  return NextResponse.json({ answer, sources });
}
