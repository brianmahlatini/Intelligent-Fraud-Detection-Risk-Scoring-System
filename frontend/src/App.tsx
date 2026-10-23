import { useEffect, useMemo, useState } from "react";
import { Activity, BadgeDollarSign, ShieldAlert, Siren } from "lucide-react";
import { fetchMetrics, fetchTransactions } from "./api/client";
import { ExplanationPanel } from "./components/ExplanationPanel";
import { LiveScoringForm } from "./components/LiveScoringForm";
import { MetricCard } from "./components/MetricCard";
import { ModelComparison } from "./components/ModelComparison";
import { TransactionQueue } from "./components/TransactionQueue";
import type { MetricsResponse, ScoredTransaction, TransactionsResponse } from "./types/fraud";
import { compact, money, percent } from "./utils/format";
import "./styles.css";

export default function App() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [queue, setQueue] = useState<TransactionsResponse | null>(null);
  const [selected, setSelected] = useState<ScoredTransaction | undefined>();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchMetrics(), fetchTransactions(140)])
      .then(([metricsPayload, transactionPayload]) => {
        setMetrics(metricsPayload);
        setQueue(transactionPayload);
        setSelected(transactionPayload.transactions.find((item) => item.flagged) ?? transactionPayload.transactions[0]);
      })
      .catch((apiError: Error) => setError(apiError.message));
  }, []);

  const highRisk = useMemo(
    () => queue?.transactions.filter((item) => item.risk_band === "high" || item.risk_band === "critical") ?? [],
    [queue]
  );

  if (error) {
    return <main className="app-shell error-screen">API error: {error}</main>;
  }

  if (!metrics || !queue) {
    return <main className="app-shell loading-screen">Loading fraud monitoring workspace...</main>;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">Fraud Operations</span>
          <h1>Intelligent Fraud Detection & Risk Scoring System</h1>
        </div>
        <div className="model-pill">
          <Activity size={18} />
          {queue.summary.model_name}
        </div>
      </header>

      <section className="metrics-grid">
        <MetricCard
          icon={<ShieldAlert size={22} />}
          label="Flagged Cases"
          value={String(queue.summary.flagged)}
          detail={`${queue.summary.critical} critical alerts`}
        />
        <MetricCard
          icon={<BadgeDollarSign size={22} />}
          label="Loss Exposure"
          value={money(queue.summary.estimated_loss_exposure)}
          detail="Probability-weighted exposure"
        />
        <MetricCard
          icon={<Siren size={22} />}
          label="Fraud Rate"
          value={percent(metrics.dataset.fraud_rate)}
          detail={`${compact(metrics.dataset.rows)} generated transactions`}
        />
        <MetricCard
          icon={<Activity size={22} />}
          label="Decision Threshold"
          value={percent(queue.summary.decision_threshold)}
          detail={`${metrics.feature_count} model features`}
        />
      </section>

      <section className="workspace-grid">
        <TransactionQueue transactions={highRisk} selectedId={selected?.transaction_id} onSelect={setSelected} />
        <ExplanationPanel transaction={selected} />
      </section>

      <section className="lower-grid">
        <ModelComparison models={metrics.models_trained} selectedModel={metrics.selected_model.model} />
        <LiveScoringForm />
      </section>
    </main>
  );
}

