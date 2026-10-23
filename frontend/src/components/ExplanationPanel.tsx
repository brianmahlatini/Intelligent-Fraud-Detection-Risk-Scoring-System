import { ClipboardCheck, ShieldAlert } from "lucide-react";
import type { ScoredTransaction } from "../types/fraud";
import { money, percent } from "../utils/format";

interface Props {
  transaction?: ScoredTransaction;
}

export function ExplanationPanel({ transaction }: Props) {
  if (!transaction) {
    return (
      <section className="panel empty-state">
        <ShieldAlert size={28} />
        <p>Select a transaction to inspect risk drivers.</p>
      </section>
    );
  }

  return (
    <section className="panel explanation-panel">
      <div className="panel-heading">
        <ClipboardCheck size={20} />
        <h2>Case Explanation</h2>
      </div>
      <div className="case-summary">
        <div>
          <span>Risk Score</span>
          <strong>{transaction.risk_score}</strong>
        </div>
        <div>
          <span>Fraud Probability</span>
          <strong>{percent(transaction.fraud_probability)}</strong>
        </div>
        <div>
          <span>Loss Exposure</span>
          <strong>{money(transaction.estimated_loss_exposure)}</strong>
        </div>
      </div>
      <p className="action">{transaction.recommended_action}</p>
      <div className="explanations">
        {transaction.explanations.map((item) => (
          <div className="explanation" key={item.feature}>
            <span>{item.label}</span>
            <strong>{item.direction}</strong>
            <div className="impact-track">
              <i style={{ width: `${Math.min(Math.abs(item.impact) * 260, 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

