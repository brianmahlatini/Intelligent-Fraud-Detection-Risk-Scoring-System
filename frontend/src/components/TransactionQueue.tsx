import { AlertTriangle, CheckCircle2 } from "lucide-react";
import type { ScoredTransaction } from "../types/fraud";
import { money, percent } from "../utils/format";

interface Props {
  transactions: ScoredTransaction[];
  selectedId?: string;
  onSelect: (transaction: ScoredTransaction) => void;
}

export function TransactionQueue({ transactions, selectedId, onSelect }: Props) {
  return (
    <section className="panel queue-panel">
      <div className="panel-heading">
        <AlertTriangle size={20} />
        <h2>Investigation Queue</h2>
      </div>
      <div className="queue-list">
        {transactions.map((transaction) => (
          <button
            className={`queue-item ${transaction.transaction_id === selectedId ? "active" : ""}`}
            type="button"
            onClick={() => onSelect(transaction)}
            key={transaction.transaction_id}
          >
            <span className={`risk-dot ${transaction.risk_band}`} />
            <span className="queue-main">
              <strong>{transaction.transaction_id}</strong>
              <small>
                {transaction.merchant_category} · {transaction.channel} · {money(transaction.amount)}
              </small>
            </span>
            <span className="queue-score">
              {transaction.flagged ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
              {percent(transaction.fraud_probability)}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

