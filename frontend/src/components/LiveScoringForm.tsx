import { useState } from "react";
import { ScanSearch } from "lucide-react";
import { scoreTransaction } from "../api/client";
import type { ScoredTransaction, TransactionRequest } from "../types/fraud";
import { percent } from "../utils/format";

const initialPayload: TransactionRequest = {
  transaction_id: "TX-LIVE-9001",
  amount: 875,
  merchant_category: "digital_goods",
  channel: "online",
  card_present: 0,
  international: 1,
  distance_from_home_km: 260,
  merchant_risk_score: 0.82,
  device_trust_score: 0.22,
  failed_login_24h: 3,
  customer_avg_amount: 80,
  customer_std_amount: 35,
  customer_txn_count_24h: 5,
  spending_velocity_1h: 910,
  spending_velocity_24h: 1320,
  is_new_merchant: 1
};

export function LiveScoringForm() {
  const [payload, setPayload] = useState<TransactionRequest>(initialPayload);
  const [result, setResult] = useState<ScoredTransaction | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    try {
      setResult(await scoreTransaction(payload));
    } finally {
      setLoading(false);
    }
  }

  function updateNumber(key: keyof TransactionRequest, value: string) {
    setPayload((current) => ({ ...current, [key]: Number(value) }));
  }

  return (
    <section className="panel scoring-panel">
      <div className="panel-heading">
        <ScanSearch size={20} />
        <h2>Runtime Scoring</h2>
      </div>
      <div className="form-grid">
        <label>
          Amount
          <input value={payload.amount} type="number" onChange={(event) => updateNumber("amount", event.target.value)} />
        </label>
        <label>
          Category
          <select
            value={payload.merchant_category}
            onChange={(event) => setPayload((current) => ({ ...current, merchant_category: event.target.value }))}
          >
            {["grocery", "fuel", "electronics", "travel", "gaming", "jewelry", "digital_goods", "restaurants", "cash_advance", "pharmacy"].map(
              (category) => (
                <option key={category}>{category}</option>
              )
            )}
          </select>
        </label>
        <label>
          Channel
          <select value={payload.channel} onChange={(event) => setPayload((current) => ({ ...current, channel: event.target.value }))}>
            {["pos", "online", "mobile", "atm"].map((channel) => (
              <option key={channel}>{channel}</option>
            ))}
          </select>
        </label>
        <label>
          Merchant Risk
          <input
            value={payload.merchant_risk_score}
            min="0"
            max="1"
            step="0.01"
            type="number"
            onChange={(event) => updateNumber("merchant_risk_score", event.target.value)}
          />
        </label>
        <label>
          Device Trust
          <input
            value={payload.device_trust_score}
            min="0"
            max="1"
            step="0.01"
            type="number"
            onChange={(event) => updateNumber("device_trust_score", event.target.value)}
          />
        </label>
        <label>
          Failed Logins
          <input value={payload.failed_login_24h} type="number" onChange={(event) => updateNumber("failed_login_24h", event.target.value)} />
        </label>
      </div>
      <div className="toggle-row">
        <label>
          <input
            checked={payload.international === 1}
            type="checkbox"
            onChange={(event) => setPayload((current) => ({ ...current, international: event.target.checked ? 1 : 0 }))}
          />
          International
        </label>
        <label>
          <input
            checked={payload.is_new_merchant === 1}
            type="checkbox"
            onChange={(event) => setPayload((current) => ({ ...current, is_new_merchant: event.target.checked ? 1 : 0 }))}
          />
          New merchant
        </label>
      </div>
      <button className="primary-button" type="button" onClick={submit} disabled={loading}>
        <ScanSearch size={18} />
        {loading ? "Scoring" : "Score Transaction"}
      </button>
      {result && (
        <div className={`score-result ${result.risk_band}`}>
          <span>{result.risk_band}</span>
          <strong>{result.risk_score}</strong>
          <p>{percent(result.fraud_probability)} fraud probability · {result.recommended_action}</p>
        </div>
      )}
    </section>
  );
}

