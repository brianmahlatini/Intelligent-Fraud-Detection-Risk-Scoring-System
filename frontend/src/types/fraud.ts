export type RiskBand = "low" | "medium" | "high" | "critical";

export interface Explanation {
  feature: string;
  label: string;
  value: number;
  impact: number;
  direction: string;
}

export interface ScoredTransaction {
  transaction_id: string;
  customer_id: number;
  merchant_id: number;
  amount: number;
  merchant_category: string;
  channel: string;
  fraud_probability: number;
  risk_score: number;
  risk_band: RiskBand;
  flagged: boolean;
  decision_threshold: number;
  estimated_loss_exposure: number;
  recommended_action: string;
  explanations: Explanation[];
  timestamp?: string;
  investigation_status?: string;
  is_fraud?: number;
}

export interface DashboardSummary {
  count: number;
  flagged: number;
  critical: number;
  estimated_loss_exposure: number;
  model_name: string;
  decision_threshold: number;
}

export interface TransactionsResponse {
  summary: DashboardSummary;
  transactions: ScoredTransaction[];
}

export interface ModelMetric {
  model: string;
  roc_auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  threshold: number;
  estimated_loss: number;
  losses_prevented: number;
}

export interface MetricsResponse {
  dataset: {
    rows: number;
    fraud_rate: number;
    target: string;
    positive_class: string;
    class_balance: Record<string, number>;
  };
  models_trained: ModelMetric[];
  selected_model: ModelMetric;
  feature_count: number;
  top_features: [string, number][];
  feature_groups: Record<string, string[]>;
}

export interface TransactionRequest {
  transaction_id: string;
  amount: number;
  merchant_category: string;
  channel: string;
  card_present: number;
  international: number;
  distance_from_home_km: number;
  merchant_risk_score: number;
  device_trust_score: number;
  failed_login_24h: number;
  customer_avg_amount: number;
  customer_std_amount: number;
  customer_txn_count_24h: number;
  spending_velocity_1h: number;
  spending_velocity_24h: number;
  is_new_merchant: number;
}

