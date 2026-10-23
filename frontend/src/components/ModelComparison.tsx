import { BarChart3 } from "lucide-react";
import type { ModelMetric } from "../types/fraud";
import { money, percent } from "../utils/format";

interface Props {
  models: ModelMetric[];
  selectedModel: string;
}

export function ModelComparison({ models, selectedModel }: Props) {
  return (
    <section className="panel model-panel">
      <div className="panel-heading">
        <BarChart3 size={20} />
        <h2>Model Comparison</h2>
      </div>
      <div className="model-table" role="table">
        <div className="model-row model-header" role="row">
          <span>Model</span>
          <span>PR-AUC</span>
          <span>Recall</span>
          <span>FPR</span>
          <span>Loss Prevented</span>
        </div>
        {models.map((model) => (
          <div className={`model-row ${model.model === selectedModel ? "selected" : ""}`} role="row" key={model.model}>
            <span>{model.model}</span>
            <span>{model.pr_auc.toFixed(3)}</span>
            <span>{percent(model.recall)}</span>
            <span>{percent(model.false_positive_rate)}</span>
            <span>{money(model.losses_prevented)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

