const usd = (n) => new Intl.NumberFormat("nb-NO", { maximumFractionDigits: 0 }).format(n);

export const ResultPanel = ({ price, metrics }) => {
  const best = metrics?.validation?.["LightGBM (rå pris)"];
  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8 shadow-sm" data-testid="result-card">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Estimert pris</p>
        {price === null ? (
          <p className="mt-3 text-sm text-slate-500" data-testid="result-placeholder">
            Fyll ut skjemaet og trykk «Estimer pris».
          </p>
        ) : (
          <p className="mt-2 text-4xl sm:text-5xl font-black text-sky-600 tracking-tight font-heading"
            data-testid="predicted-price-output">${usd(price)}</p>
        )}
        <p className="mt-3 text-xs text-slate-500">Prisen er i amerikanske dollar (USD), som i datasettet.</p>
      </div>
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm text-sm text-slate-700 leading-relaxed"
        data-testid="model-info-section">
        <h2 className="font-heading font-bold text-base text-slate-900">Om modellen</h2>
        <p className="mt-2">
          LightGBM-modell trent på datasettet fra Kaggle-konkurransen Playground Series S4E9
          ({metrics ? usd(metrics.n_train_rows) : "–"} biler). Konkurransen bruker RMSE som metrikk.
        </p>
        {best && (
          <dl className="mt-4 grid grid-cols-3 gap-3 font-mono text-xs">
            <div><dt className="text-slate-500">RMSE</dt><dd className="font-semibold" data-testid="metric-rmse">${usd(best.rmse)}</dd></div>
            <div><dt className="text-slate-500">MAE</dt><dd className="font-semibold" data-testid="metric-mae">${usd(best.mae)}</dd></div>
            <div><dt className="text-slate-500">R²</dt><dd className="font-semibold" data-testid="metric-r2">{best.r2.toFixed(3)}</dd></div>
          </dl>
        )}
        <p className="mt-4 text-xs text-slate-500">
          Merk: prisene i datasettet varierer mye, så estimatet kan avvike betydelig for enkeltbiler.
        </p>
      </div>
    </div>
  );
};
