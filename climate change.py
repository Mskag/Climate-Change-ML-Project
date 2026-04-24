import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.preprocessing   import PolynomialFeatures, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm             import SVR
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb


years_raw = np.array([
    1880,1885,1890,1895,1900,1905,1910,1915,1920,1925,
    1930,1935,1940,1945,1950,1955,1960,1965,1970,1975,
    1980,1985,1990,1995,2000,2005,2010,2015,2020,2023
])
temp_anomalies = np.array([
    -0.20,-0.22,-0.24,-0.21,-0.10,-0.18,-0.25,-0.18,-0.15,-0.10,
    -0.05,-0.02,-0.05, 0.10, 0.02,-0.01, 0.00, 0.00, 0.04, 0.02,
     0.26, 0.12, 0.44, 0.38, 0.42, 0.67, 0.72, 0.90, 1.02, 1.17
])

year_min  = years_raw.min()
X_raw     = years_raw.reshape(-1, 1)
X_norm    = (years_raw - year_min).reshape(-1, 1)   # 0-based
y         = temp_anomalies

future_years_raw  = np.arange(1880, 2101)
future_years_norm = (future_years_raw - year_min).reshape(-1, 1)

models = {
    "Linear Regression": Pipeline([
        ("reg", LinearRegression())
    ]),
    "Polynomial (deg 3)": Pipeline([
        ("poly",  PolynomialFeatures(degree=3, include_bias=False)),
        ("ridge", Ridge(alpha=0.1))
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=5, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    ),
    "SVR (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=10, epsilon=0.05))
    ]),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    ),
}

results    = {}
cv_scores  = {}

for name, mdl in models.items():
    mdl.fit(X_norm, y)
    train_pred = mdl.predict(X_norm)
    future_pred = mdl.predict(future_years_norm)

    rmse = np.sqrt(mean_squared_error(y, train_pred))
    mae  = mean_absolute_error(y, train_pred)
    r2   = r2_score(y, train_pred)
    cv   = cross_val_score(mdl, X_norm, y, cv=5, scoring="r2")

    results[name]   = {"pred": future_pred, "train_pred": train_pred,
                       "rmse": rmse, "mae": mae, "r2": r2}
    cv_scores[name] = cv
    print(f"{name:25s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}"
          f"  CV-R²={cv.mean():.4f}±{cv.std():.4f}")

weights = np.ones(len(models)) / len(models)

ensemble_pred = sum(
    w * results[n]["pred"]
    for w, n in zip(weights, models)
)
ensemble_train = sum(
    w * results[n]["train_pred"]
    for w, n in zip(weights, models)
)
results["Ensemble"] = {
    "pred": ensemble_pred,
    "train_pred": ensemble_train,
    "rmse": np.sqrt(mean_squared_error(y, ensemble_train)),
    "mae":  mean_absolute_error(y, ensemble_train),
    "r2":   r2_score(y, ensemble_train),
}
print(f"\n{'Ensemble':25s}  RMSE={results['Ensemble']['rmse']:.4f}"
      f"  MAE={results['Ensemble']['mae']:.4f}"
      f"  R²={results['Ensemble']['r2']:.4f}")

scenarios = {
    "RCP 2.6 (Aggressive mitigation)": 0.55,
    "RCP 4.5 (Moderate mitigation)":   0.85,
    "RCP 8.5 (Business-as-usual)":     1.30,
}
hist_mask   = future_years_raw <= 2023
future_mask = future_years_raw >  2023
baseline    = ensemble_pred.copy()

scenario_preds = {}
for label, factor in scenarios.items():
    s                    = baseline.copy()
    s[future_mask]       = baseline[hist_mask][-1] + \
                           (baseline[future_mask] - baseline[hist_mask][-1]) * factor
    scenario_preds[label] = s

COLORS = {
    "Linear Regression":     "#a0aec0",
    "Polynomial (deg 3)":    "#ed8936",
    "Random Forest":         "#48bb78",
    "Gradient Boosting":     "#9f7aea",
    "SVR (RBF)":             "#f687b3",
    "XGBoost":               "#63b3ed",
    "Ensemble":              "#e53e3e",
    "RCP 2.6 (Aggressive mitigation)": "#38a169",
    "RCP 4.5 (Moderate mitigation)":   "#d69e2e",
    "RCP 8.5 (Business-as-usual)":     "#c53030",
}

fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
fig.suptitle("Global Temperature Anomaly — ML Projection Suite",
             fontsize=18, color="white", fontweight="bold", y=0.98)

gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
ax1 = fig.add_subplot(gs[0, :])   # full-width top
ax2 = fig.add_subplot(gs[1, 0])   # bottom-left
ax3 = fig.add_subplot(gs[1, 1])   # bottom-right

style = dict(facecolor="#161b22", edgecolor="#30363d")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor(style["facecolor"])
    for spine in ax.spines.values():
        spine.set_edgecolor(style["edgecolor"])
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color="#21262d", linewidth=0.7, linestyle="--")

# ── Panel 1: All models + observed ──────────────────────────
for name, res in results.items():
    lw   = 2.5 if name == "Ensemble" else 1.2
    ls   = "-"  if name == "Ensemble" else "--"
    zord = 5    if name == "Ensemble" else 2
    ax1.plot(future_years_raw, res["pred"],
             color=COLORS[name], lw=lw, ls=ls, label=name, zorder=zord, alpha=0.85)

ax1.scatter(years_raw, y, color="white", s=40, zorder=10,
            edgecolors="#e53e3e", linewidths=1.2, label="Observed data")
ax1.axvline(2023, color="#4a5568", lw=1, linestyle=":")
ax1.text(2025, ax1.get_ylim()[0] if ax1.get_ylim()[0] > -1 else -0.5,
         "→ Projection", color="#4a5568", fontsize=9)
ax1.set_title("Model Comparison — Historical Fit & Future Projection", fontsize=13)
ax1.set_xlabel("Year");  ax1.set_ylabel("Temperature Anomaly (°C)")
ax1.legend(loc="upper left", fontsize=8, framealpha=0.2,
           labelcolor="white", facecolor="#161b22", edgecolor="#30363d")

ax2.fill_between(future_years_raw,
                 scenario_preds["RCP 2.6 (Aggressive mitigation)"],
                 scenario_preds["RCP 8.5 (Business-as-usual)"],
                 alpha=0.15, color="#e53e3e", label="Scenario range")

for label, spred in scenario_preds.items():
    ax2.plot(future_years_raw, spred, color=COLORS[label], lw=2, label=label)

ax2.scatter(years_raw, y, color="white", s=30, zorder=5, edgecolors="#aaa", lw=0.8)
ax2.axhline(1.5, color="#d69e2e", lw=1, linestyle=":", alpha=0.7)
ax2.axhline(2.0, color="#c53030", lw=1, linestyle=":", alpha=0.7)
ax2.text(1882, 1.52, "Paris 1.5°C target", color="#d69e2e", fontsize=8)
ax2.text(1882, 2.02, "2°C threshold",       color="#c53030", fontsize=8)
ax2.axvline(2023, color="#4a5568", lw=1, linestyle=":")
ax2.set_title("Scenario-Based Projections (RCP)", fontsize=12)
ax2.set_xlabel("Year");  ax2.set_ylabel("Temperature Anomaly (°C)")
ax2.legend(fontsize=8, framealpha=0.2, labelcolor="white",
           facecolor="#161b22", edgecolor="#30363d")

metric_names = [n for n in results if n != "Ensemble"] + ["Ensemble"]
r2_vals  = [results[n]["r2"]   for n in metric_names]
rmse_vals= [results[n]["rmse"] for n in metric_names]
x_pos    = np.arange(len(metric_names))
bar_cols = [COLORS[n] for n in metric_names]

bars = ax3.bar(x_pos - 0.2, r2_vals,  width=0.35, color=bar_cols, alpha=0.85, label="R²")
ax3.bar(x_pos + 0.2, rmse_vals, width=0.35, color=bar_cols, alpha=0.4,  label="RMSE")
ax3.set_xticks(x_pos)
ax3.set_xticklabels([n.split(" ")[0] for n in metric_names],
                    rotation=30, ha="right", fontsize=9, color="white")
ax3.set_title("Model Performance (R² vs RMSE)", fontsize=12)
ax3.set_ylabel("Score"); ax3.legend(fontsize=9, framealpha=0.2,
           labelcolor="white", facecolor="#161b22", edgecolor="#30363d")
for bar, val in zip(bars, r2_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", fontsize=7, color="white")

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "climate_ml_analysis.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"\nPlot saved → {save_path}")
plt.show()


print("\n" + "="*65)
print(f"{'Model':<28} {'R²':>6}  {'RMSE':>7}  {'MAE':>7}")
print("="*65)
for name, res in results.items():
    print(f"{name:<28} {res['r2']:>6.4f}  {res['rmse']:>7.4f}  {res['mae']:>7.4f}")
print("="*65)

print("\n── 2100 Ensemble Projections by Scenario ──")
for label, spred in scenario_preds.items():
    print(f"  {label:<42}  {spred[-1]:+.2f}°C")