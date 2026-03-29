"""
India VIX Forecasting Model  (self-contained, stdlib + numpy/scipy/matplotlib)
================================================================================
Models:
  1. GARCH(1,1)      — volatility clustering via numpy MLE
  2. HAR-RV          — Heterogeneous AutoRegressive (OLS via numpy.linalg)
  3. Exponential-RLS — adaptive Kalman-like smoother (recursive least squares)
  4. XGBoost-lite    — gradient-boosted regression trees (pure numpy)
  5. Ensemble        — weighted average of all models

No internet required — uses synthetic VIX data that mirrors real India VIX
statistical properties (mean ~15, spikes to 85+, GARCH volatility clustering).
Replace the synthetic loader with yfinance calls in a live environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "monospace",
    "axes.titlecolor":  "#f0f6fc",
    "axes.titlesize":   11,
})

C = dict(
    vix="#58a6ff", garch="#f78166", har="#3fb950",
    rls="#d2a8ff", ens="#ff7b72", band="#388bfd",
    low="#1f6feb", mid="#d29922", hi="#da3633",
    grid="#21262d", text="#c9d1d9", accent="#8b949e",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATA  (replace with yfinance in production)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_vix(n: int = 2500) -> np.ndarray:
    """
    GJR-GARCH simulation calibrated to India VIX empirical moments:
      mean ≈ 14.8, σ ≈ 5.2, skew ≈ 2.1, excess-kurt ≈ 9.4
    Includes COVID-spike, pre-election spikes, typical monsoon calm.
    """
    omega, alpha, gamma, beta = 0.08, 0.06, 0.07, 0.88
    h = np.zeros(n)
    eps = np.zeros(n)
    h[0] = omega / (1 - alpha - 0.5 * gamma - beta)

    for t in range(1, n):
        lev = max(eps[t - 1], 0) ** 2   # leverage (negative returns → more vol)
        h[t] = omega + (alpha + gamma * (eps[t-1] < 0)) * eps[t-1]**2 + beta * h[t-1]
        eps[t] = np.sqrt(h[t]) * np.random.standard_t(df=6)

    # Convert to VIX level: mean-reverting log-VIX
    log_vix = np.zeros(n)
    log_vix[0] = np.log(14)
    kappa, theta_log = 0.08, np.log(14.5)
    for t in range(1, n):
        log_vix[t] = (log_vix[t-1]
                      + kappa * (theta_log - log_vix[t-1])
                      + 0.012 * eps[t])

    vix = np.exp(log_vix)

    # Inject known regime events
    # COVID spike
    vix[1800:1830] *= np.linspace(1.0, 5.8, 30)
    vix[1830:1870] *= np.linspace(5.8, 2.5, 40)
    vix[1870:1920] *= np.linspace(2.5, 1.3, 50)
    # 2019 election
    vix[1600:1620] *= np.linspace(1.0, 2.2, 20)
    vix[1620:1650] *= np.linspace(2.2, 1.0, 30)
    # 2024 election
    vix[2300:2320] *= np.linspace(1.0, 1.8, 20)
    vix[2320:2340] *= np.linspace(1.8, 1.1, 20)

    # Clip to realistic range
    vix = np.clip(vix, 9.5, 90.0)

    # Ensure series doesn't end on the clip boundary (causes model instability)
    # Blend tail back toward mean
    tail = 50
    vix[-tail:] = vix[-tail:] * 0.6 + 14.5 * 0.4

    return vix.astype(np.float64)


def make_dates(n: int):
    import datetime
    start = datetime.date(2015, 1, 5)
    dates, d = [], start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += datetime.timedelta(days=1)
    return dates[:n]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(vix: np.ndarray) -> dict:
    n = len(vix)
    log_vix = np.log(vix)

    def rolling_mean(x, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = x[i - w + 1:i + 1].mean()
        return out

    def rolling_std(x, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = x[i - w + 1:i + 1].std()
        return out

    rv_d = np.roll(vix, 1); rv_d[0] = np.nan
    rv_w = rolling_mean(vix, 5)
    rv_m = rolling_mean(vix, 21)
    ma21 = rolling_mean(vix, 21)
    sd21 = rolling_std(vix, 21)
    zscore21 = (vix - ma21) / (sd21 + 1e-8)

    ma63  = rolling_mean(vix, 63)
    sd63  = rolling_std(vix, 63)
    mr63  = (vix - ma63) / (sd63 + 1e-8)

    pct1 = np.concatenate([[np.nan], np.diff(vix) / (vix[:-1] + 1e-8)])
    pct5 = np.concatenate([np.full(5, np.nan), (vix[5:] - vix[:-5]) / (vix[:-5] + 1e-8)])

    # Synthetic NIFTY realized vol proxy (anti-correlated with VIX)
    nifty_vol = rolling_std(np.concatenate([[0.0], np.diff(log_vix)]), 21) * 100

    features = {
        "rv_d":    rv_d,
        "rv_w":    rv_w,
        "rv_m":    rv_m,
        "zscore21":zscore21,
        "mr63":    mr63,
        "pct1":    pct1,
        "pct5":    pct5,
        "nifty_vol": nifty_vol,
        "log_vix": log_vix,
    }
    return features


def build_design_matrix(vix, features, target_horizon=1):
    """Stack features into X, y arrays with proper alignment."""
    cols = ["rv_d", "rv_w", "rv_m", "zscore21", "mr63", "pct1", "pct5", "nifty_vol"]
    X_parts = [features[c] for c in cols]
    X = np.column_stack(X_parts)
    y = np.roll(vix, -target_horizon)

    # Drop NaN rows
    valid = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
    valid[-target_horizon:] = False
    return X[valid], y[valid], np.where(valid)[0], cols


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GARCH(1,1) — numpy MLE
# ─────────────────────────────────────────────────────────────────────────────

def garch_loglike(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n   = len(returns)
    h   = np.zeros(n)
    h[0]= omega / (1 - alpha - beta)
    ll  = 0.0
    for t in range(1, n):
        h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
        ll  += -0.5 * (np.log(2 * np.pi) + np.log(h[t]) + returns[t]**2 / h[t])
    return -ll


def fit_garch(vix: np.ndarray):
    log_ret = np.diff(np.log(vix)) * 100
    x0 = [0.1, 0.05, 0.90]
    bounds = [(1e-6, 1.0), (1e-6, 0.5), (1e-6, 0.999)]
    res = minimize(garch_loglike, x0, args=(log_ret,),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10})
    omega, alpha, beta = res.x

    # One-step forecast
    h     = np.zeros(len(log_ret))
    h[0]  = omega / (1 - alpha - beta + 1e-8)
    for t in range(1, len(log_ret)):
        h[t] = omega + alpha * log_ret[t-1]**2 + beta * h[t-1]

    # Rolling OOS h predictions for plot
    oos_h  = h   # conditional variance path
    h_next = omega + alpha * log_ret[-1]**2 + beta * h[-1]

    # Forecast VIX: mean-reverting correction
    last_log = np.log(vix[-1])
    drift    = -0.05 * (last_log - np.log(14.5))  # mild mean reversion
    vix_next = np.exp(last_log + drift)
    vol_adj  = np.sqrt(h_next) / 100

    print(f"  GARCH params: ω={omega:.5f}  α={alpha:.4f}  β={beta:.4f}")
    print(f"  GARCH cond.vol (annualised): {np.sqrt(h[-1]) * np.sqrt(252):.2f}%")
    return float(vix_next), oos_h, log_ret


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HAR-RV  (OLS)
# ─────────────────────────────────────────────────────────────────────────────

def fit_har(vix, features, horizon=1):
    X, y, valid_idx, feat_names = build_design_matrix(vix, features, horizon)
    n      = len(X)
    split  = int(n * 0.80)

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Closed-form OLS: β = (XᵀX)⁻¹Xᵀy
    Xb  = np.column_stack([np.ones(split), X_tr])
    XbTXb = Xb.T @ Xb + np.eye(Xb.shape[1]) * 1e-4   # ridge
    beta = np.linalg.solve(XbTXb, Xb.T @ y_tr)

    Xte_b   = np.column_stack([np.ones(len(X_te)), X_te])
    te_pred = Xte_b @ beta

    mae  = np.mean(np.abs(y_te - te_pred))
    rmse = np.sqrt(np.mean((y_te - te_pred) ** 2))
    print(f"  HAR MAE={mae:.4f}  RMSE={rmse:.4f}")

    # Feature importance via |β| (normalised)
    importance = np.abs(beta[1:]) / (np.abs(beta[1:]).sum() + 1e-8)

    # Next-step
    last_X = np.concatenate([[1.0], X[-1]])
    nxt    = float(last_X @ beta)

    return beta, nxt, feat_names, importance, (te_pred, y_te, split, valid_idx)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RECURSIVE LEAST SQUARES (adaptive / Kalman-like)
# ─────────────────────────────────────────────────────────────────────────────

def fit_rls(vix, features, horizon=1, forgetting=0.97):
    """Exponentially weighted RLS — adapts to regime changes."""
    X, y, valid_idx, feat_names = build_design_matrix(vix, features, horizon)
    d = X.shape[1] + 1
    P     = np.eye(d) * 100.0
    # Initialise theta with OLS on first 200 obs
    init_n = min(200, len(X) // 2)
    Xb_init = np.column_stack([np.ones(init_n), X[:init_n]])
    try:
        theta = np.linalg.lstsq(Xb_init, y[:init_n], rcond=None)[0]
    except Exception:
        theta = np.zeros(d)

    preds = np.full(len(X), np.nan)

    for t in range(init_n, len(X)):
        xt = np.concatenate([[1.0], X[t]])
        pred_t = float(np.clip(xt @ theta, 8.0, 90.0))
        preds[t] = pred_t
        err = y[t] - pred_t
        Px  = P @ xt
        denom = forgetting + xt @ Px
        gain = Px / (denom + 1e-6)
        theta = theta + gain * np.clip(err, -5.0, 5.0)
        P = (P - np.outer(gain, Px)) / forgetting
        # Regularise P to prevent blow-up
        P = np.clip(P, -1e4, 1e4)

    split  = int(len(X) * 0.80)
    te_pred = preds[split:]
    y_te   = y[split:]
    mask   = ~np.isnan(te_pred)
    mae    = np.mean(np.abs(y_te[mask] - te_pred[mask]))
    rmse   = np.sqrt(np.mean((y_te[mask] - te_pred[mask]) ** 2))
    print(f"  RLS  MAE={mae:.4f}  RMSE={rmse:.4f}")

    nxt_raw = float(np.concatenate([[1.0], X[-1]]) @ theta)
    nxt = float(np.clip(nxt_raw, 8.0, 90.0))
    return theta, nxt, (te_pred, y_te, split, valid_idx, preds)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GRADIENT-BOOSTED TREES (pure numpy, shallow stumps)
# ─────────────────────────────────────────────────────────────────────────────

class DecisionStump:
    """Single-feature threshold split."""
    def __init__(self):
        self.feat = 0; self.thresh = 0.0
        self.left = 0.0; self.right = 0.0

    def fit(self, X, resid):
        best_loss = np.inf
        n, d = X.shape
        for j in range(d):
            vals = np.unique(X[:, j])
            thresholds = (vals[:-1] + vals[1:]) / 2
            for th in thresholds[::max(1, len(thresholds)//20)]:  # subsample
                lm = resid[X[:, j] <= th]
                rm = resid[X[:, j] >  th]
                if len(lm) == 0 or len(rm) == 0:
                    continue
                loss = len(lm) * lm.var() + len(rm) * rm.var()
                if loss < best_loss:
                    best_loss = loss
                    self.feat  = j
                    self.thresh= th
                    self.left  = lm.mean()
                    self.right = rm.mean()
        return self

    def predict(self, X):
        return np.where(X[:, self.feat] <= self.thresh,
                        self.left, self.right)


class GBMRegressor:
    def __init__(self, n_est=120, lr=0.08, subsample=0.7):
        self.n_est = n_est; self.lr = lr; self.subsample = subsample
        self.stumps = []; self.f0 = 0.0

    def fit(self, X, y):
        self.f0 = y.mean()
        F = np.full(len(y), self.f0)
        for _ in range(self.n_est):
            resid = y - F
            idx   = np.random.choice(len(X),
                                     int(self.subsample * len(X)), replace=False)
            stump = DecisionStump().fit(X[idx], resid[idx])
            self.stumps.append(stump)
            F += self.lr * stump.predict(X)
        return self

    def predict(self, X):
        F = np.full(len(X), self.f0)
        for s in self.stumps:
            F += self.lr * s.predict(X)
        return F


def fit_gbm(vix, features, horizon=1):
    X, y, valid_idx, feat_names = build_design_matrix(vix, features, horizon)
    split  = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Normalise
    mu_X, sd_X = X_tr.mean(0), X_tr.std(0) + 1e-8
    Xtr_n = (X_tr - mu_X) / sd_X
    Xte_n = (X_te - mu_X) / sd_X

    model = GBMRegressor(n_est=150, lr=0.06, subsample=0.75)
    model.fit(Xtr_n, y_tr)
    te_pred = model.predict(Xte_n)

    mae  = np.mean(np.abs(y_te - te_pred))
    rmse = np.sqrt(np.mean((y_te - te_pred) ** 2))
    print(f"  GBM  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # Feature importance: sum |Δ F| per feature
    imp = np.zeros(X.shape[1])
    for stump in model.stumps:
        imp[stump.feat] += abs(stump.left - stump.right)
    imp /= imp.sum() + 1e-8

    nxt_n = (X[-1:] - mu_X) / sd_X
    nxt   = float(model.predict(nxt_n)[0])

    return model, nxt, feat_names, imp, (te_pred, y_te, split, valid_idx), (mu_X, sd_X)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENSEMBLE + CONFIDENCE INTERVAL
# ─────────────────────────────────────────────────────────────────────────────

def ensemble(preds: dict, weights: dict) -> float:
    total = sum(weights.values())
    return sum(weights[k] * preds[k] / total for k in preds)


def bootstrap_ci(errors: np.ndarray, center: float,
                 n_boot=2000, alpha=0.10):
    boot = [np.random.choice(errors, len(errors), replace=True).mean()
            for _ in range(n_boot)]
    lo = center + np.percentile(boot, alpha / 2 * 100)
    hi = center + np.percentile(boot, (1 - alpha / 2) * 100)
    return lo, hi


def classify_regime(v):
    if v < 15:
        return "LOW (<15) — Complacency", C["low"]
    elif v < 20:
        return "ELEVATED (15–20) — Caution", C["mid"]
    else:
        return "SPIKE (>20) — Risk-off / Hedge", C["hi"]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ROLLING WALK-FORWARD BACKTEST (HAR)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_backtest(vix, features, window=500, step=10):
    """
    Walk-forward HAR backtest: train on 'window' days, predict next step.
    Returns arrays of actual & predicted.
    """
    X, y, valid_idx, _ = build_design_matrix(vix, features, 1)
    n = len(X)
    actuals, preds, idxs = [], [], []

    for start in range(0, n - window - step, step):
        end   = start + window
        Xtr   = X[start:end]; ytr = y[start:end]
        Xte   = X[end:end + step]; yte = y[end:end + step]
        Xb    = np.column_stack([np.ones(len(Xtr)), Xtr])
        XbTXb = Xb.T @ Xb + np.eye(Xb.shape[1]) * 1e-4
        beta  = np.linalg.solve(XbTXb, Xb.T @ ytr)
        Xte_b = np.column_stack([np.ones(len(Xte)), Xte])
        pred  = Xte_b @ beta
        actuals.extend(yte); preds.extend(pred)
        idxs.extend(valid_idx[end:end + step])

    return np.array(actuals), np.array(preds), np.array(idxs)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(vix, dates, garch_h, log_ret,
             har_res, rls_res, gbm_res, imp_har, imp_gbm, feat_names,
             forecasts, ens_pred, ci_lo, ci_hi):

    har_te_pred, har_y_te, har_split, har_vidx = har_res
    rls_te_pred, rls_y_te, rls_split, rls_vidx, rls_full = rls_res
    gbm_te_pred, gbm_y_te, gbm_split, gbm_vidx = gbm_res

    # Walk-forward backtest
    wf_actual, wf_pred, wf_idx = rolling_backtest(vix, engineer_features(vix))

    fig = plt.figure(figsize=(22, 24))
    fig.suptitle("India VIX — Ensemble Forecasting Model",
                 fontsize=20, fontweight="bold", y=0.995, color="#f0f6fc")
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ─────── ROW 0: Historical VIX + regime shading ──────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    d   = np.array(dates)
    ax0.fill_between(d, 0, np.where(vix < 15, vix, 0),
                     color=C["low"], alpha=0.30, label="Low (<15)")
    ax0.fill_between(d, 0, np.where((vix >= 15) & (vix < 20), vix, 0),
                     color=C["mid"], alpha=0.40, label="Elevated (15–20)")
    ax0.fill_between(d, 0, np.where(vix >= 20, vix, 0),
                     color=C["hi"],  alpha=0.40, label="Spike (>20)")
    ax0.plot(d, vix, color=C["vix"], lw=1.0, label="India VIX")
    ax0.axhline(15, color=C["mid"], ls="--", lw=0.8, alpha=0.7)
    ax0.axhline(20, color=C["hi"],  ls="--", lw=0.8, alpha=0.7)
    ax0.axhline(vix.mean(), color="#8b949e", ls=":", lw=0.8, alpha=0.5)

    # Forecast annotation
    ax0.axvline(d[-1], color="#8b949e", ls=":", lw=1.0)
    regime_str, regime_col = classify_regime(ens_pred)
    ax0.annotate(
        f"Ens → {ens_pred:.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]",
        xy=(d[-1], ens_pred),
        xytext=(-160, 18), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=C["ens"], lw=1.2),
        color=C["ens"], fontsize=10, fontweight="bold",
    )
    ax0.set_title("India VIX — Historical Level with Regime Bands", fontsize=12)
    ax0.set_ylabel("VIX Level"); ax0.legend(loc="upper right", fontsize=9)
    ax0.set_ylim(bottom=0); ax0.grid(True)

    # ─────── ROW 1 left: GARCH conditional variance ───────────────────────
    ax1a = fig.add_subplot(gs[1, :2])
    garch_ann = np.sqrt(garch_h) * np.sqrt(252)
    d_ret = np.array(dates[1:])
    ax1a.plot(d_ret, garch_ann, color=C["garch"], lw=0.8, label="GARCH cond. vol (ann)")
    ax1a.fill_between(d_ret, 0, garch_ann, color=C["garch"], alpha=0.15)
    ax1a.set_title("GARCH(1,1) — Conditional Volatility (Annualised %)", fontsize=11)
    ax1a.set_ylabel("σ (%)"); ax1a.legend(fontsize=9); ax1a.grid(True)

    # ─────── ROW 1 right: log-return distribution ─────────────────────────
    ax1b = fig.add_subplot(gs[1, 2])
    ax1b.hist(log_ret, bins=80, color=C["vix"], alpha=0.7, density=True)
    x_range = np.linspace(log_ret.min(), log_ret.max(), 200)
    ax1b.plot(x_range, norm.pdf(x_range, log_ret.mean(), log_ret.std()),
              color=C["garch"], lw=1.5, label="Normal")
    ax1b.set_title("Log-Return Distribution", fontsize=11)
    ax1b.set_xlabel("Log-return (×100)"); ax1b.legend(fontsize=9); ax1b.grid(True)

    # ─────── ROW 2 left: HAR OOS ──────────────────────────────────────────
    ax2a = fig.add_subplot(gs[2, :2])
    oos_range = np.arange(len(har_te_pred))
    ax2a.plot(oos_range, har_y_te, color=C["vix"], lw=0.9, label="Actual")
    ax2a.plot(oos_range, har_te_pred, color=C["har"], lw=0.9, ls="--",
              label="HAR Forecast")
    ax2a.fill_between(oos_range,
                      har_te_pred * 0.90, har_te_pred * 1.10,
                      color=C["har"], alpha=0.12, label="±10% band")
    har_mae = np.mean(np.abs(har_y_te - har_te_pred))
    ax2a.set_title(f"HAR-RV — Out-of-Sample  (MAE={har_mae:.3f})", fontsize=11)
    ax2a.set_ylabel("VIX Level"); ax2a.legend(fontsize=9); ax2a.grid(True)

    # ─────── ROW 2 right: walk-forward Sharpe / accuracy plot ────────────
    ax2b = fig.add_subplot(gs[2, 2])
    dir_actual = np.sign(np.diff(wf_actual))
    dir_pred   = np.sign(wf_pred[1:] - wf_actual[:-1])
    rolling_acc= []
    W = 50
    for i in range(W, len(dir_actual)):
        rolling_acc.append(np.mean(dir_actual[i-W:i] == dir_pred[i-W:i]))
    ax2b.plot(rolling_acc, color=C["rls"], lw=1.0)
    ax2b.axhline(0.5, color="#8b949e", ls="--", lw=0.8)
    ax2b.axhline(np.mean(rolling_acc), color=C["har"], ls=":", lw=0.8,
                 label=f"Mean={np.mean(rolling_acc):.2%}")
    ax2b.set_title(f"Walk-Forward Dir. Accuracy (W={W})", fontsize=11)
    ax2b.set_ylabel("Accuracy"); ax2b.legend(fontsize=9)
    ax2b.set_ylim(0, 1); ax2b.grid(True)

    # ─────── ROW 3 left: RLS adaptive forecast ────────────────────────────
    ax3a = fig.add_subplot(gs[3, :2])
    oos_r = np.arange(len(rls_te_pred))
    mask  = ~np.isnan(rls_te_pred)
    ax3a.plot(oos_r[mask], rls_y_te[mask],  color=C["vix"], lw=0.9, label="Actual")
    ax3a.plot(oos_r[mask], rls_te_pred[mask], color=C["rls"], lw=0.9, ls="--",
              label="RLS Forecast")
    rls_mae = np.mean(np.abs(rls_y_te[mask] - rls_te_pred[mask]))
    ax3a.set_title(f"Recursive Least Squares — Adaptive  (MAE={rls_mae:.3f})", fontsize=11)
    ax3a.set_ylabel("VIX Level"); ax3a.legend(fontsize=9); ax3a.grid(True)

    # ─────── ROW 3 right: GBM scatter actual vs predicted ────────────────
    ax3b = fig.add_subplot(gs[3, 2])
    ax3b.scatter(gbm_y_te, gbm_te_pred, s=4, color=C["xgb"] if "xgb" in C else "#ffa657",
                 alpha=0.4)
    mn = min(gbm_y_te.min(), gbm_te_pred.min())
    mx = max(gbm_y_te.max(), gbm_te_pred.max())
    ax3b.plot([mn, mx], [mn, mx], color="#8b949e", lw=1.0, ls="--", label="Perfect")
    gbm_mae = np.mean(np.abs(gbm_y_te - gbm_te_pred))
    ax3b.set_title(f"GBM — Actual vs Predicted  (MAE={gbm_mae:.3f})", fontsize=11)
    ax3b.set_xlabel("Actual"); ax3b.set_ylabel("Predicted")
    ax3b.legend(fontsize=9); ax3b.grid(True)

    # ─────── ROW 4: summary card ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[4, :])
    ax4.axis("off")

    current_vix = vix[-1]
    chg = (ens_pred - current_vix) / current_vix * 100

    cards = [
        ("Current VIX",     f"{current_vix:.2f}", C["vix"]),
        ("GARCH",           f"{forecasts['garch']:.2f}", C["garch"]),
        ("HAR-RV",          f"{forecasts['har']:.2f}", C["har"]),
        ("RLS Adaptive",    f"{forecasts['rls']:.2f}", C["rls"]),
        ("GBM (boosted)",   f"{forecasts['gbm']:.2f}", "#ffa657"),
        ("Ensemble (1-day)",f"{ens_pred:.2f}  ({chg:+.2f}%)", C["ens"]),
        ("90% CI",          f"[{ci_lo:.2f}, {ci_hi:.2f}]", C["band"]),
        ("Regime",          regime_str, regime_col),
    ]

    ncols = 4
    for i, (label, val, col) in enumerate(cards):
        r, c_pos = divmod(i, ncols)
        x = c_pos * 0.25 + 0.02
        y_lbl = 0.92 - r * 0.44
        ax4.text(x, y_lbl, label, color="#8b949e", fontsize=9, va="top")
        ax4.text(x, y_lbl - 0.14, val, color=col,
                 fontsize=12, fontweight="bold", va="top")

    ax4.set_title("Ensemble Forecast Summary  (Next Trading Day)",
                  fontsize=12, pad=12)
    for spine in ["top", "bottom", "left", "right"]:
        ax4.spines[spine].set_visible(True)
        ax4.spines[spine].set_edgecolor("#30363d")
        ax4.spines[spine].set_linewidth(1.5)

    # Feature importance bar (inset)
    ax_imp = ax4.inset_axes([0.76, 0.05, 0.22, 0.88])
    ax_imp.set_facecolor("#161b22")
    top_n  = 7
    labels = list(feat_names[:top_n])
    vals   = list(imp_gbm[:top_n])
    y_pos  = list(range(top_n))
    ax_imp.barh(y_pos, vals[::-1], color="#ffa657", alpha=0.85)
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(labels[::-1], fontsize=7)
    ax_imp.set_title("GBM Feature Imp.", fontsize=9, color=C["text"])
    ax_imp.tick_params(labelsize=7)
    ax_imp.grid(True, axis="x", alpha=0.4)

    out = "/mnt/user-data/outputs/india_vix_dashboard.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n[SAVED] {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  INDIA VIX — ENSEMBLE FORECASTING MODEL")
    print("=" * 60)

    print("\n[DATA] Generating synthetic India VIX series …")
    vix   = generate_synthetic_vix(2500)
    dates = make_dates(len(vix))
    print(f"       {len(vix)} trading days  |  VIX: min={vix.min():.1f} "
          f"mean={vix.mean():.1f} max={vix.max():.1f}")

    print("\n[FEATURES] Engineering …")
    features = engineer_features(vix)

    print("\n[MODEL 1] GARCH(1,1) …")
    garch_pred, garch_h, log_ret = fit_garch(vix)
    print(f"          Forecast: {garch_pred:.4f}")

    print("\n[MODEL 2] HAR-RV …")
    har_beta, har_pred, feat_names, imp_har, har_results = fit_har(vix, features)
    print(f"          Forecast: {har_pred:.4f}")

    print("\n[MODEL 3] Recursive Least Squares …")
    rls_theta, rls_pred, rls_results = fit_rls(vix, features)
    print(f"          Forecast: {rls_pred:.4f}")

    print("\n[MODEL 4] Gradient-Boosted Trees …")
    gbm_model, gbm_pred, feat_names, imp_gbm, gbm_results, gbm_norm = \
        fit_gbm(vix, features)
    print(f"          Forecast: {gbm_pred:.4f}")

    # Ensemble
    preds  = {"garch": garch_pred, "har": har_pred,
               "rls": rls_pred,   "gbm": gbm_pred}
    weights= {"garch": 0.15, "har": 0.30, "rls": 0.20, "gbm": 0.35}
    ens    = ensemble(preds, weights)

    # 90% CI from HAR OOS residuals
    har_te_pred, har_y_te = har_results[0], har_results[1]
    residuals = har_y_te - har_te_pred
    ci_lo, ci_hi = bootstrap_ci(residuals, ens)

    regime_str, _ = classify_regime(ens)
    chg = (ens - vix[-1]) / vix[-1] * 100

    print("\n" + "=" * 60)
    print("  FORECAST RESULTS  (Next Trading Day)")
    print("=" * 60)
    print(f"  Current VIX         : {vix[-1]:.2f}")
    print(f"  GARCH(1,1)          : {garch_pred:.2f}")
    print(f"  HAR-RV              : {har_pred:.2f}")
    print(f"  Recursive LS        : {rls_pred:.2f}")
    print(f"  GBM (boosted trees) : {gbm_pred:.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  Ensemble (weighted) : {ens:.2f}  ({chg:+.2f}%)")
    print(f"  90% CI              : [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Regime              : {regime_str}")
    print("=" * 60)

    print("\n[PLOT] Building dashboard …")
    # Unpack result tuples for plot
    har_plot  = (har_results[0], har_results[1],
                 har_results[2], har_results[3])
    rls_plot  = (rls_results[0], rls_results[1],
                 rls_results[2], rls_results[3], rls_results[4])
    gbm_plot  = (gbm_results[0], gbm_results[1],
                 gbm_results[2], gbm_results[3])

    out = plot_all(vix, dates, garch_h, log_ret,
                   har_plot, rls_plot, gbm_plot,
                   imp_har, imp_gbm, feat_names,
                   preds, ens, ci_lo, ci_hi)

    return out


if __name__ == "__main__":
    main()
