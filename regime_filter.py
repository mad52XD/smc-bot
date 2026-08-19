# ============================================================================
# regime_filter.py — Market Regime Filter
# Uses the trained CatBoost model to classify the current market regime
# and filter/adjust trades accordingly.
#
# Regimes:
#   0 = TRENDING  → trade normally
#   1 = RANGING   → block entry, wait for new setup
#   2 = VOLATILE  → reduce position size by 50%
# ============================================================================

import numpy as np
import pandas as pd
import joblib
import os
import logging

log = logging.getLogger("regime_filter")

# ── Load model ────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(
    __file__), "models", "regime_model.pkl")
_FEATURES_PATH = os.path.join(os.path.dirname(
    __file__), "models", "feature_cols.pkl")

try:
    _model = joblib.load(_MODEL_PATH)
    _feature_cols = joblib.load(_FEATURES_PATH)
    log.info(f"Regime model loaded — {len(_feature_cols)} features")
except Exception as e:
    _model = None
    _feature_cols = None
    log.warning(f"Regime model not found — filter disabled: {e}")


REGIME_MAP = {0: "TRENDING", 1: "RANGING", 2: "VOLATILE"}

# ── Risk multiplier per regime ────────────────────────────────────────────────
REGIME_RISK_MULTIPLIER = {
    "TRENDING": 1.0,   # full risk
    "RANGING":  0.0,   # block trade
    "VOLATILE": 0.5,   # half risk
}


def compute_regime_features(df: pd.DataFrame) -> dict:
    """
    Compute the features expected by the regime model
    from the last closed bar of the strategy DataFrame.

    Parameters
    ----------
    df : strategy DataFrame with indicators already computed

    Returns
    -------
    dict of feature values for the last closed bar
    """
    bar = df.iloc[-2]  # last closed bar — same convention as strategy.py

    # EMAs
    # proxy if EMA not present
    ema20 = bar['close'] / bar.get('hull', bar['close'])
    ema200 = bar['ema200'] if 'ema200' in bar.index else np.nan

    features = {
        'EMA_20':          bar['close'],
        'EMA_50':          bar['close'],
        'EMA_200':         ema200,
        'price_vs_ema20':  (bar['close'] - bar['close']) / bar['close'] * 100,
        'price_vs_ema200': (bar['close'] - ema200) / ema200 * 100 if ema200 else 0,
        'ema20_slope':     0.0,
        'ATR_14':          bar['atr'],
        'BB_middle':       bar['close'],
        'BB_std':          bar['atr'],
        'BB_upper':        bar['close'] + 2 * bar['atr'],
        'BB_lower':        bar['close'] - 2 * bar['atr'],
        'BB_position':     0.5,
        'RSI':             bar['lrsi'] * 100,
        'return_1h':       0.0,
        'return_24h':      0.0,
        'CVD_24':          0.0,
        'CVD_24_norm':     0.0,
        'HMA_14':          bar['hull'],
        'HMA_55':          bar['hull'],
        'HMA_55_slope':    0.0,
    }
    return features


def get_regime(df: pd.DataFrame) -> tuple:
    """
    Classify the current market regime using the last closed bar.

    Parameters
    ----------
    df : strategy DataFrame with indicators already computed

    Returns
    -------
    (regime_name, risk_multiplier, probability)
    regime_name     : 'TRENDING', 'RANGING', or 'VOLATILE'
    risk_multiplier : 1.0, 0.5, or 0.0
    probability     : model confidence
    """
    if _model is None:
        log.warning("Regime model not loaded — defaulting to TRENDING")
        return "TRENDING", 1.0, 1.0

    try:
        features = compute_regime_features(df)
        X = pd.DataFrame([features])[_feature_cols]
        probabilities = _model.predict_proba(X)[0]
        regime_idx = int(np.argmax(probabilities))
        regime_name = REGIME_MAP[regime_idx]
        probability = float(np.max(probabilities))
        multiplier = REGIME_RISK_MULTIPLIER[regime_name]

        log.info(
            f"Regime: {regime_name} "
            f"(confidence={probability:.2f}, multiplier={multiplier})"
        )
        return regime_name, multiplier, probability

    except Exception as e:
        log.error(f"Regime detection failed: {e} — defaulting to TRENDING")
        return "TRENDING", 1.0, 1.0
