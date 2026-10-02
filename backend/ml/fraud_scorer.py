"""Fraud Detection Ensemble Scorer (Model 1).

Architecture:
    - Logistic Regression  : baseline / conservative probability
    - XGBoost              : main supervised model (best recall)
    - Isolation Forest     : anomaly gate (boosts fraud score for outliers)

Final probability = weighted blend of the three, clamped to [0, 1].

Decision tiers (matching bankassure risk policy):
    prob <  0.35  -> APPROVE  (straight-through)
    prob >= 0.35 and prob < 0.75 -> REVIEW (analyst queue)
    prob >= 0.75  -> REJECT   (automatic block)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ML_DIR = Path(__file__).resolve().parent
MODEL_DIR = ML_DIR / "models"

WEIGHTS = {
    "lr": 0.35,
    "xgb": 0.55,
    "iforest": 0.10,
}

DECISION_THRESHOLDS = {
    "reject": 0.75,
    "review_lo": 0.35,
}


@dataclass
class FraudScoreResult:
    transaction_id: str
    fraud_probability: float
    decision: str
    component_scores: dict[str, float]
    reasons: list[str]


class FraudScorer:
    """Loads the 3-model ensemble once and exposes score_transaction()."""

    def __init__(self) -> None:
        self._loaded = False

        self.lr_pipe = None
        self.xgb_pipe = None
        self.iforest_pipe = None

        self.lr_threshold = 0.5
        self.xgb_threshold = 0.5
        self.if_q_normal: float = 0.0
        self.if_q_anomaly: float = 0.0

    # ------------------------------------------------------------------
    # Lazy loading (so the server only pays the cost on first request)
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        self.lr_pipe = joblib.load(MODEL_DIR / "fraud_logistic_regression.joblib")
        self.xgb_pipe = joblib.load(MODEL_DIR / "fraud_xgboost.joblib")
        self.iforest_pipe = joblib.load(MODEL_DIR / "fraud_isolation_forest.joblib")

        with open(MODEL_DIR / "fraud_logistic_regression_metadata.json") as f:
            self.lr_threshold = float(json.load(f)["default_threshold"])

        with open(MODEL_DIR / "fraud_xgboost_metadata.json") as f:
            self.xgb_threshold = float(json.load(f)["default_threshold"])

        with open(MODEL_DIR / "fraud_isolation_forest_metadata.json") as f:
            md = json.load(f)
            self.if_q_normal = float(md["calibration_q_normal"])
            self.if_q_anomaly = float(md["calibration_q_anomaly"])

        self._loaded = True

    # ------------------------------------------------------------------
    # Feature bridge: API input -> DataFrame that matches training schema
    # ------------------------------------------------------------------
    @staticmethod
    def _to_frame(tx: dict[str, Any]) -> pd.DataFrame:
        """Convert a raw transaction dict to the 16-feature schema used at
        training time.

        Required input keys (loose match):
            transaction_id, amount, hour, day_of_week,
            is_international, merchant_risk, transaction_type,
            merchant_category, country,
            customer_transaction_count, customer_avg_amount,
            transactions_24h, transactions_7d,
            customer_international_ratio.

        Optional defaults are provided for fields that the caller might omit
        (e.g., a brand-new customer has no frequency yet).
        """
        tx = dict(tx)

        hour = int(tx.get("hour", 12))
        dow = int(tx.get("day_of_week", tx.get("day", 1)))
        is_weekend = 1 if dow in (5, 6) else 0
        is_night = 1 if (hour >= 22 or hour < 6) else 0

        customer_avg_amount = float(tx.get("customer_avg_amount", 0.0) or 0.0)
        amount = float(tx.get("amount", 0.0) or 0.0)
        amount_vs_avg = (
            amount / customer_avg_amount if customer_avg_amount > 1e-6 else 0.0
        )

        row = {
            "amount": amount,
            "merchant_risk": float(tx.get("merchant_risk", 0.0) or 0.0),
            "is_international": bool(tx.get("is_international", False)),
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "transaction_type": str(tx.get("transaction_type", "card_payment")),
            "merchant_category": str(tx.get("merchant_category", "retail")),
            "country": str(tx.get("country", "Unknown")),
            "customer_transaction_count": int(tx.get("customer_transaction_count", 0) or 0),
            "customer_avg_amount": customer_avg_amount,
            "amount_vs_customer_avg": amount_vs_avg,
            "transactions_24h": float(tx.get("transactions_24h",
                                             tx.get("transaction_frequency", 0.0)) or 0.0),
            "transactions_7d": float(tx.get("transactions_7d", 0.0) or 0.0),
            "customer_international_ratio": float(
                tx.get("customer_international_ratio", 0.0) or 0.0
            ),
        }
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Probability calibration helpers
    #
    # Each model lives on a very different probability scale:
    #   - LR outputs calibrated probabilities at threshold ~0.5
    #   - XGB (with scale_pos_weight) outputs low probabilities; its best
    #     F1 threshold is ~0.365 meaning "0.365 = suspicious"
    #   - IsolationForest probability is a rough 0..1 anomaly indicator
    #
    # We normalize each score through its tuned decision threshold so that
    # every model contributes on a common scale where:
    #     prob = threshold  -> normalized_prob = 0.50
    #     prob > threshold  -> normalized_prob > 0.50 (risky)
    #     prob < threshold  -> normalized_prob < 0.50 (safe)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(p: float, thr: float) -> float:
        """Map p into a shared probability scale centered at threshold."""
        p = float(max(1e-6, min(1 - 1e-6, p)))
        thr = float(max(1e-3, min(1 - 1e-3, thr)))

        # Log-odds shift so that threshold -> 0.5 on the shared scale.
        # shifted_logit = logit(p) - logit(thr)
        import math as _m

        shifted = _m.log(p / (1 - p)) - _m.log(thr / (1 - thr))
        out = 1.0 / (1.0 + _m.exp(-shifted))
        return float(max(0.0005, min(0.9995, out)))

    def _lr_prob(self, pipeline, X: pd.DataFrame, thr: float) -> tuple[float, float]:
        raw = float(pipeline.predict_proba(X)[0, 1])
        return raw, self._normalize(raw, thr)

    def _xgb_prob(self, pipeline, X: pd.DataFrame, thr: float) -> tuple[float, float]:
        raw = float(pipeline.predict_proba(X)[0, 1])
        return raw, self._normalize(raw, thr)

    def _if_prob(self, pipeline, X: pd.DataFrame) -> tuple[float, float]:
        raw = float(pipeline.decision_function(X)[0])
        s = -raw
        raw_prob = (s - self.if_q_normal) / (self.if_q_anomaly - self.if_q_normal + 1e-9)
        raw_prob = float(max(0.0, min(1.0, raw_prob)))
        # iForest threshold = 0.5 by convention (midpoint of its calibrated range)
        return raw_prob, self._normalize(raw_prob, 0.5)

    # ------------------------------------------------------------------
    # Reason-code extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _reasons(tx: dict[str, Any], scores: dict[str, float],
                 ensemble_prob: float) -> list[str]:
        reasons: list[str] = []
        if float(tx.get("is_international", False)):
            reasons.append("international_transaction")
        mr = float(tx.get("merchant_risk", 0.0) or 0.0)
        if mr >= 8.0:
            reasons.append(f"high_merchant_risk({mr:.1f})")
        hour = int(tx.get("hour", 12))
        if hour >= 22 or hour < 6:
            reasons.append(f"late_night_hour({hour})")
        avg = float(tx.get("customer_avg_amount", 0.0) or 0.0)
        amt = float(tx.get("amount", 0.0) or 0.0)
        if avg > 0 and amt / avg >= 2.0:
            reasons.append(f"amount_{amt/avg:.1f}x_customer_avg")
        vel = float(tx.get("transactions_24h", 0.0) or 0.0)
        if vel >= 5:
            reasons.append(f"high_velocity_{vel:.0f}_24h")
        if scores["iforest"] >= 0.65:
            reasons.append("anomalous_pattern_detected")
        if ensemble_prob >= 0.75:
            reasons.append("ensemble_score_critical")
        if not reasons:
            reasons.append("no_risk_flags")
        return reasons

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    @staticmethod
    def _decide(prob: float) -> str:
        if prob >= DECISION_THRESHOLDS["reject"]:
            return "REJECT"
        if prob >= DECISION_THRESHOLDS["review_lo"]:
            return "REVIEW"
        return "APPROVE"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score_transaction(self, transaction: dict[str, Any]) -> FraudScoreResult:
        self._ensure_loaded()

        tid = str(transaction.get("transaction_id", "UNKNOWN"))
        X = self._to_frame(transaction)

        lr_raw, lr_n = self._lr_prob(self.lr_pipe, X, self.lr_threshold)
        xgb_raw, xgb_n = self._xgb_prob(self.xgb_pipe, X, self.xgb_threshold)
        if_raw, if_n = self._if_prob(self.iforest_pipe, X)

        # --------------------------------------------------------------
        # Ensemble design (battle-tested for this dataset)
        #
        # BASELINE = Logistic Regression raw probability.  LR is the best
        # calibrated model here (well-trained log-loss, ROC-AUC champion,
        # threshold ~0.5) so it is always the ground-truth probability.
        #
        # BOOST = one-way signals from the other two models. We ONLY boost
        # risk upwards when an ancillary model agrees (> threshold). We
        # NEVER reduce risk because XGB / iForest are untrustworthy on
        # their "safe" calls (they both missed the obvious fraud sample
        # TX18292 at training time because of calibration / weak signal).
        #
        #   base_logit + sum(boost_weight * vote_lo if vote_lo > 0 else 0)
        # --------------------------------------------------------------
        base = lr_raw

        def vote_logit(p_norm: float) -> float:
            return math.log(max(p_norm, 1e-6) / max(1 - p_norm, 1e-6))

        xgb_lo = vote_logit(xgb_n)
        if_lo = vote_logit(if_n)

        base_lo = math.log(max(base, 1e-6) / max(1 - base, 1e-6))
        boosted_lo = (
            base_lo
            + WEIGHTS["xgb"] * max(xgb_lo, 0.0)
            + WEIGHTS["iforest"] * max(if_lo, 0.0)
        )
        prob = float(1.0 / (1.0 + math.exp(-boosted_lo)))
        prob = max(0.0, min(1.0, prob))

        scores = {
            "logistic_regression": round(lr_raw, 4),
            "xgboost": round(xgb_raw, 4),
            "iforest": round(if_raw, 4),
            "normalized": {
                "logistic_regression": round(lr_n, 4),
                "xgboost": round(xgb_n, 4),
                "iforest": round(if_n, 4),
            },
            "votes": {
                "xgboost_logit": round(xgb_lo, 4),
                "iforest_logit": round(if_lo, 4),
                "xgb_boosted": round(max(xgb_lo, 0.0) * WEIGHTS["xgb"], 4),
                "iforest_boosted": round(max(if_lo, 0.0) * WEIGHTS["iforest"], 4),
            },
        }

        decision = self._decide(prob)
        reasons = self._reasons(transaction, scores, prob)

        return FraudScoreResult(
            transaction_id=tid,
            fraud_probability=round(prob, 4),
            decision=decision,
            component_scores=scores,
            reasons=reasons,
        )


# Module-level singleton for the FastAPI layer
_scorer: FraudScorer | None = None


def get_scorer() -> FraudScorer:
    global _scorer
    if _scorer is None:
        _scorer = FraudScorer()
    return _scorer


__all__ = [
    "FraudScorer",
    "FraudScoreResult",
    "get_scorer",
    "WEIGHTS",
    "DECISION_THRESHOLDS",
]
