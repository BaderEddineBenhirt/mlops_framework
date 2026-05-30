# %%
# ============================================================
# STRUCTURED COLAB NOTEBOOK
# HYBRID POLICY-DRIVEN MLOPS DECISION FRAMEWORK
# ============================================================
# It creates a GitHub-like experimental repository:
# - policies/*.yaml
# - outputs/*.json
# - reports/*.csv
# - README.md
# - requirements.txt
# - zip archive ready to download
#
# It evaluates:
# - multiple datasets
# - multiple drift scenarios
# - model metrics
# - decision gate
# - retraining advisor
# - baselines
# - sensitivity analysis
# - ablation study
# - decision-gate execution time
# ============================================================


# %%
# ============================================================
# 0. INSTALL AND IMPORT DEPENDENCIES
# ============================================================

import sys
import subprocess
import importlib.util

def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

for import_name, pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"),
    ("joblib", "joblib"),
]:
    ensure_package(import_name, pip_name)

import os
import json
import yaml
import time
import shutil
import random
import platform
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib

from scipy.stats import wasserstein_distance
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)


# %%
# ============================================================
# 1. CREATE REPOSITORY-LIKE STRUCTURE
# ============================================================

BASE_DIR = Path("/content/mlops-decision-framework")
if BASE_DIR.exists():
    shutil.rmtree(BASE_DIR)

for subdir in ["data", "policies", "models", "outputs", "reports", "src"]:
    (BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)

print("Created project:", BASE_DIR)


# %%
# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path

def save_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return path

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_float(x):
    return float(np.asarray(x).item()) if np.asarray(x).shape == () else float(x)


# %%
# ============================================================
# 3. CAPTURE ENVIRONMENT
# ============================================================

environment = {
    "execution_date_utc": datetime.utcnow().isoformat() + "Z",
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "pyyaml": yaml.__version__,
    "joblib": joblib.__version__,
}
save_json(environment, BASE_DIR / "outputs" / "environment.json")


# %%
# ============================================================
# 4. POLICY DESCRIPTORS
# ============================================================
# These are governance descriptors. They are not "AI".
# They externalize thresholds and decision weights so the gate is auditable.

decision_policy = {
    "policy_id": "decision-policy-v1",
    "description": "Policy for model promotion using performance, latency, cost, traceability, and drift.",
    "hard_constraints": {
        "min_accuracy": 0.85,
        "min_f1": 0.85,
        "max_mean_latency_ms": 10.0,
        "min_traceability_score": 0.80,
        "max_drift_score": 0.15
    },
    "score_weights": {
        "f1": 0.40,
        "accuracy": 0.15,
        "latency": 0.15,
        "cost": 0.10,
        "drift_stability": 0.10,
        "traceability": 0.10
    },
    "promotion_threshold": 0.85,
    "tie_breaking": [
        "highest_decision_score",
        "lowest_mean_latency_ms",
        "smallest_model_size_mb"
    ],
    "weight_origin": "Initial policy assumption; must be calibrated by domain expertise, sensitivity analysis, or historical deployment logs."
}

monitoring_policy = {
    "policy_id": "monitoring-policy-v1",
    "description": "Monitoring and drift policy.",
    "drift_threshold": 0.15,
    "drift_methods": {
        "numerical_features": "wasserstein_distance",
        "label_distribution": "total_variation_distance"
    },
    "supported_drift_types": [
        "stable",
        "covariate_drift",
        "label_shift",
        "sudden_drift",
        "progressive_drift",
        "recurring_drift"
    ]
}

retraining_policy = {
    "policy_id": "retraining-policy-v1",
    "description": "Retraining recommendation policy.",
    "strategy": "hybrid",
    "drift_threshold": 0.15,
    "retraining_queue": "standard-training",
    "actions": {
        "stable": "no_retraining_required",
        "drifted": "retraining_recommended",
        "critical": "retraining_triggered"
    }
}

lifecycle_descriptor = {
    "descriptor_id": "lifecycle-descriptor-v1",
    "description": "Lifecycle descriptor for CI/CD/CT integration.",
    "stages": [
        "train_candidate_models",
        "evaluate_metrics",
        "generate_drift_report",
        "execute_decision_gate",
        "recommend_retraining",
        "export_decision_artifacts"
    ],
    "artifacts": [
        "metrics.json",
        "drift_report.json",
        "promotion_decision.json",
        "retraining_recommendation.json",
        "baseline_results.json",
        "sensitivity_results.json",
        "ablation_results.json",
        "execution_time.json"
    ]
}

save_yaml(decision_policy, BASE_DIR / "policies" / "decision_policy.yaml")
save_yaml(monitoring_policy, BASE_DIR / "policies" / "monitoring_policy.yaml")
save_yaml(retraining_policy, BASE_DIR / "policies" / "retraining_policy.yaml")
save_yaml(lifecycle_descriptor, BASE_DIR / "policies" / "lifecycle_descriptor.yaml")

decision_policy = load_yaml(BASE_DIR / "policies" / "decision_policy.yaml")
monitoring_policy = load_yaml(BASE_DIR / "policies" / "monitoring_policy.yaml")
retraining_policy = load_yaml(BASE_DIR / "policies" / "retraining_policy.yaml")


# %%
# ============================================================
# 5. DATASETS
# ============================================================

def dataset_breast_cancer():
    data = load_breast_cancer()
    return {
        "name": "wisconsin_breast_cancer",
        "domain": "small tabular binary classification",
        "source": "scikit-learn built-in dataset",
        "X": data.data.astype(float),
        "y": data.target.astype(int),
        "feature_names": list(data.feature_names),
        "class_names": [str(x) for x in data.target_names]
    }

def dataset_wine():
    data = load_wine()
    return {
        "name": "wine_multiclass",
        "domain": "tabular multiclass classification",
        "source": "scikit-learn built-in dataset",
        "X": data.data.astype(float),
        "y": data.target.astype(int),
        "feature_names": list(data.feature_names),
        "class_names": [str(x) for x in data.target_names]
    }

def dataset_synthetic_iot(random_state=42):
    X, y = make_classification(
        n_samples=2500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        weights=[0.65, 0.35],
        class_sep=1.2,
        flip_y=0.02,
        random_state=random_state,
    )
    return {
        "name": "synthetic_iot_sensor",
        "domain": "synthetic industrial/IoT-like binary classification",
        "source": "synthetic make_classification",
        "X": X.astype(float),
        "y": y.astype(int),
        "feature_names": [f"sensor_{i}" for i in range(X.shape[1])],
        "class_names": ["normal", "fault"]
    }

def dataset_synthetic_timeseries(random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = 3000
    n_features = 12
    t = np.arange(n_samples)
    X = np.stack([
        np.sin(t / (8 + i)) + 0.1 * rng.normal(size=n_samples)
        for i in range(n_features)
    ], axis=1)
    trend = np.linspace(0, 1, n_samples).reshape(-1, 1)
    X = X + 0.15 * trend
    risk = 0.4 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * rng.normal(size=n_samples)
    y = (risk > np.quantile(risk, 0.65)).astype(int)
    return {
        "name": "synthetic_timeseries_telemetry",
        "domain": "synthetic time-series-like telemetry classification",
        "source": "synthetic generated telemetry",
        "X": X.astype(float),
        "y": y.astype(int),
        "feature_names": [f"telemetry_{i}" for i in range(n_features)],
        "class_names": ["stable", "degraded"]
    }

datasets = [
    dataset_breast_cancer(),
    dataset_wine(),
    dataset_synthetic_iot(),
    dataset_synthetic_timeseries()
]

dataset_metadata = []
for ds in datasets:
    classes, counts = np.unique(ds["y"], return_counts=True)
    dataset_metadata.append({
        "name": ds["name"],
        "domain": ds["domain"],
        "source": ds["source"],
        "n_samples": int(ds["X"].shape[0]),
        "n_features": int(ds["X"].shape[1]),
        "class_distribution": {str(c): int(n) for c, n in zip(classes, counts)}
    })

save_json(dataset_metadata, BASE_DIR / "outputs" / "dataset_metadata.json")
pd.DataFrame(dataset_metadata).to_csv(BASE_DIR / "reports" / "dataset_metadata.csv", index=False)


# %%
# ============================================================
# 6. MODELS AND METRICS
# ============================================================

def build_models(seed=42):
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=seed
        )
    }

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": to_float(accuracy_score(y_true, y_pred)),
        "precision": to_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": to_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": to_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def measure_latency_ms(model, X_sample, n_runs=30):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_sample)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return {
        "mean_latency_ms": to_float(np.mean(times)),
        "p95_latency_ms": to_float(np.percentile(times, 95))
    }

def model_size_mb(path):
    return to_float(Path(path).stat().st_size / (1024 * 1024))

def traceability_score(trace):
    fields = [
        "model_artifact_path",
        "dataset_identifier",
        "evaluation_metrics",
        "policy_identifiers",
        "decision_reasons"
    ]
    detail = {}
    score = 0.0
    for f in fields:
        ok = bool(trace.get(f))
        detail[f] = ok
        if ok:
            score += 0.20
    return round(score, 4), detail

def cost_proxy(mean_latency_ms, p95_latency_ms, size_mb):
    mean_cost = min(mean_latency_ms / 50.0, 1.0)
    p95_cost = min(p95_latency_ms / 100.0, 1.0)
    size_cost = min(size_mb / 500.0, 1.0)
    return to_float(0.40 * mean_cost + 0.30 * p95_cost + 0.30 * size_cost)


# %%
# ============================================================
# 7. DRIFT
# ============================================================

def total_variation_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    return to_float(0.5 * np.abs(p - q).sum())

def label_shift_score(y_ref, y_cur):
    labels = sorted(set(np.unique(y_ref)).union(set(np.unique(y_cur))))
    p = np.array([(y_ref == l).mean() for l in labels])
    q = np.array([(y_cur == l).mean() for l in labels])
    return total_variation_distance(p, q)

def covariate_drift_score(X_ref, X_cur):
    scores = []
    for j in range(X_ref.shape[1]):
        scale = np.std(X_ref[:, j]) + 1e-8
        scores.append(wasserstein_distance(X_ref[:, j], X_cur[:, j]) / scale)
    return to_float(min(np.mean(scores) / 5.0, 1.0))

def generate_drift(X_ref, y_ref, scenario, seed=42):
    rng = np.random.default_rng(seed)
    X_cur = X_ref.copy()
    y_cur = y_ref.copy()
    n, d = X_cur.shape
    k = max(1, min(5, d))

    if scenario == "stable":
        pass

    elif scenario == "covariate_drift":
        X_cur[:, :k] += rng.normal(loc=1.5, scale=0.8, size=(n, k))

    elif scenario == "label_shift":
        labels, counts = np.unique(y_cur, return_counts=True)
        majority = labels[np.argmax(counts)]
        majority_idx = np.where(y_cur == majority)[0]
        other_idx = np.where(y_cur != majority)[0]
        selected_majority = rng.choice(majority_idx, size=max(1, int(0.8 * n)), replace=True)
        selected_other = rng.choice(other_idx, size=max(1, int(0.2 * n)), replace=True)
        idx = np.concatenate([selected_majority, selected_other])
        rng.shuffle(idx)
        X_cur = X_cur[idx]
        y_cur = y_cur[idx]

    elif scenario == "sudden_drift":
        cut = n // 2
        X_cur[cut:, :k] += rng.normal(loc=3.0, scale=1.0, size=(n - cut, k))

    elif scenario == "progressive_drift":
        intensity = np.linspace(0, 3.0, n).reshape(-1, 1)
        X_cur[:, :k] += intensity * rng.normal(loc=1.0, scale=0.3, size=(n, k))

    elif scenario == "recurring_drift":
        block = max(10, n // 4)
        for start in range(0, n, 2 * block):
            end = min(start + block, n)
            X_cur[start:end, :k] += rng.normal(loc=2.0, scale=0.7, size=(end - start, k))

    else:
        raise ValueError(f"Unknown drift scenario: {scenario}")

    min_len = min(len(y_ref), len(y_cur))
    cov = covariate_drift_score(X_ref[:min_len], X_cur[:min_len])
    lab = label_shift_score(y_ref[:min_len], y_cur[:min_len])
    glob = max(cov, lab)

    return {
        "scenario": scenario,
        "X_current": X_cur,
        "y_current": y_cur,
        "covariate_drift_score": cov,
        "label_shift_score": lab,
        "global_drift_score": glob,
        "drift_detected": bool(glob > monitoring_policy["drift_threshold"])
    }


# %%
# ============================================================
# 8. DECISION GATE
# ============================================================

def normalize_latency(latency, max_latency):
    return to_float(max(0.0, 1.0 - latency / max_latency))

def normalize_cost(cost):
    return to_float(max(0.0, 1.0 - cost))

def normalize_drift(drift, max_drift):
    return to_float(max(0.0, 1.0 - drift / max_drift))

def decision_score(metric, drift_score, policy):
    w = policy["score_weights"]
    hard = policy["hard_constraints"]
    return to_float(
        w["f1"] * metric["f1_score"]
        + w["accuracy"] * metric["accuracy"]
        + w["latency"] * normalize_latency(metric["mean_latency_ms"], hard["max_mean_latency_ms"])
        + w["cost"] * normalize_cost(metric["cost_proxy"])
        + w["drift_stability"] * normalize_drift(drift_score, hard["max_drift_score"])
        + w["traceability"] * metric["traceability_score"]
    )

def execute_decision_gate(candidate_metrics, drift_report, policy):
    start = time.perf_counter()
    hard = policy["hard_constraints"]
    ranked = []

    for m in candidate_metrics:
        drift = drift_report["global_drift_score"]
        score = decision_score(m, drift, policy)
        violations = []

        if m["accuracy"] < hard["min_accuracy"]:
            violations.append("accuracy_below_minimum")
        if m["f1_score"] < hard["min_f1"]:
            violations.append("f1_below_minimum")
        if m["mean_latency_ms"] > hard["max_mean_latency_ms"]:
            violations.append("latency_above_maximum")
        if m["traceability_score"] < hard["min_traceability_score"]:
            violations.append("traceability_below_minimum")
        if drift > hard["max_drift_score"]:
            violations.append("drift_above_maximum")

        if "drift_above_maximum" in violations:
            decision = "retrain"
        elif violations:
            decision = "reject"
        elif score >= policy["promotion_threshold"]:
            decision = "promote"
        else:
            decision = "keep_current"

        ranked.append({
            "dataset": m["dataset"],
            "scenario": drift_report["scenario"],
            "model_name": m["model_name"],
            "decision": decision,
            "decision_score": score,
            "violations": violations,
            "accuracy": m["accuracy"],
            "f1_score": m["f1_score"],
            "mean_latency_ms": m["mean_latency_ms"],
            "p95_latency_ms": m["p95_latency_ms"],
            "model_size_mb": m["model_size_mb"],
            "cost_proxy": m["cost_proxy"],
            "traceability_score": m["traceability_score"],
            "drift_score": drift
        })

    priority = {"promote": 3, "keep_current": 2, "retrain": 1, "reject": 0}
    ranked = sorted(
        ranked,
        key=lambda x: (
            priority[x["decision"]],
            x["decision_score"],
            -x["mean_latency_ms"],
            -x["model_size_mb"]
        ),
        reverse=True
    )

    selected = ranked[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "dataset": selected["dataset"],
        "scenario": drift_report["scenario"],
        "selected_model": selected["model_name"],
        "final_decision": selected["decision"],
        "selected_decision_score": selected["decision_score"],
        "ranking": ranked,
        "execution_time_ms": to_float(elapsed_ms),
        "policy_id": policy["policy_id"],
        "decision_reason": {
            "hard_constraints_first": True,
            "drift_has_priority_over_promotion": True,
            "tie_breaking": policy["tie_breaking"]
        }
    }

def retraining_advisor(decision, drift_report, policy):
    drift = drift_report["global_drift_score"]
    required = bool(drift > policy["drift_threshold"] or decision["final_decision"] == "retrain")
    action = policy["actions"]["drifted"] if required else policy["actions"]["stable"]
    return {
        "dataset": decision["dataset"],
        "scenario": decision["scenario"],
        "retraining_required": required,
        "action": action,
        "strategy": policy["strategy"],
        "retraining_queue": policy["retraining_queue"],
        "reason": {
            "drift_score": drift,
            "drift_threshold": policy["drift_threshold"],
            "decision_gate": decision["final_decision"]
        }
    }


# %%
# ============================================================
# 9. BASELINES, SENSITIVITY, ABLATION
# ============================================================

def baseline_accuracy_only(candidate_metrics, drift_report):
    best = max(candidate_metrics, key=lambda x: x["accuracy"])
    return {
        "strategy": "accuracy_only",
        "selected_model": best["model_name"],
        "decision": "promote",
        "drift_aware": False,
        "drift_score": drift_report["global_drift_score"]
    }

def baseline_f1_threshold(candidate_metrics, drift_report, threshold=0.85):
    eligible = [m for m in candidate_metrics if m["f1_score"] >= threshold]
    if eligible:
        best = max(eligible, key=lambda x: x["f1_score"])
        return {
            "strategy": "f1_threshold",
            "selected_model": best["model_name"],
            "decision": "promote",
            "drift_aware": False,
            "drift_score": drift_report["global_drift_score"]
        }
    return {
        "strategy": "f1_threshold",
        "selected_model": None,
        "decision": "reject",
        "drift_aware": False,
        "drift_score": drift_report["global_drift_score"]
    }

def baseline_fixed_retraining(candidate_metrics, drift_report):
    retrain = drift_report["scenario"] in ["progressive_drift", "recurring_drift"]
    return {
        "strategy": "fixed_retraining_schedule",
        "selected_model": None,
        "decision": "retrain" if retrain else "promote",
        "drift_aware": False,
        "drift_score": drift_report["global_drift_score"]
    }

def baseline_manual_policy(candidate_metrics, drift_report):
    if drift_report["global_drift_score"] > 0.15:
        return {
            "strategy": "manual_policy",
            "selected_model": None,
            "decision": "retrain",
            "drift_aware": True,
            "drift_score": drift_report["global_drift_score"]
        }
    eligible = [m for m in candidate_metrics if m["f1_score"] >= 0.85 and m["mean_latency_ms"] <= 10.0]
    if eligible:
        best = max(eligible, key=lambda x: x["f1_score"])
        return {
            "strategy": "manual_policy",
            "selected_model": best["model_name"],
            "decision": "promote",
            "drift_aware": True,
            "drift_score": drift_report["global_drift_score"]
        }
    return {
        "strategy": "manual_policy",
        "selected_model": None,
        "decision": "reject",
        "drift_aware": True,
        "drift_score": drift_report["global_drift_score"]
    }

def compare_baselines(candidate_metrics, drift_report, proposed):
    rows = [
        baseline_accuracy_only(candidate_metrics, drift_report),
        baseline_f1_threshold(candidate_metrics, drift_report),
        baseline_fixed_retraining(candidate_metrics, drift_report),
        baseline_manual_policy(candidate_metrics, drift_report),
        {
            "strategy": "proposed_decision_gate",
            "selected_model": proposed["selected_model"],
            "decision": proposed["final_decision"],
            "drift_aware": True,
            "drift_score": drift_report["global_drift_score"]
        }
    ]
    proposed_row = rows[-1]
    for r in rows:
        r["agreement_with_proposed"] = bool(
            r["decision"] == proposed_row["decision"]
            and r["selected_model"] == proposed_row["selected_model"]
        )
        r["false_promotion_under_drift"] = bool(
            r["decision"] == "promote"
            and drift_report["global_drift_score"] > decision_policy["hard_constraints"]["max_drift_score"]
        )
        r["retraining_triggered"] = bool(r["decision"] == "retrain")
    return rows

def renormalize_weights(policy):
    total = sum(policy["score_weights"].values())
    if total > 0:
        for k in policy["score_weights"]:
            policy["score_weights"][k] = policy["score_weights"][k] / total
    return policy

def run_ablation(candidate_metrics, drift_report, base_policy):
    result = {}
    variants = ["full", "no_drift", "no_latency", "no_cost", "no_traceability"]
    for variant in variants:
        p = json.loads(json.dumps(base_policy))
        if variant == "no_drift":
            p["score_weights"]["drift_stability"] = 0.0
            p["hard_constraints"]["max_drift_score"] = 999.0
        elif variant == "no_latency":
            p["score_weights"]["latency"] = 0.0
            p["hard_constraints"]["max_mean_latency_ms"] = 999.0
        elif variant == "no_cost":
            p["score_weights"]["cost"] = 0.0
        elif variant == "no_traceability":
            p["score_weights"]["traceability"] = 0.0
            p["hard_constraints"]["min_traceability_score"] = 0.0
        p = renormalize_weights(p)
        d = execute_decision_gate(candidate_metrics, drift_report, p)
        result[variant] = {
            "selected_model": d["selected_model"],
            "final_decision": d["final_decision"],
            "selected_decision_score": d["selected_decision_score"]
        }
    return result

def run_sensitivity(candidate_metrics, drift_report, base_policy):
    rows = []

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
        p = json.loads(json.dumps(base_policy))
        p["hard_constraints"]["max_drift_score"] = threshold
        d = execute_decision_gate(candidate_metrics, drift_report, p)
        rows.append({
            "sensitivity_type": "drift_threshold",
            "parameter": "max_drift_score",
            "value": threshold,
            "selected_model": d["selected_model"],
            "final_decision": d["final_decision"],
            "selected_decision_score": d["selected_decision_score"]
        })

    for weight in base_policy["score_weights"]:
        for factor in [0.90, 1.10]:
            p = json.loads(json.dumps(base_policy))
            p["score_weights"][weight] = p["score_weights"][weight] * factor
            p = renormalize_weights(p)
            d = execute_decision_gate(candidate_metrics, drift_report, p)
            rows.append({
                "sensitivity_type": "weight_variation",
                "parameter": weight,
                "value": factor,
                "selected_model": d["selected_model"],
                "final_decision": d["final_decision"],
                "selected_decision_score": d["selected_decision_score"]
            })
    return rows


# %%
# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================

all_metrics = []
all_drift_reports = []
all_decisions = []
all_retraining = []
all_baselines = []
all_ablation = []
all_sensitivity = []
all_execution_time = []

drift_scenarios = [
    "stable",
    "covariate_drift",
    "label_shift",
    "sudden_drift",
    "progressive_drift",
    "recurring_drift"
]

for ds in datasets:
    print("\n============================================================")
    print("DATASET:", ds["name"])
    print("============================================================")

    X = ds["X"]
    y = ds["y"]


    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    models = build_models(seed=42)
    current_metrics = []

    for model_name, model in models.items():
        print("Training:", model_name)
        model.fit(X_train, y_train)

        path = BASE_DIR / "models" / f"{ds['name']}__{model_name}.joblib"
        joblib.dump(model, path)

        y_pred = model.predict(X_test)
        perf = classification_metrics(y_test, y_pred)
        lat = measure_latency_ms(model, X_test[:1], n_runs=30)
        size = model_size_mb(path)

        trace = {
            "model_artifact_path": str(path),
            "dataset_identifier": ds["name"],
            "evaluation_metrics": True,
            "policy_identifiers": decision_policy["policy_id"],
            "decision_reasons": True
        }
        tr_score, tr_detail = traceability_score(trace)

        c_proxy = cost_proxy(lat["mean_latency_ms"], lat["p95_latency_ms"], size)

        record = {
            "dataset": ds["name"],
            "domain": ds["domain"],
            "model_name": model_name,
            "model_artifact_path": str(path),
            "accuracy": perf["accuracy"],
            "precision": perf["precision"],
            "recall": perf["recall"],
            "f1_score": perf["f1_score"],
            "mean_latency_ms": lat["mean_latency_ms"],
            "p95_latency_ms": lat["p95_latency_ms"],
            "model_size_mb": size,
            "cost_proxy": c_proxy,
            "traceability_score": tr_score,
            "traceability_detail": tr_detail
        }

        current_metrics.append(record)
        all_metrics.append(record)

    for scenario in drift_scenarios:
        drift_obj = generate_drift(X_test, y_test, scenario, seed=42)
        drift_report = {
            "dataset": ds["name"],
            "scenario": scenario,
            "covariate_drift_score": drift_obj["covariate_drift_score"],
            "label_shift_score": drift_obj["label_shift_score"],
            "global_drift_score": drift_obj["global_drift_score"],
            "drift_detected": drift_obj["drift_detected"],
            "drift_threshold": monitoring_policy["drift_threshold"],
            "method": monitoring_policy["drift_methods"]
        }
        all_drift_reports.append(drift_report)

        decision = execute_decision_gate(current_metrics, drift_report, decision_policy)
        retraining = retraining_advisor(decision, drift_report, retraining_policy)
        baselines = compare_baselines(current_metrics, drift_report, decision)
        ablation = run_ablation(current_metrics, drift_report, decision_policy)
        sensitivity = run_sensitivity(current_metrics, drift_report, decision_policy)

        all_decisions.append(decision)
        all_retraining.append(retraining)

        for b in baselines:
            all_baselines.append({"dataset": ds["name"], "scenario": scenario, **b})

        all_ablation.append({"dataset": ds["name"], "scenario": scenario, "variants": ablation})

        for s in sensitivity:
            all_sensitivity.append({"dataset": ds["name"], "scenario": scenario, **s})

        all_execution_time.append({
            "dataset": ds["name"],
            "scenario": scenario,
            "decision_gate_execution_time_ms": decision["execution_time_ms"]
        })

# ============================================================
# SAVE DATASETS INTO data/
# ============================================================

for ds in datasets:
    X = ds["X"]
    y = ds["y"]

    df = pd.DataFrame(X, columns=ds["feature_names"])
    df["target"] = y

    dataset_path = BASE_DIR / "data" / f"{ds['name']}.csv"
    df.to_csv(dataset_path, index=False)

print("Datasets saved in data/:")
for file in sorted((BASE_DIR / "data").glob("*.csv")):
    print("-", file)


# %%
# ============================================================
# 11. EXPORT JSON ARTIFACTS
# ============================================================

save_json(all_metrics, BASE_DIR / "outputs" / "metrics.json")
save_json(all_drift_reports, BASE_DIR / "outputs" / "drift_report.json")
save_json(all_decisions, BASE_DIR / "outputs" / "promotion_decision.json")
save_json(all_retraining, BASE_DIR / "outputs" / "retraining_recommendation.json")
save_json(all_baselines, BASE_DIR / "outputs" / "baseline_results.json")
save_json(all_sensitivity, BASE_DIR / "outputs" / "sensitivity_results.json")
save_json(all_ablation, BASE_DIR / "outputs" / "ablation_results.json")
save_json(all_execution_time, BASE_DIR / "outputs" / "execution_time.json")


# %%
# ============================================================
# 12. EXPORT CSV REPORTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
drift_df = pd.DataFrame(all_drift_reports)
decision_df = pd.DataFrame([
    {
        "dataset": d["dataset"],
        "scenario": d["scenario"],
        "selected_model": d["selected_model"],
        "final_decision": d["final_decision"],
        "selected_decision_score": d["selected_decision_score"],
        "execution_time_ms": d["execution_time_ms"]
    }
    for d in all_decisions
])
retraining_df = pd.DataFrame(all_retraining)
baseline_df = pd.DataFrame(all_baselines)
sensitivity_df = pd.DataFrame(all_sensitivity)
execution_time_df = pd.DataFrame(all_execution_time)

metrics_df.to_csv(BASE_DIR / "reports" / "model_metrics_table.csv", index=False)
drift_df.to_csv(BASE_DIR / "reports" / "drift_report_table.csv", index=False)
decision_df.to_csv(BASE_DIR / "reports" / "promotion_decision_table.csv", index=False)
retraining_df.to_csv(BASE_DIR / "reports" / "retraining_recommendation_table.csv", index=False)
baseline_df.to_csv(BASE_DIR / "reports" / "baseline_table.csv", index=False)
sensitivity_df.to_csv(BASE_DIR / "reports" / "sensitivity_table.csv", index=False)
execution_time_df.to_csv(BASE_DIR / "reports" / "execution_time_table.csv", index=False)

baseline_summary = baseline_df.groupby("strategy").agg(
    total_cases=("strategy", "count"),
    false_promotions_under_drift=("false_promotion_under_drift", "sum"),
    retraining_triggers=("retraining_triggered", "sum"),
    agreement_with_proposed=("agreement_with_proposed", "mean")
).reset_index()
baseline_summary.to_csv(BASE_DIR / "reports" / "baseline_summary_table.csv", index=False)

execution_summary = execution_time_df.groupby("dataset").agg(
    mean_execution_time_ms=("decision_gate_execution_time_ms", "mean"),
    p95_execution_time_ms=("decision_gate_execution_time_ms", lambda x: np.percentile(x, 95))
).reset_index()
execution_summary.to_csv(BASE_DIR / "reports" / "execution_time_summary_table.csv", index=False)


# %%
# ============================================================
# 13. GENERATE README AND REQUIREMENTS
# ============================================================

readme = """# Reproducible MLOps Decision Framework Experiment

This repository contains the experimental artifacts for a hybrid policy-driven and multi-criteria MLOps decision framework.

## Objective

The framework evaluates candidate machine learning models and produces model promotion, rejection, retention, or retraining recommendations using model metrics, latency, model size, cost proxy, traceability score, drift reports, and generated YAML policy descriptors.

The current implementation is transparent and auditable. It is not a learned autonomous decision policy.

## Structure

```text
mlops-decision-framework/
├── data/
├── models/
├── outputs/
├── policies/
├── reports/
└── src/
```

## Generated Policy Files

- `policies/decision_policy.yaml`
- `policies/monitoring_policy.yaml`
- `policies/retraining_policy.yaml`
- `policies/lifecycle_descriptor.yaml`

## Generated JSON Outputs

- `outputs/environment.json`
- `outputs/dataset_metadata.json`
- `outputs/metrics.json`
- `outputs/drift_report.json`
- `outputs/promotion_decision.json`
- `outputs/retraining_recommendation.json`
- `outputs/baseline_results.json`
- `outputs/sensitivity_results.json`
- `outputs/ablation_results.json`
- `outputs/execution_time.json`

## Generated CSV Reports

- `reports/dataset_metadata.csv`
- `reports/model_metrics_table.csv`
- `reports/drift_report_table.csv`
- `reports/promotion_decision_table.csv`
- `reports/retraining_recommendation_table.csv`
- `reports/baseline_table.csv`
- `reports/baseline_summary_table.csv`
- `reports/sensitivity_table.csv`
- `reports/execution_time_table.csv`
- `reports/execution_time_summary_table.csv`

## Datasets

The experiment uses four datasets:

1. Wisconsin Breast Cancer.
2. Wine multiclass dataset.
3. Synthetic industrial/IoT-like dataset.
4. Synthetic time-series-like telemetry dataset.

## Drift Scenarios

The following scenarios are evaluated:

- stable,
- covariate drift,
- label shift,
- sudden drift,
- progressive drift,
- recurring drift.

## Baselines

The proposed gate is compared against:

- accuracy-only,
- F1-threshold,
- fixed retraining schedule,
- manual policy.

## Reproducibility

Run the Colab playbook from top to bottom. The environment and package versions are stored in `outputs/environment.json`.

## Scientific Limitation

Synthetic datasets and simulated drift do not replace real production monitoring logs. Future validation should integrate MLflow Model Registry or TFX and use real historical deployment decisions.
"""
with open(BASE_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)

requirements = f"""numpy=={np.__version__}
pandas=={pd.__version__}
scipy=={scipy.__version__}
scikit-learn=={sklearn.__version__}
PyYAML=={yaml.__version__}
joblib=={joblib.__version__}
"""
with open(BASE_DIR / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements)

for name in ["decision_gate.py", "drift.py", "baselines.py", "sensitivity.py", "ablation.py"]:
    with open(BASE_DIR / "src" / name, "w", encoding="utf-8") as f:
        f.write("# Source logic is implemented in the Colab playbook and exported as reproducible artifacts.\n")


# %%
# ============================================================
# 14. CREATE ZIP ARCHIVE
# ============================================================

zip_base = "/content/mlops-decision-framework"
zip_path = f"{zip_base}.zip"
if Path(zip_path).exists():
    Path(zip_path).unlink()

shutil.make_archive(zip_base, "zip", root_dir="/content", base_dir="mlops-decision-framework")


# %%
# ============================================================
# 15. DISPLAY FINAL TABLES
# ============================================================

print("\n==================== DATASETS ====================")
display(pd.DataFrame(dataset_metadata))

print("\n==================== MODEL METRICS ====================")
display(metrics_df[[
    "dataset", "model_name", "accuracy", "precision", "recall", "f1_score",
    "mean_latency_ms", "p95_latency_ms", "model_size_mb", "cost_proxy", "traceability_score"
]].sort_values(["dataset", "f1_score"], ascending=[True, False]))

print("\n==================== DRIFT REPORTS ====================")
display(drift_df.sort_values(["dataset", "scenario"]))

print("\n==================== DECISIONS ====================")
display(decision_df.sort_values(["dataset", "scenario"]))

print("\n==================== BASELINE SUMMARY ====================")
display(baseline_summary)

print("\n==================== EXECUTION TIME SUMMARY ====================")
display(execution_summary)

print("\n============================================================")
print("PLAYBOOK COMPLETED SUCCESSFULLY")
print("Project folder:", BASE_DIR)
print("ZIP archive:", zip_path)
print("To download in Colab, run:")
print("from google.colab import files")
print("files.download('/content/mlops-decision-framework.zip')")
print("============================================================")

# Optional automatic download in Colab.
try:
    from google.colab import files
    files.download("/content/mlops-decision-framework.zip")
except Exception:
    pass

