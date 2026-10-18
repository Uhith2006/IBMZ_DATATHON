"""
CarbonSched AI — single-file Flask web application
Frontend: embedded HTML + CSS + JavaScript
Backend: Python, pandas, NumPy, scikit-learn
Run: python app.py
Open: http://127.0.0.1:5000

Research note:
- Borg CSV fields are workload observations; environmental variables are synthetic.
- This is an academic simulator, not a production data-center controller.
- The scheduler ranks jobs and assigns simulated start/completion times across a
  small set of simulated servers. RL is represented by a fitted Q-iteration
  policy trained on simulator-generated logged transitions.
"""

from __future__ import annotations

import ast
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "borg_traces_data.csv"
SEED = 42
app = Flask(__name__)

# ---------------------------- Data + simulation -----------------------------

def scalar(value: Any, default=np.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]", "{}"}:
        return default
    try:
        return float(s)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            for key in ("cpus", "cpu", "memory", "mem"):
                if key in obj:
                    return scalar(obj[key], default)
            nums = [scalar(v, np.nan) for v in obj.values()]
            nums = [x for x in nums if np.isfinite(x)]
            return float(np.mean(nums)) if nums else default
        if isinstance(obj, (list, tuple)):
            nums = [scalar(v, np.nan) for v in obj]
            nums = [x for x in nums if np.isfinite(x)]
            return float(np.mean(nums)) if nums else default
    except Exception:
        pass
    return default


def _find_col(df, names):
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in mapping:
            return mapping[name.lower()]
    return None


def load_workload(csv_path=DEFAULT_CSV, limit=50000):
    """Load a bounded sample of Borg rows and turn observations into job proxies."""
    if not Path(csv_path).exists():
        return demo_workload(1500), "Demo workload (CSV not found)"
    pieces, read_rows = [], 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=10000, low_memory=False):
            # Deterministic subsample each chunk to keep memory modest.
            remaining = max(0, limit - read_rows)
            if remaining <= 0:
                break
            if len(chunk) > min(5000, remaining):
                chunk = chunk.sample(n=min(5000, remaining), random_state=SEED + len(pieces))
            pieces.append(chunk)
            read_rows += len(chunk)
            if read_rows >= limit:
                break
        raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    except Exception:
        raw = pd.read_csv(csv_path, nrows=limit, low_memory=False)
    if raw.empty:
        return demo_workload(1500), "Demo workload (CSV was empty)"

    cpu_col = _find_col(raw, ["average_usage", "cpu_usage", "cpu"])
    max_cpu_col = _find_col(raw, ["maximum_usage", "max_usage"])
    mem_col = _find_col(raw, ["assigned_memory", "memory_usage", "memory"])
    request_col = _find_col(raw, ["resource_request"])
    job_col = _find_col(raw, ["collection_id", "alloc_collection_id", "instance_index"])
    priority_col = _find_col(raw, ["priority", "scheduling_class"])
    start_col = _find_col(raw, ["start_time", "time"])
    end_col = _find_col(raw, ["end_time"])

    df = pd.DataFrame()
    df["job_id"] = raw[job_col].astype(str) if job_col else [f"borg_{i:06d}" for i in range(len(raw))]
    df["cpu_raw"] = raw[cpu_col].map(scalar) if cpu_col else np.nan
    if df["cpu_raw"].notna().sum() < len(df) * .1 and max_cpu_col:
        df["cpu_raw"] = df["cpu_raw"].fillna(raw[max_cpu_col].map(scalar))
    df["memory_raw"] = raw[mem_col].map(scalar) if mem_col else np.nan
    if request_col:
        req = raw[request_col].map(scalar)
        df["cpu_raw"] = df["cpu_raw"].fillna(req)
    df["priority"] = pd.to_numeric(raw[priority_col], errors="coerce") if priority_col else 0
    df["priority"] = df["priority"].fillna(df["priority"].median() if df["priority"].notna().any() else 0)

    if start_col:
        ts = pd.to_numeric(raw[start_col], errors="coerce")
        if ts.notna().mean() > .5 and ts.notna().any():
            med = ts.dropna().abs().median()
            unit = "us" if med > 1e14 else "ms" if med > 1e11 else "s"
            timestamps = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
            order = (timestamps - timestamps.min()).dt.total_seconds()
            df["arrival"] = order.fillna(pd.Series(np.arange(len(df))*30, index=df.index)).clip(lower=0)
        else:
            timestamps = pd.to_datetime(raw[start_col], errors="coerce", utc=True)
            order = (timestamps - timestamps.min()).dt.total_seconds()
            df["arrival"] = order.fillna(pd.Series(np.arange(len(df))*30, index=df.index)).clip(lower=0)
    else:
        df["arrival"] = np.arange(len(df), dtype=float) * 30

    if end_col and start_col:
        starts = pd.to_datetime(raw[start_col], errors="coerce", utc=True)
        ends = pd.to_datetime(raw[end_col], errors="coerce", utc=True)
        durations = (ends - starts).dt.total_seconds()
        df["duration"] = durations.where((durations > 0) & (durations < 86400))
    else:
        df["duration"] = np.nan

    # Select rows with any useful resource signal, then robustly impute.
    if df["cpu_raw"].notna().sum() > 0:
        df["cpu_raw"] = df["cpu_raw"].clip(lower=0, upper=df["cpu_raw"].quantile(.99))
    if df["memory_raw"].notna().sum() > 0:
        df["memory_raw"] = df["memory_raw"].clip(lower=0, upper=df["memory_raw"].quantile(.99))
    df["cpu_raw"] = df["cpu_raw"].fillna(df["cpu_raw"].median() if df["cpu_raw"].notna().any() else .4)
    df["memory_raw"] = df["memory_raw"].fillna(df["memory_raw"].median() if df["memory_raw"].notna().any() else .4)
    df["duration"] = df["duration"].fillna(df["duration"].median() if df["duration"].notna().any() else 180)
    df["duration"] = df["duration"].clip(10, 7200)
    for source, target in [("cpu_raw", "cpu"), ("memory_raw", "memory")]:
        arr = df[source].to_numpy(dtype=float)
        low, high = np.nanpercentile(arr, [1, 99])
        df[target] = np.clip((arr-low)/(high-low), 0, 1) if high > low else np.clip(arr, 0, 1)
    df["priority_norm"] = (df["priority"] - df["priority"].min()) / (df["priority"].max() - df["priority"].min() + 1e-9)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cpu", "memory"]).reset_index(drop=True)
    return df, f"Google Borg 2019 CSV sample ({len(df):,} workload observations)"


def demo_workload(n=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "job_id": [f"demo_{i:06d}" for i in range(n)],
        "cpu_raw": rng.beta(2, 3, n), "memory_raw": rng.beta(2, 3.2, n),
        "cpu": rng.beta(2, 3, n), "memory": rng.beta(2, 3.2, n),
        "priority": rng.integers(0, 10, n), "priority_norm": rng.random(n),
        "arrival": np.arange(n)*30, "duration": np.clip(rng.lognormal(4, .7, n), 10, 7200)
    })


def synthesize_environment(df, carbon_base=420, ambient=25, seed=SEED):
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    hour = (out["arrival"].to_numpy()/3600 + 8) % 24
    carbon_shape = .70 + .22*np.sin((hour-7)*2*np.pi/24) + .10*np.sin(hour*4*np.pi/24)
    out["carbon"] = np.clip(carbon_base*carbon_shape + rng.normal(0, carbon_base*.035, len(out)), 50, 1000)
    out["ambient"] = np.clip(ambient + 3.5*np.sin((hour-14)*2*np.pi/24) + rng.normal(0, 1, len(out)), 10, 45)
    out["it_power"] = 85 + 350*out["cpu"] + 45*out["memory"]
    out["cooling_power"] = out["it_power"]*(.22 + .012*np.maximum(out["ambient"]-20, 0))
    out["temperature"] = np.clip(out["ambient"] + 12 + 32*out["cpu"] + 7*out["memory"] + rng.normal(0, 1.7, len(out)), 20, 105)
    out["deadline_slack"] = rng.uniform(1.5, 5.0, len(out))*out["duration"]*(1.15-.45*out["priority_norm"])
    out["provenance"] = "SYNTHETIC_ENVIRONMENT"
    return out


def state(row, current_temp=None):
    return np.array([float(row["cpu"]), float(row["memory"]), float(row["carbon"])/1000,
                     ((float(current_temp if current_temp is not None else row["temperature"])-20)/80),
                     min(float(row["deadline_slack"])/7200, 1), 0.5], dtype=float)


class FQIPolicy:
    """Small fitted-Q-iteration policy over logged simulator transitions."""
    def __init__(self, seed=SEED):
        self.seed, self.models, self.scaler, self.ready = seed, [], StandardScaler(), False
        self.losses = []

    def fit(self, env, n=5000):
        rng = np.random.default_rng(self.seed)
        sample = env.iloc[rng.integers(0, len(env), n)].reset_index(drop=True)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i, row in sample.iterrows():
            s = state(row)
            # Behaviour policy is deliberately mixed to create a fixed logged dataset.
            carbon, temp, slack = row["carbon"], row["temperature"], row["deadline_slack"]
            a = int(rng.integers(0, 3)) if rng.random() < .25 else (1 if carbon > 520 and slack > 300 and temp < 75 else 0)
            delay = [0, 1, 2][a]
            next_temp = temp + (5*row["cpu"] + 1.3*row["memory"] if a == 0 else -1.6*delay)
            miss = float(slack < delay*120 + 60)
            energy = (row["it_power"] + row["cooling_power"])*max(1, delay+1)/1000/12
            carbon_cost = energy*carbon
            reward = -(carbon_cost/100 + max(0, next_temp-74)**2/100 + miss*2 + delay*.03)
            ns = s.copy()
            ns[2] = np.clip((carbon*(1+.1*np.sin(i/13)))/1000, 0, 1.5)
            ns[3] = np.clip((next_temp-20)/80, 0, 1.2)
            ns[4] = max(0, s[4] - .02*(delay+1))
            states.append(s); actions.append(a); rewards.append(reward); next_states.append(ns); dones.append(float(i % 32 == 31))
        S, A, R, NS, D = map(np.asarray, (states, actions, rewards, next_states, dones))
        self.scaler.fit(np.vstack([S, NS]))
        Sx, NSx = self.scaler.transform(S), self.scaler.transform(NS)
        self.models = [ExtraTreesRegressor(n_estimators=40, min_samples_leaf=3, random_state=self.seed+a, n_jobs=-1) for a in range(3)]
        for a in range(3):
            mask = A == a
            if mask.sum() < 2:
                mask = np.ones(len(A), dtype=bool)
            self.models[a].fit(Sx[mask], R[mask])
        for _ in range(6):
            qnext = np.column_stack([m.predict(NSx) for m in self.models])
            target = R + .94*(1-D)*qnext.max(axis=1)
            errors = []
            for a in range(3):
                mask = A == a
                if mask.sum() > 1:
                    self.models[a].fit(Sx[mask], target[mask])
                    errors.append(np.mean((self.models[a].predict(Sx[mask])-target[mask])**2))
            self.losses.append(float(np.mean(errors)))
        self.ready = True
        return len(S)

    def q(self, s):
        x = self.scaler.transform(np.asarray(s).reshape(1, -1))
        return np.array([m.predict(x)[0] for m in self.models])


POLICY = FQIPolicy()
ENV_CACHE = None
SOURCE_LABEL = "Not loaded"


def get_environment():
    global ENV_CACHE, SOURCE_LABEL
    if ENV_CACHE is None:
        workload, SOURCE_LABEL = load_workload()
        ENV_CACHE = synthesize_environment(workload)
    return ENV_CACHE


def make_schedule(payload):
    env = get_environment()
    cfg = {
        "carbon_weight": float(payload.get("carbonWeight", 45))/100,
        "thermal_limit": float(payload.get("thermalLimit", 82)),
        "server_count": int(payload.get("servers", 3)),
        "carbon_base": float(payload.get("carbonBase", 420)),
        "ambient": float(payload.get("ambient", 25)),
        "use_rl": bool(payload.get("useRL", True)),
        "job_count": int(payload.get("jobCount", 10)),
        "seed": int(payload.get("seed", 42)),
    }
    # Rebuild environmental conditions for chosen scenario settings.
    workload, source = load_workload(limit=50000)
    scenario_env = synthesize_environment(workload, cfg["carbon_base"], cfg["ambient"], cfg["seed"])
    SOURCE_LABEL = source
    rng = np.random.default_rng(cfg["seed"])
    count = max(1, min(30, cfg["job_count"]))
    idx = rng.choice(len(scenario_env), size=count, replace=len(scenario_env)<count)
    jobs = scenario_env.iloc[idx].copy().reset_index(drop=True)
    jobs["job_id"] = [f"JOB-{i+1:03d}" for i in range(count)]
    jobs["duration_min"] = np.clip(jobs["duration"].to_numpy()/60, .5, 120)
    # For user-friendly scheduling, derive demand fractions, and set deadlines from service duration.
    jobs["deadline_min"] = jobs["duration_min"] + np.clip(jobs["deadline_slack"].to_numpy()/60, 1, 180)
    jobs["deadline_slack_min"] = jobs["deadline_min"] - jobs["duration_min"]

    server_count = cfg["server_count"]
    servers = [{"id": f"NODE-{i+1}", "available": 0.0, "temp": float(cfg["ambient"]+18),
                "cpu": 0.0, "memory": 0.0, "energy": 0.0, "carbon": 0.0} for i in range(server_count)]
    # Sort by urgency/priority; policy selects among run/defer candidates, then assign to best server.
    jobs = jobs.sort_values(["deadline_slack_min", "priority_norm"], ascending=[True, False]).reset_index(drop=True)
    output = []
    for _, job in jobs.iterrows():
        candidates = []
        for server in servers:
            start = server["available"]
            cpu_pressure = server["cpu"] + float(job["cpu"])
            mem_pressure = server["memory"] + float(job["memory"])
            start_temp = server["temp"]
            s = np.array([job["cpu"], job["memory"], job["carbon"]/1000,
                          (start_temp-20)/80, min(job["deadline_slack_min"]/180, 1),
                          min(start/240, 1)], dtype=float)
            if cfg["use_rl"] and POLICY.ready:
                q = POLICY.q(s)
                action = int(np.argmax(q))
            else:
                # Transparent carbon/thermal heuristic fallback.
                action = 1 if job["carbon"] > 520 and job["deadline_slack_min"] > 10 and start_temp < cfg["thermal_limit"]-8 else 0
                q = np.array([np.nan, np.nan, np.nan])
            delays = [0, 5, 10]
            proposed_start = start + delays[action]
            proposed_temp = start_temp + (5*job["cpu"] + 1.3*job["memory"] if action == 0 else -1.6*delays[action]/5)
            safe = (cpu_pressure <= 1.65 and mem_pressure <= 1.75 and proposed_temp < cfg["thermal_limit"])
            if not safe:
                # Shield picks an immediately feasible candidate if available, else the earliest cooling option.
                action = 1 if start_temp >= cfg["thermal_limit"]-5 else 0
                proposed_start = start + delays[action]
                proposed_temp = start_temp - (1.6 if action else 0) + (5*job["cpu"]+1.3*job["memory"] if action == 0 else 0)
            duration = float(job["duration_min"])
            finish = proposed_start + duration
            deadline = float(job["deadline_min"])
            # Simplified power/energy estimate; one simulation minute per schedule minute.
            power_w = float(job["it_power"] + job["cooling_power"])
            energy_kwh = power_w * duration / 60 / 1000
            carbon_g = energy_kwh * float(job["carbon"])
            temp_after = float(np.clip(proposed_temp + 1.8*job["cpu"]*duration/10 - .5, cfg["ambient"], 105))
            miss = finish > deadline
            override = not safe
            server["available"] = finish
            server["temp"] = temp_after
            server["cpu"] = max(0, cpu_pressure*.55)
            server["memory"] = max(0, mem_pressure*.55)
            server["energy"] += energy_kwh
            server["carbon"] += carbon_g
            output.append({
                "job_id": job["job_id"], "server": server["id"], "cpu": round(float(job["cpu"]), 3),
                "memory": round(float(job["memory"]), 3), "carbon": round(float(job["carbon"]), 1),
                "temp_before": round(start_temp, 1), "temp_after": round(temp_after, 1),
                "start_min": round(proposed_start, 1), "finish_min": round(finish, 1),
                "deadline_min": round(deadline, 1), "deadline_miss": bool(miss),
                "duration_min": round(duration, 1), "energy_kwh": round(energy_kwh, 4),
                "carbon_g": round(carbon_g, 2), "action": ["RUN_NOW", "DEFER_1", "DEFER_2"][action],
                "shield_override": bool(override), "q_run": None if not np.isfinite(q[0]) else round(float(q[0]), 4),
                "q_defer1": None if not np.isfinite(q[1]) else round(float(q[1]), 4),
                "q_defer2": None if not np.isfinite(q[2]) else round(float(q[2]), 4),
            })
    result = pd.DataFrame(output)
    summary = {
        "jobs": len(result),
        "servers": server_count,
        "total_energy": round(float(result["energy_kwh"].sum()), 4),
        "total_carbon": round(float(result["carbon_g"].sum()), 2),
        "deadline_misses": int(result["deadline_miss"].sum()),
        "thermal_flags": int((result["temp_after"] >= cfg["thermal_limit"]).sum()),
        "shield_overrides": int(result["shield_override"].sum()),
        "avg_finish": round(float(result["finish_min"].mean()), 1),
        "source": source,
        "policy": "Offline FQI" if cfg["use_rl"] and POLICY.ready else "Carbon/thermal heuristic",
        "synthetic": True,
    }
    return {"summary": summary, "jobs": result.to_dict(orient="records"),
            "servers": [{"id": s["id"], "energy": round(s["energy"], 4), "carbon": round(s["carbon"], 2),
                         "temp": round(s["temp"], 1), "available": round(s["available"], 1)} for s in servers]}


# -------------------------------- Web UI --------------------------------------
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CarbonSched AI</title>
<style>
:root{--bg:#f5f7fb;--ink:#14213d;--muted:#65738b;--card:#fff;--line:#e4eaf2;--green:#087f5b;--green2:#0ba879;--blue:#4263eb;--orange:#f59f00;--red:#e03131;--shadow:0 14px 38px rgba(24,39,75,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,select{font:inherit}.app{display:grid;grid-template-columns:255px minmax(0,1fr);min-height:100vh}
.sidebar{background:#101c31;color:#edf3ff;padding:26px 18px;position:sticky;top:0;height:100vh;overflow:auto}
.brand{display:flex;align-items:center;gap:11px;font-weight:850;font-size:19px;letter-spacing:-.5px}.logo{width:40px;height:40px;border-radius:14px;background:linear-gradient(135deg,#2dd4a0,#7ce7c3);display:grid;place-items:center;color:#063b30;font-size:22px}
.side-copy{font-size:12px;color:#9fb0cb;line-height:1.6;margin:12px 0 26px}.nav{display:grid;gap:7px}.nav button{border:0;color:#b9c7df;background:transparent;text-align:left;padding:12px;border-radius:12px;cursor:pointer;font-weight:650}.nav button.active,.nav button:hover{background:#203654;color:#fff}.side-bottom{margin-top:32px;padding:14px;border:1px solid #2a3c58;border-radius:14px;color:#c3d0e6;font-size:12px;line-height:1.6}
main{padding:28px clamp(18px,3vw,48px) 50px;max-width:1600px;width:100%;margin:auto}.topline{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:22px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:var(--green);font-weight:800}.h1{font-size:clamp(27px,3vw,40px);letter-spacing:-1.5px;line-height:1.1;margin:7px 0 9px}.subtitle{color:var(--muted);max-width:780px;line-height:1.65}.pill{border-radius:999px;padding:8px 12px;font-size:12px;font-weight:800;background:#e3f8ef;color:#087f5b;white-space:nowrap}
.hero{background:linear-gradient(115deg,#123d37,#174a55 55%,#304d88);color:white;border-radius:24px;padding:27px;display:flex;align-items:center;justify-content:space-between;gap:25px;box-shadow:var(--shadow);margin-bottom:22px;overflow:hidden;position:relative}.hero h2{font-size:clamp(22px,2.5vw,31px);letter-spacing:-.8px;margin:0 0 9px}.hero p{margin:0;color:#d3eee8;line-height:1.6;max-width:650px}.hero-art{width:150px;height:130px;flex:none;display:grid;place-items:center;font-size:70px;border-radius:36px;background:rgba(255,255,255,.09);transform:rotate(4deg)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin:18px 0}.metric,.card{background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}.metric{padding:18px}.metric-label{font-size:12px;color:var(--muted);font-weight:700}.metric-value{font-size:27px;font-weight:850;letter-spacing:-.8px;margin:8px 0 5px}.metric-foot{font-size:11px;color:var(--muted)}.card{padding:21px;margin-bottom:18px}.card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:15px}.card h3{margin:0;font-size:17px;letter-spacing:-.3px}.card p{color:var(--muted);font-size:13px;line-height:1.65;margin:8px 0 0}.section{display:none}.section.active{display:block;animation:fade .22s ease}@keyframes fade{from{opacity:.5;transform:translateY(4px)}to{opacity:1;transform:none}}
.controls{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.field label{display:flex;justify-content:space-between;gap:10px;font-size:12px;font-weight:750;margin-bottom:8px}.field output{color:var(--green)}input[type=range]{width:100%;accent-color:var(--green)}input[type=number],select{width:100%;border:1px solid #d9e2ef;border-radius:10px;padding:10px;background:white;color:var(--ink)}.btn{border:0;background:var(--green);color:white;padding:12px 17px;border-radius:12px;font-weight:800;cursor:pointer;box-shadow:0 7px 15px #087f5b25;transition:.18s}.btn:hover{transform:translateY(-1px);background:#06694c}.btn.secondary{background:#edf2f8;color:#24344e;box-shadow:none}.btn.outline{background:white;color:var(--green);border:1px solid #b8e5d4;box-shadow:none}.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}.note{padding:13px 15px;background:#effaf5;border:1px solid #c8efdf;border-radius:12px;color:#235c4b;font-size:12px;line-height:1.6;margin:14px 0}.warning{background:#fff8e6;border-color:#ffe3a3;color:#795700}
.steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.step{padding:15px;border:1px solid var(--line);border-radius:14px;background:#fbfcff}.step-num{width:28px;height:28px;border-radius:9px;background:#e1f7ee;color:var(--green);display:grid;place-items:center;font-weight:900;font-size:12px}.step h4{font-size:13px;margin:12px 0 5px}.step p{font-size:12px;color:var(--muted);line-height:1.5;margin:0}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:13px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th{text-align:left;background:#f3f7fb;color:#63738b;font-size:10px;text-transform:uppercase;letter-spacing:.6px;padding:12px}td{padding:12px;border-top:1px solid #edf1f6}tbody tr:hover{background:#f8fbff}.status{padding:5px 8px;border-radius:999px;font-size:10px;font-weight:800;background:#e2f8ee;color:#087f5b}.status.warn{background:#fff0d9;color:#9c5d00}.status.bad{background:#ffe7e7;color:#b42323}
.barlist{display:grid;gap:13px}.barrow{display:grid;grid-template-columns:85px 1fr 70px;align-items:center;gap:12px;font-size:12px}.bar{height:9px;border-radius:99px;background:#edf1f6;overflow:hidden}.bar span{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#20b486,#4263eb)}.barrow b{text-align:right}
.toast{display:none;padding:12px 14px;border-radius:12px;margin:12px 0;font-size:13px}.toast.show{display:block}.toast.ok{background:#e3f8ef;color:#087f5b}.toast.err{background:#fff0f0;color:#b42323}.spinner{display:none;color:var(--muted);font-size:13px;margin-top:12px}.spinner.show{display:block}
.footer{color:#8491a5;font-size:11px;text-align:center;padding:25px}.two-col{display:grid;grid-template-columns:1.1fr .9fr;gap:18px}.mini-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.mini{padding:13px;background:#f7f9fc;border-radius:12px}.mini strong{display:block;font-size:20px;margin-top:4px}.mini span{font-size:11px;color:var(--muted)}
@media(max-width:1050px){.app{grid-template-columns:210px 1fr}.grid{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:repeat(2,1fr)}.steps{grid-template-columns:repeat(2,1fr)}.hero-art{display:none}}
@media(max-width:700px){.app{display:block}.sidebar{position:relative;height:auto;padding:15px}.side-copy,.side-bottom{display:none}.nav{grid-template-columns:repeat(3,1fr);margin-top:15px}.nav button{font-size:11px;text-align:center;padding:9px 5px}.grid{grid-template-columns:repeat(2,1fr)}main{padding:18px 13px}.controls,.steps,.two-col{grid-template-columns:1fr}.hero{padding:21px}.metric-value{font-size:22px}.topline{align-items:flex-start;flex-direction:column}.mini-cards{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="app">
<aside class="sidebar">
  <div class="brand"><div class="logo">🌱</div><div>CarbonSched<span style="color:#58d9b0"> AI</span></div></div>
  <div class="side-copy">A smarter scheduling lab for lower-carbon, thermally-aware computing.</div>
  <nav class="nav">
    <button class="active" data-page="home">⌂ &nbsp;Overview</button>
    <button data-page="scheduler">⚡ &nbsp;Job scheduler</button>
    <button data-page="results">▥ &nbsp;Results</button>
    <button data-page="model">◈ &nbsp;RL model</button>
    <button data-page="about">ⓘ &nbsp;How it works</button>
  </nav>
  <div class="side-bottom"><b>SIMULATION MODE</b><br>Carbon, heat and cooling values are synthetic estimates. No real data-center equipment is controlled.</div>
</aside>
<main>
  <div class="topline"><div><div class="eyebrow">Intelligent infrastructure · Research preview</div><h1 class="h1" id="pageTitle">Make every compute cycle count.</h1><div class="subtitle" id="pageSubtitle">Explore how workload-aware scheduling can balance carbon emissions, energy, deadlines and thermal limits.</div></div><div class="pill">● System ready</div></div>

  <section class="section active" id="home">
    <div class="hero"><div><h2>Schedule smarter.<br>Waste less energy.</h2><p>Turn workload traces into an interactive scheduling experiment. Test a batch of jobs, compare policy choices and inspect the trade-offs.</p><div class="actions"><button class="btn" onclick="go('scheduler')">Run a scheduling test →</button><button class="btn secondary" onclick="go('about')">Explore the method</button></div></div><div class="hero-art">🌍</div></div>
    <div class="grid">
      <div class="metric"><div class="metric-label">WORKLOAD SOURCE</div><div class="metric-value" style="font-size:19px" id="sourceMetric">Loading…</div><div class="metric-foot">Real trace or demo fallback</div></div>
      <div class="metric"><div class="metric-label">JOBS IN LAST RUN</div><div class="metric-value" id="jobMetric">—</div><div class="metric-foot">Batch schedule size</div></div>
      <div class="metric"><div class="metric-label">ESTIMATED CARBON</div><div class="metric-value" id="carbonMetric">—</div><div class="metric-foot">gCO₂e · simulated</div></div>
      <div class="metric"><div class="metric-label">DEADLINE MISSES</div><div class="metric-value" id="missMetric">—</div><div class="metric-foot">Lower is better</div></div>
    </div>
    <div class="card"><div class="card-head"><div><h3>Your experiment in four steps</h3><p>No ML jargon required. Start with jobs, run the policy, then compare its results.</p></div><span class="pill">Guided workflow</span></div>
      <div class="steps">
        <div class="step"><div class="step-num">01</div><h4>Choose workload</h4><p>Use the Google Borg 2019 sample or a generated demo workload.</p></div>
        <div class="step"><div class="step-num">02</div><h4>Set conditions</h4><p>Adjust carbon, ambient temperature, server count and job batch size.</p></div>
        <div class="step"><div class="step-num">03</div><h4>Build schedule</h4><p>Assign jobs to simulated nodes with start times and deadlines.</p></div>
        <div class="step"><div class="step-num">04</div><h4>Compare outcomes</h4><p>Review energy, emissions, heat, missed deadlines and shield actions.</p></div>
      </div>
    </div>
    <div class="note">Research transparency: the Borg trace provides workload information. Carbon intensity, server temperature, cooling power and deadlines are added as synthetic variables and are not historical Google measurements.</div>
  </section>

  <section class="section" id="scheduler">
    <div class="card"><div class="card-head"><div><h3>Build a job schedule</h3><p>Choose your scenario. The scheduler will create a complete simulated schedule for the selected batch.</p></div><span class="pill">Interactive lab</span></div>
      <div class="controls">
        <div class="field"><label>Jobs in this batch <output id="jobCountOut">10</output></label><input id="jobCount" type="range" min="5" max="30" value="10" oninput="jobCountOut.value=this.value"></div>
        <div class="field"><label>Simulated servers <output id="serversOut">3</output></label><input id="servers" type="range" min="1" max="8" value="3" oninput="serversOut.value=this.value"></div>
        <div class="field"><label>Grid carbon baseline <output id="carbonBaseOut">420</output></label><input id="carbonBase" type="range" min="100" max="800" step="10" value="420" oninput="carbonBaseOut.value=this.value+' g/kWh'"></div>
        <div class="field"><label>Ambient temperature <output id="ambientOut">25 °C</output></label><input id="ambient" type="range" min="15" max="40" value="25" oninput="ambientOut.value=this.value+' °C'"></div>
        <div class="field"><label>Thermal limit <output id="thermalLimitOut">82 °C</output></label><input id="thermalLimit" type="range" min="65" max="95" value="82" oninput="thermalLimitOut.value=this.value+' °C'"></div>
        <div class="field"><label>Carbon importance <output id="carbonWeightOut">45%</output></label><input id="carbonWeight" type="range" min="0" max="100" value="45" oninput="carbonWeightOut.value=this.value+'%'"></div>
        <div class="field"><label>Scheduling policy</label><select id="policy"><option value="rl">Offline RL (train first)</option><option value="heuristic">Carbon + thermal heuristic</option></select></div>
        <div class="field"><label>Reproducibility seed</label><input id="seed" type="number" value="42" min="0" max="999999"></div>
      </div>
      <div class="note">The model is a research simulator. A safety check can override an action when the simplified modeled limits are exceeded.</div>
      <div class="actions"><button class="btn" onclick="runSchedule()">⚡ Run scheduling simulation</button><button class="btn secondary" onclick="resetControls()">Reset scenario</button></div>
      <div class="spinner" id="scheduleSpinner">Simulating jobs and estimating outcomes…</div><div class="toast" id="scheduleToast"></div>
    </div>
    <div class="card"><div class="card-head"><div><h3>Schedule output</h3><p>Once the simulation runs, each job appears with its assigned node, start time, completion time and estimated impact.</p></div></div><div id="schedulePlaceholder" class="note">Your schedule will appear here. Start with 10 jobs and 3 simulated servers.</div><div id="scheduleTable" class="table-wrap" style="display:none"></div></div>
  </section>

  <section class="section" id="results">
    <div class="grid">
      <div class="metric"><div class="metric-label">TOTAL ESTIMATED ENERGY</div><div class="metric-value" id="resEnergy">—</div><div class="metric-foot">kWh, IT + cooling proxy</div></div>
      <div class="metric"><div class="metric-label">TOTAL ESTIMATED CARBON</div><div class="metric-value" id="resCarbon">—</div><div class="metric-foot">gCO₂e, synthetic grid scenario</div></div>
      <div class="metric"><div class="metric-label">THERMAL FLAGS</div><div class="metric-value" id="resThermal">—</div><div class="metric-foot">Jobs at/above selected limit</div></div>
      <div class="metric"><div class="metric-label">SHIELD OVERRIDES</div><div class="metric-value" id="resShield">—</div><div class="metric-foot">Actions changed by modeled shield</div></div>
    </div>
    <div class="two-col">
      <div class="card"><h3>Energy by simulated server</h3><p>Compare the workload assigned to each node.</p><div id="serverBars" class="barlist" style="margin-top:18px"><div class="note">Run a schedule to see results.</div></div></div>
      <div class="card"><h3>What do these numbers mean?</h3><p>Carbon is estimated as energy multiplied by synthetic grid carbon intensity. Thermal flags are based on the simplified temperature proxy. These are comparative simulation metrics, not facility measurements.</p><div class="note warning">A lower carbon estimate is not automatically better if it causes excessive deadline misses or violates thermal constraints. Evaluate all metrics together.</div></div>
    </div>
    <div class="card"><h3>Full result table</h3><p>Download the run to inspect it outside the dashboard.</p><div id="resultsTable" class="table-wrap"><div class="note">Run a schedule first.</div></div><div class="actions"><button class="btn outline" onclick="downloadResults()">↓ Download results CSV</button></div></div>
  </section>

  <section class="section" id="model">
    <div class="card"><div class="card-head"><div><h3>Offline reinforcement-learning lab</h3><p>Train a fitted Q-iteration policy on a fixed set of simulator-generated transitions.</p></div><span class="pill">FQI · 3 actions</span></div>
      <div class="note warning">This is offline RL trained on a synthetic simulator replay buffer. It is not proof of real-world performance, and the simulator should be validated before making research claims.</div>
      <div class="mini-cards"><div class="mini"><span>State inputs</span><strong>6</strong><span>CPU, memory, carbon, heat, slack, time</span></div><div class="mini"><span>Actions</span><strong>3</strong><span>Run now, defer 1, defer 2</span></div><div class="mini"><span>Training method</span><strong>FQI</strong><span>ExtraTrees Q-value regressors</span></div></div>
      <div class="actions"><button class="btn" id="trainBtn" onclick="trainModel()">Train offline RL policy</button></div><div class="spinner" id="trainSpinner">Building replay data and fitting Q-functions… this may take a little while.</div><div class="toast" id="trainToast"></div><div id="trainOutput" style="margin-top:15px"></div>
    </div>
    <div class="card"><h3>What the policy learns</h3><p>For each simulated state, the policy estimates the long-term value of each action. The scheduler then considers the action, modeled resource/thermal limits and the job's placement on available nodes. The policy is not guaranteed to be optimal; it must be evaluated against baselines on unseen scenarios.</p></div>
  </section>

  <section class="section" id="about">
    <div class="card"><h3>How CarbonSched AI works</h3><p>The project combines a real workload trace with a synthetic data-center environment, then evaluates scheduling decisions in a controlled simulator.</p>
      <div class="steps" style="margin-top:18px">
        <div class="step"><div class="step-num">01</div><h4>Real workload</h4><p>CPU/memory proxies, duration and priority are extracted from Borg CSV when available.</p></div>
        <div class="step"><div class="step-num">02</div><h4>Synthetic context</h4><p>Carbon, ambient temperature, IT/cooling power, server temperature and deadlines are generated by explicit assumptions.</p></div>
        <div class="step"><div class="step-num">03</div><h4>Offline RL</h4><p>FQI learns from a fixed logged simulator dataset; it does not control live infrastructure.</p></div>
        <div class="step"><div class="step-num">04</div><h4>Schedule + evaluate</h4><p>Jobs are assigned to simulated servers; outcomes are compared across energy, carbon, heat and deadlines.</p></div>
      </div>
    </div>
    <div class="card"><h3>Research boundaries</h3><p>1. Synthetic environmental values are not real historical measurements. 2. The power/thermal model is simplified and needs calibration. 3. The current safety shield enforces only modeled constraints. 4. Borg data does not provide counterfactual outcomes for actions not historically taken. 5. Train/test splits should be done by time or scenario, not random row mixing, to avoid leakage. 6. A credible claim of efficiency requires comparisons against baselines over multiple unseen scenarios and random seeds.</p></div>
  </section>
  <div class="footer">CarbonSched AI · Research prototype · Synthetic environment · No production infrastructure is controlled</div>
</main>
</div>
<script>
let lastResult=null;
const titles={home:["Make every compute cycle count.","Explore how workload-aware scheduling can balance carbon emissions, energy, deadlines and thermal limits."],scheduler:["Build a better schedule.","Choose a workload batch and environmental conditions. The simulator will assign jobs to nodes and estimate their outcomes."],results:["See the trade-offs.","Review energy, carbon, thermal flags and deadline misses together, not as isolated scores."],model:["Teach the policy to choose.","Train an offline fitted Q-iteration model using a fixed simulator replay buffer."],about:["Transparent by design.","Understand which inputs are real workload data, which are synthetic assumptions, and how the simulation makes decisions."]};
function go(page){document.querySelectorAll('.section').forEach(s=>s.classList.toggle('active',s.id===page));document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===page));document.getElementById('pageTitle').textContent=titles[page][0];document.getElementById('pageSubtitle').textContent=titles[page][1];window.scrollTo({top:0,behavior:'smooth'});}
document.querySelectorAll('.nav button').forEach(b=>b.addEventListener('click',()=>go(b.dataset.page)));
function toast(id,msg,ok=true){const el=document.getElementById(id);el.textContent=msg;el.className='toast show '+(ok?'ok':'err');}
function resetControls(){[['jobCount',10],['servers',3],['carbonBase',420],['ambient',25],['thermalLimit',82],['carbonWeight',45],['seed',42]].forEach(([id,v])=>document.getElementById(id).value=v);['jobCount','servers','carbonBase','ambient','thermalLimit','carbonWeight'].forEach(id=>{const out=document.getElementById(id+'Out');if(out)out.textContent=document.getElementById(id).value+(id==='carbonBase'?' g/kWh':(['ambient','thermalLimit'].includes(id)?' °C':id==='carbonWeight'?'%':''));});}
async function trainModel(){const btn=document.getElementById('trainBtn'),sp=document.getElementById('trainSpinner');btn.disabled=true;sp.classList.add('show');try{const r=await fetch('/api/train',{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.error||'Training failed');toast('trainToast',`Training complete. ${d.transitions.toLocaleString()} logged transitions; ${d.iterations} FQI iterations.`);document.getElementById('trainOutput').innerHTML=`<div class="mini-cards"><div class="mini"><span>Logged transitions</span><strong>${d.transitions.toLocaleString()}</strong></div><div class="mini"><span>FQI iterations</span><strong>${d.iterations}</strong></div><div class="mini"><span>Final fitted-target MSE</span><strong>${Number(d.finalLoss).toFixed(4)}</strong></div></div><p style="color:var(--muted);font-size:12px">The model is ready for simulated scheduling. Lower training error alone does not prove better scheduling quality.</p>`;}catch(e){toast('trainToast',e.message,false)}finally{btn.disabled=false;sp.classList.remove('show')}}
async function runSchedule(){const sp=document.getElementById('scheduleSpinner'),btn=document.querySelector('#scheduler .btn');sp.classList.add('show');document.getElementById('scheduleToast').className='toast';const payload={jobCount:+jobCount.value,servers:+servers.value,carbonBase:+carbonBase.value,ambient:+ambient.value,thermalLimit:+thermalLimit.value,carbonWeight:+carbonWeight.value,useRL:policy.value==='rl',seed:+seed.value};try{const r=await fetch('/api/schedule',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw new Error(d.error||'Scheduling failed');lastResult=d;renderResult(d);toast('scheduleToast',`Schedule created for ${d.summary.jobs} jobs across ${d.summary.servers} simulated nodes.`);document.getElementById('schedulePlaceholder').style.display='none';document.getElementById('scheduleTable').style.display='block';document.getElementById('scheduleTable').innerHTML=tableHTML(d.jobs);go('scheduler');}catch(e){toast('scheduleToast',e.message,false)}finally{sp.classList.remove('show')}}
function tableHTML(rows){if(!rows.length)return '<div class="note">No jobs returned.</div>';const cols=[['job_id','Job'],['server','Server'],['action','Action'],['start_min','Start (min)'],['finish_min','Finish (min)'],['deadline_min','Deadline (min)'],['carbon_g','Carbon (g)'],['temp_after','Temp (°C)'],['deadline_miss','Deadline'],['shield_override','Shield']];return '<table><thead><tr>'+cols.map(c=>`<th>${c[1]}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+cols.map(([k,label])=>{let v=row[k];if(k==='deadline_miss'||k==='shield_override'){v=v?'Flag':'OK';return `<td><span class="status ${v==='Flag'?'warn':''}">${v}</span></td>`}return `<td>${v??'—'}</td>`}).join('')+'</tr>').join('')+'</tbody></table>'}
function renderResult(d){const s=d.summary;document.getElementById('jobMetric').textContent=s.jobs;document.getElementById('carbonMetric').textContent=s.total_carbon.toLocaleString();document.getElementById('missMetric').textContent=s.deadline_misses;document.getElementById('sourceMetric').textContent=s.source.includes('Google')?'Borg 2019':'Demo fallback';document.getElementById('resEnergy').textContent=s.total_energy;document.getElementById('resCarbon').textContent=s.total_carbon.toLocaleString();document.getElementById('resThermal').textContent=s.thermal_flags;document.getElementById('resShield').textContent=s.shield_overrides;document.getElementById('resultsTable').innerHTML=tableHTML(d.jobs);const max=Math.max(1,...d.servers.map(x=>x.energy));document.getElementById('serverBars').innerHTML=d.servers.map(x=>`<div class="barrow"><span>${x.id}</span><div class="bar"><span style="width:${100*x.energy/max}%"></span></div><b>${x.energy.toFixed(3)} kWh</b></div>`).join('');}
function downloadResults(){if(!lastResult){alert('Run a scheduling simulation first.');return}const rows=lastResult.jobs,cols=Object.keys(rows[0]||{});const csv=[cols.join(','),...rows.map(r=>cols.map(c=>JSON.stringify(r[c]??'')).join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='carbonsched_schedule.csv';a.click();URL.revokeObjectURL(a.href)}
fetch('/api/status').then(r=>r.json()).then(d=>{document.getElementById('sourceMetric').textContent=d.source.includes('Google')?'Borg 2019':'Demo fallback';}).catch(()=>{});
</script>
</body></html>
"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/api/status")
def status():
    env = get_environment()
    return jsonify({"source": SOURCE_LABEL, "rows": len(env), "rl_trained": POLICY.ready})


@app.post("/api/train")
def train():
    try:
        env = get_environment()
        transitions = POLICY.fit(env, n=5000)
        return jsonify({"transitions": transitions, "iterations": len(POLICY.losses),
                        "finalLoss": POLICY.losses[-1] if POLICY.losses else 0})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/schedule")
def schedule():
    try:
        payload = request.get_json(silent=True) or {}
        result = make_schedule(payload)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("\nCarbonSched AI is starting.")
    print(f"CSV expected at: {DEFAULT_CSV}")
    print("Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
