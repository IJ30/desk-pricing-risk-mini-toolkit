import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm

# Black-Scholes core

def d1gen(S, K, r, q, vol, T):
    return (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))

def d2gen(S, K, r, q, vol, T):
    return d1gen(S, K, r, q, vol, T) - vol * np.sqrt(T)

def bs_price(opt_type, S, K, r, q, vol, T):
    d1 = d1gen(S, K, r, q, vol, T)
    d2 = d2gen(S, K, r, q, vol, T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    if opt_type.lower() == "call":
        return disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
    elif opt_type.lower() == "put":
        return disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1)
    else:
        raise ValueError("type must be call/put")

def bs_greeks(opt_type, S, K, r, q, vol, T):
    d1 = d1gen(S, K, r, q, vol, T)
    d2 = d2gen(S, K, r, q, vol, T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    delta_call = disc_q * norm.cdf(d1)
    delta_put  = disc_q * (norm.cdf(d1) - 1.0)
    delta = delta_call if opt_type.lower() == "call" else delta_put

    gamma = disc_q * norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega  = disc_q * S * norm.pdf(d1) * np.sqrt(T)  # per 1.0 vol (not 1%)
    return delta, gamma, vega

def put_call_parity_check(S, K, r, q, vol, T):
    c = bs_price("call", S, K, r, q, vol, T)
    p = bs_price("put",  S, K, r, q, vol, T)
    lhs = c - p
    rhs = np.exp(-q*T)*S - np.exp(-r*T)*K
    return float(lhs - rhs)


# Validation / controls

def validate_row(row):
    issues = []
    for f in ["spot","strike","rate","div","vol","tau","qty"]:
        if pd.isna(row[f]):
            issues.append(f"missing_{f}")
    if row.get("spot", 0) <= 0: issues.append("spot_nonpositive")
    if row.get("strike", 0) <= 0: issues.append("strike_nonpositive")
    if row.get("vol", 0) <= 0: issues.append("vol_missing_or_nonpositive")
    if row.get("tau", 0) <= 0: issues.append("tau_nonpositive")
    if str(row.get("type","")).lower() not in ["call","put"]: issues.append("bad_type")
    return issues

def fd_greeks(opt_type, S, K, r, q, vol, T, eps_S=1e-3, eps_vol=1e-4):
    # Finite-diff delta/gamma on S; vega on vol
    p0 = bs_price(opt_type, S, K, r, q, vol, T)

    p_up = bs_price(opt_type, S*(1+eps_S), K, r, q, vol, T)
    p_dn = bs_price(opt_type, S*(1-eps_S), K, r, q, vol, T)
    delta = (p_up - p_dn) / (2 * S * eps_S)
    gamma = (p_up - 2*p0 + p_dn) / ((S*eps_S)**2)

    p_vup = bs_price(opt_type, S, K, r, q, vol+eps_vol, T)
    p_vdn = bs_price(opt_type, S, K, r, q, max(vol-eps_vol, 1e-12), T)
    vega = (p_vup - p_vdn) / (2*eps_vol)
    return float(delta), float(gamma), float(vega)

# Scenario runner

SCENARIOS = {
    "base":     {"dS": 0.00, "dVol": 0.00},
    "spot_up":  {"dS": 0.01, "dVol": 0.00},
    "spot_dn":  {"dS":-0.01, "dVol": 0.00},
    "vol_up":   {"dS": 0.00, "dVol": 0.01},  # +1 vol point
    "vol_dn":   {"dS": 0.00, "dVol":-0.01},
}

def run(input_csv="sample_portfolio.csv", out_csv="outputs/report.csv"):
    df = pd.read_csv(input_csv)

    rows = []
    for index, row in df.iterrows():
        issues = validate_row(row)
        base = {
            "id": row["id"],
            "type": row["type"],
            "spot": row["spot"],
            "strike": row["strike"],
            "rate": row["rate"],
            "div": row["div"],
            "vol": row["vol"],
            "tau": row["tau"],
            "qty": row["qty"],
            "issues": ";".join(issues) if issues else ""
        }

        if issues:
            rows.append({**base})
            continue

        S,K,r,q,vol,T = float(row["spot"]), float(row["strike"]), float(row["rate"]), float(row["div"]), float(row["vol"]), float(row["tau"])
        opt_type = str(row["type"]).lower()

        price = bs_price(opt_type, S,K,r,q,vol,T)
        delta, gamma, vega = bs_greeks(opt_type, S,K,r,q,vol,T)
        fd_delta, fd_gamma, fd_vega = fd_greeks(opt_type, S,K,r,q,vol,T)

        parity_err = put_call_parity_check(S,K,r,q,vol,T)

        rec = {
            **base,
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "fd_delta": fd_delta,
            "fd_gamma": fd_gamma,
            "fd_vega": fd_vega,
            "delta_diff": float(delta - fd_delta),
            "gamma_diff": float(gamma - fd_gamma),
            "vega_diff":  float(vega  - fd_vega),
            "parity_err": parity_err,
            "position_value": float(price * row["qty"]),
            "position_delta": float(delta * row["qty"]),
            "position_gamma": float(gamma * row["qty"]),
            "position_vega":  float(vega  * row["qty"]),
        }

        # scenarios
        for sc, shock in SCENARIOS.items():
            Ss = S * (1 + shock["dS"])
            vols = max(vol + shock["dVol"], 1e-12)
            rec[f"price_{sc}"] = float(bs_price(opt_type, Ss,K,r,q,vols,T))
        # scenario PnL vs base
        for sc in SCENARIOS:
            rec[f"pnl_{sc}"] = float((rec[f"price_{sc}"] - rec["price"]) * row["qty"])

        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    
    # Build a validation report (one row per issue)
    val_rows = []
    for index, r in out.iterrows():
        if isinstance(r.get("issues",""), str) and r["issues"].strip():
            for issue in r["issues"].split(";"):
                val_rows.append({
                    "id": r["id"],
                    "issue": issue,
                    "type": r.get("type",""),
                    "spot": r.get("spot", np.nan),
                    "strike": r.get("strike", np.nan),
                    "vol": r.get("vol", np.nan),
                    "tau": r.get("tau", np.nan),
                })

    validation_df = pd.DataFrame(val_rows)
    validation_df.to_csv("outputs/validation_report.csv", index=False)
    print("Wrote: outputs/validation_report.csv")


    ok = out[out["issues"] == ""]
    if len(ok) > 0:
        print("\nPortfolio (valid rows only):")
        print("Total Value:", ok["position_value"].sum())
        print("Total Delta:", ok["position_delta"].sum())
        print("Total Gamma:", ok["position_gamma"].sum())
        print("Total Vega :", ok["position_vega"].sum())
        print("\nTop breaks (if any):")
        breaks = out[out["issues"] != ""]
        if len(breaks) > 0:
            print(breaks[["id","issues"]].to_string(index=False))

    
os.makedirs("outputs", exist_ok=True)
run()
#make_png("outputs/report.csv")

