import os
import pandas as pd
import matplotlib.pyplot as plt


report_csv="outputs/report.csv"
out_png = "outputs/report.png"
df = pd.read_csv(report_csv)
if "issues" in df.columns:
    df = df[df["issues"].fillna("") == ""].copy()

cols = [c for c in ["id","type","price","delta","gamma","vega","pnl_spot_up","pnl_vol_up"] if c in df.columns]
view = df[cols].head(8).copy()

# for c in view.columns:
#     if pd.api.types.is_numeric_dtype(view[c]):
#         view[c] = view[c].round(6)

fig, ax = plt.subplots(figsize=(12, 2 + 0.35 * len(view)))
ax.axis("off")

table = ax.table(
    cellText=view.values,
    colLabels=view.columns,
    cellLoc="center",
    loc="center"
)

plt.close(fig)

print(f"Saved PNG: {out_png}")


