import json

from summarize import batch_summary
import pandas as pd


df = pd.read_csv("../datasets/airports.csv", encoding="utf-8")
df = df.iloc[0:800]
summary = batch_summary(df.to_dict())

print(json.dumps(summary, indent=4))
print(df[df["state"] == "AL"].shape)
#
df_1 = df[df["name"].str.contains("Regional", case=False, na=False)]
print(df_1.shape, df_1.shape[0] / df.shape[0])
#
# filtered = df[
#     (df["latitude"] >= 30)
#     & (df["latitude"] <= 40)
#     & (df["longitude"] >= -95)
#     & (df["longitude"] <= -85)
# ]
#
# print(filtered.shape, filtered.shape[0] / df.shape[0])
#
# alaska = df[df["state"].str.strip().eq("AK")]
#
# alaska_sorted = alaska.sort_values(by="latitude", ascending=False)
#
# print(alaska_sorted.tail())
southeast_abbr = {"AL", "MS", "LA", "GA", "FL", "TN", "KY", "AR"}
southeast = df[df["state"].isin(southeast_abbr)].copy()

# summary
per_state = southeast.groupby("state").size().sort_index()
total = int(per_state.sum())

print("Per-state counts:")
print(per_state.to_string())
print(f"\nTotal airports in listed Southeast states: {total}")
