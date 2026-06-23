import pandas as pd

# Read all three files
part1 = pd.read_csv("hel1os_peaks_part1.csv")
part2 = pd.read_csv("hel1os_peaks_part2.csv")
part3 = pd.read_csv("hel1os_peaks_part3.csv")

# Merge
merged = pd.concat([part1, part2, part3], ignore_index=True)

# Convert and sort by time
merged["Time"] = pd.to_datetime(merged["Time"])
merged = merged.sort_values("Time").reset_index(drop=True)

# Save final file
merged.to_csv("HEL1OS_Peaks_June14.csv", index=False)

print(merged)
print("\nTotal Peaks:", len(merged))