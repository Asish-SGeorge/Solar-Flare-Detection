import pandas as pd

# Read both files
part1 = pd.read_csv("hel1os_peaks_part1.csv")
part2 = pd.read_csv("hel1os_peaks_part2.csv")

# Merge
merged = pd.concat([part1, part2], ignore_index=True)

# Convert and sort by time
merged["Time"] = pd.to_datetime(merged["Time"])
merged = merged.sort_values("Time")

# Save final file
merged.to_csv("HEL1OS_Peaks_June12.csv", index=False)

print(merged)
print("\nTotal Peaks:", len(merged))