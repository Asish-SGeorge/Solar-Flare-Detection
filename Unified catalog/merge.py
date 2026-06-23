import pandas as pd
import glob

# ==========================================
# READ ALL UNIFIED CATALOGS
# ==========================================

files = glob.glob("Unified_Master_Flare_Catalog_*.csv")

catalogs = []

for file in files:

    df = pd.read_csv(file)

    # Skip empty catalogs (e.g., 16 June)
    if df.empty:
        continue

    # Add source file name
    df["Source_File"] = file

    catalogs.append(df)

# ==========================================
# MERGE ALL CATALOGS
# ==========================================

final_catalog = pd.concat(catalogs, ignore_index=True)

# ==========================================
# SORT CHRONOLOGICALLY
# ==========================================

final_catalog["SoLEXS_Time"] = pd.to_datetime(final_catalog["SoLEXS_Time"])
final_catalog["HEL1OS_Time"] = pd.to_datetime(final_catalog["HEL1OS_Time"])

final_catalog = final_catalog.sort_values("SoLEXS_Time")

# ==========================================
# RESET INDEX
# ==========================================

final_catalog.reset_index(drop=True, inplace=True)

# Remove old Event_ID column if present
if "Event_ID" in final_catalog.columns:
    final_catalog.drop(columns=["Event_ID"], inplace=True)

# Add Global Event ID
final_catalog.insert(
    0,
    "Global_Event_ID",
    range(1, len(final_catalog) + 1)
)

# ==========================================
# DISPLAY
# ==========================================

print("\n===== FINAL UNIFIED FLARE CATALOG =====\n")
print(final_catalog)

# ==========================================
# SAVE
# ==========================================

final_catalog.to_csv(
    "Unified_Flare_Catalog.csv",
    index=False
)

print("\nSaved as Unified_Flare_Catalog.csv")
print("Total Unified Events:", len(final_catalog))