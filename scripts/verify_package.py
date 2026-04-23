
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "paper" / "Network_Based_Latent_Heterogeneity.pdf",
    ROOT / "scripts" / "replication_pipeline.py",
    ROOT / "results" / "figures" / "Figure1_EmbeddingCharacterization.png",
    ROOT / "results" / "figures" / "Figure2_CoefficientComparison.png",
    ROOT / "results" / "figures" / "Figure3_HazardProfiles.png",
    ROOT / "results" / "figures" / "Figure4_BootstrapDistributions.png",
    ROOT / "results" / "tables" / "Table1_SummaryStatistics.csv",
    ROOT / "results" / "tables" / "Table2_AnnualExitRates.csv",
    ROOT / "results" / "tables" / "Table3_MarketHeterogeneity.csv",
    ROOT / "results" / "tables" / "Table4_BaselineHazard.csv",
    ROOT / "results" / "tables" / "Table5_EmbeddingHazard.csv",
    ROOT / "results" / "tables" / "Table6_ModelComparison.csv",
    ROOT / "results" / "tables" / "Table7_FixedEffectsBenchmark.csv",
    ROOT / "results" / "tables" / "Table8_Robustness.csv",
    ROOT / "results" / "tables" / "Table_BootstrapCoverage.csv",
    ROOT / "results" / "tables" / "Table_EmbeddingCorrelations.csv",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    print("Missing files:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

raw_data = ROOT / "data" / "raw" / "data_ryan_java.csv"
print("Package structure looks good.")
print(f"Raw data present: {'yes' if raw_data.exists() else 'no'}")
print("\nKey file checksums (SHA256, first 16 chars):")
for p in required[:6]:
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"{p.relative_to(ROOT)}  {h}")
