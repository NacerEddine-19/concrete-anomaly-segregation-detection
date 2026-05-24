# ─── Class labels (must match training order) ─────────────────────────────────
CLASSES      = ['crack', 'crack_segregation', 'segregation', 'normal']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

SEG_IMG_SIZE = 256       # U-Net input size
CLS_IMG_SIZE = 224       # ResNet-18 input size
GAI_MIN_THRESHOLD = 0.05 # SI_ia is unreliable below this GAI level

# ─── Defect cluster → damage label ────────────────────────────────────────────
DAMAGE_LABELS = {0: "Low damage", 1: "Medium damage", 2: "High damage"}
DAMAGE_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}   # green / orange / red

# ─── Material cluster means (cluster 0 = strongest → cluster 2 = weakest) ────
# Mapping: defect_cluster i  ↔  material_cluster i  (symmetric severity)
MATERIAL_CLUSTER_MEANS: dict[int, dict[str, float]] = {
    0: {  # strongest concrete — maps to LOW damage
        "mat_cement":           381.3,
        "mat_blast_furnace_slag": 115.9,
        "mat_fly_ash":           28.2,
        "mat_water":            163.2,
        "mat_superplasticizer":  12.3,
        "mat_coarse_aggregate": 921.5,
        "mat_fine_aggregate":   783.0,
        "mat_strength":          54.5,
        "mat_age":               33.0,
    },
    1: {  # medium concrete — maps to MEDIUM damage
        "mat_cement":           298.4,
        "mat_blast_furnace_slag":  73.1,
        "mat_fly_ash":           54.6,
        "mat_water":            185.7,
        "mat_superplasticizer":   6.1,
        "mat_coarse_aggregate": 972.3,
        "mat_fine_aggregate":   796.2,
        "mat_strength":          35.8,
        "mat_age":               90.0,
    },
    2: {  # weakest concrete — maps to HIGH damage
        "mat_cement":           218.6,
        "mat_blast_furnace_slag":  22.5,
        "mat_fly_ash":           87.3,
        "mat_water":            207.4,
        "mat_superplasticizer":   2.8,
        "mat_coarse_aggregate": 998.6,
        "mat_fine_aggregate":   801.1,
        "mat_strength":          19.2,
        "mat_age":              180.0,
    },
}

# ─── Stage-rank classification thresholds (from notebook Section 5) ───────────
# Uses GAI (0–1) + SI_ia (0–100) to assign a stage rank [0–3]
# Stage 0 → defect cluster 0 (low damage)
# Stage 1 → defect cluster 1 (medium damage)
# Stages 2 & 3 / TRANSITIONAL → defect cluster 2 (high damage)

GAI_LOW  = 0.15
GAI_HIGH = 0.35
SI_LOW   = 30.0
SI_HIGH  = 60.0
