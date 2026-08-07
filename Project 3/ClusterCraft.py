import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

df = pd.read_excel("Copy of Dataset for Data Analytics.xlsx")
print("Raw shape:", df.shape)

df["Date"] = pd.to_datetime(df["Date"])
snapshot_date = df["Date"].max() + pd.Timedelta(days=1)

feat = pd.DataFrame()

feat["CustomerID"] = df["CustomerID"]
feat["Recency"] = (snapshot_date - df["Date"]).dt.days
feat["Quantity"] = df["Quantity"]
feat["UnitPrice"] = df["UnitPrice"]
feat["TotalPrice"] = df["TotalPrice"]
feat["ItemsInCart"] = df["ItemsInCart"]

feat["CouponUsed"] = df["CouponCode"].notna().astype(int)
feat["OrderCompleted"] = (df["OrderStatus"] == "Delivered").astype(int)

cat_cols = [
    "Product",
    "PaymentMethod",
    "OrderStatus",
    "ReferralSource"
]

feat = pd.concat(
    [feat, pd.get_dummies(df[cat_cols], prefix=cat_cols)],
    axis=1
)

print("Engineered feature shape (incl. ID):", feat.shape)

X = feat.drop(columns=["CustomerID"])

print("Total numeric feature columns:", X.shape[1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)

n_components_95 = np.argmax(cum_var >= 0.95) + 1

print("Components needed for 95% variance:", n_components_95)

plt.figure(figsize=(6,4))
plt.plot(range(1, len(cum_var)+1), cum_var, marker="o")
plt.axhline(0.95, color="orange", linestyle="--", label="95% Threshold")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=120)
plt.close()

pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

wcss = []

K_range = range(1,11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    wcss.append(km.inertia_)

kneedle = KneeLocator(
    list(K_range),
    wcss,
    curve="convex",
    direction="decreasing"
)

optimal_k = kneedle.elbow

print("Elbow suggests K =", optimal_k)

plt.figure(figsize=(6,4))
plt.plot(K_range, wcss, marker="o")

if optimal_k is not None:
    plt.axvline(
        optimal_k,
        color="red",
        linestyle="--",
        label=f"Elbow K={optimal_k}"
    )

plt.xlabel("Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.legend()
plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=120)
plt.close()

sil_scores = {}

for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    sil_scores[k] = silhouette_score(X_pca, labels)

best_k = max(sil_scores, key=sil_scores.get)

print("Silhouette Scores")

for k, score in sil_scores.items():
    print(f"K={k}: {score:.3f}")

print("Best K =", best_k)

plt.figure(figsize=(6,4))
plt.plot(
    list(sil_scores.keys()),
    list(sil_scores.values()),
    marker="o",
    color="green"
)

plt.xlabel("Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=120)
plt.close()

FINAL_K = best_k

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=10
)

feat["Cluster"] = kmeans.fit_predict(X_pca)

print("\nFinal Model K =", FINAL_K)
print(feat["Cluster"].value_counts())

plt.figure(figsize=(7,5))

scatter = plt.scatter(
    X_pca_2d[:,0],
    X_pca_2d[:,1],
    c=feat["Cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"Customer Segments (K={FINAL_K})")

plt.legend(
    *scatter.legend_elements(),
    title="Cluster"
)

plt.tight_layout()
plt.savefig("cluster_scatter.png", dpi=120)
plt.close()

persona_cols = [
    "Recency",
    "Quantity",
    "UnitPrice",
    "TotalPrice",
    "ItemsInCart",
    "CouponUsed",
    "OrderCompleted"
]

persona = (
    feat.groupby("Cluster")[persona_cols]
    .mean()
    .round(1)
)

persona["CustomerCount"] = (
    feat["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n===== CUSTOMER PERSONAS =====")
print(persona)

persona.to_csv("cluster_personas.csv")

feat.to_csv(
    "customers_with_clusters.csv",
    index=False
)

print("\nFiles Saved Successfully")
print("- pca_variance.png")
print("- elbow_plot.png")
print("- silhouette_plot.png")
print("- cluster_scatter.png")
print("- cluster_personas.csv")
print("- customers_with_clusters.csv")