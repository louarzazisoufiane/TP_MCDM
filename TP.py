"""Interactive crisp MCDM decision-support application."""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Aide à la décision MCDM", page_icon="📊", layout="wide")

EXAMPLE = pd.DataFrame({
    "Alternative": ["M1", "M2", "M3", "M4"],
    "Style": [7, 8, 9, 6], "Fiabilité": [9, 7, 6, 7],
    "Économie": [9, 8, 8, 8], "Coût": [8, 7, 9, 6],
})


def benefit(direction):
    return direction == "À maximiser (+)"


def minmax(x, directions):
    """Return a 0–1 matrix in which 1 is always preferred."""
    result = np.zeros_like(x, dtype=float)
    for j, direction in enumerate(directions):
        low, high = x[:, j].min(), x[:, j].max()
        if np.isclose(low, high):
            result[:, j] = 1.0
        elif benefit(direction):
            result[:, j] = (x[:, j] - low) / (high - low)
        else:
            result[:, j] = (high - x[:, j]) / (high - low)
    return result


def ratio_normalize(x, directions):
    """Course WSM/WPM normalization, safely falling back when a cost is zero."""
    result = np.zeros_like(x, dtype=float)
    for j, direction in enumerate(directions):
        col = x[:, j]
        if benefit(direction):
            result[:, j] = col / col.max() if col.max() else 1.0
        elif col.min() > 0:
            result[:, j] = col.min() / col
        else:
            result[:, j] = minmax(x[:, [j]], [direction]).ravel()
    return result


def entropy_weights(x, directions):
    # Direction-aware, nonnegative input is required by the entropy formula.
    normalized = minmax(x, directions) + 1e-12
    p = normalized / normalized.sum(axis=0, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(len(x))
    divergence = 1 - entropy
    weights = divergence / divergence.sum() if divergence.sum() > 1e-12 else np.full(x.shape[1], 1 / x.shape[1])
    return weights, pd.DataFrame({"Entropie Eⱼ": entropy, "Divergence 1 − Eⱼ": divergence, "Poids": weights})


def critic_weights(x, directions):
    normalized = minmax(x, directions)
    std = np.std(normalized, axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
    correlation = pd.DataFrame(normalized).corr().fillna(0).to_numpy()
    information = std * np.sum(1 - np.abs(correlation), axis=1)
    weights = information / information.sum() if information.sum() > 1e-12 else np.full(x.shape[1], 1 / x.shape[1])
    return weights, pd.DataFrame({"Écart-type σⱼ": std, "Information Cⱼ": information, "Poids": weights}), correlation


def topsis(x, weights, directions):
    denominator = np.sqrt((x ** 2).sum(axis=0))
    normalized = np.divide(x, denominator, out=np.zeros_like(x, dtype=float), where=denominator != 0)
    weighted = normalized * weights
    positive = np.array([weighted[:, j].max() if benefit(d) else weighted[:, j].min() for j, d in enumerate(directions)])
    negative = np.array([weighted[:, j].min() if benefit(d) else weighted[:, j].max() for j, d in enumerate(directions)])
    s_plus = np.sqrt(((weighted - positive) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted - negative) ** 2).sum(axis=1))
    score = np.divide(s_minus, s_plus + s_minus, out=np.full(len(x), .5), where=(s_plus + s_minus) != 0)
    return score, {"Matrice normalisée pondérée": weighted, "Idéal positif": positive, "Idéal négatif": negative, "S⁺": s_plus, "S⁻": s_minus}


def vikor(x, weights, directions, v):
    best = np.array([x[:, j].max() if benefit(d) else x[:, j].min() for j, d in enumerate(directions)])
    worst = np.array([x[:, j].min() if benefit(d) else x[:, j].max() for j, d in enumerate(directions)])
    losses = np.divide(weights * (best - x), best - worst, out=np.zeros_like(x, dtype=float), where=(best - worst) != 0)
    s, r = losses.sum(axis=1), losses.max(axis=1)
    s_term = np.divide(s - s.min(), s.max() - s.min(), out=np.zeros_like(s), where=(s.max() - s.min()) != 0)
    r_term = np.divide(r - r.min(), r.max() - r.min(), out=np.zeros_like(r), where=(r.max() - r.min()) != 0)
    return v * s_term + (1 - v) * r_term, {"Meilleur f*": best, "Pire f⁻": worst, "S (utilité de groupe)": s, "R (regret individuel)": r}


def calculate(method, x, weights, directions, lam=.5, v=.5):
    if method in ("WSM", "WPM", "WASPAS"):
        normalized = ratio_normalize(x, directions)
        wsm = (normalized * weights).sum(axis=1)
        wpm = np.prod(np.power(np.clip(normalized, 1e-12, None), weights), axis=1)
        if method == "WSM": return wsm, False, {"Matrice normalisée": normalized, "Score WSM": wsm}
        if method == "WPM": return wpm, False, {"Matrice normalisée": normalized, "Score WPM": wpm}
        return lam * wsm + (1 - lam) * wpm, False, {"Matrice normalisée": normalized, "Q¹ (WSM)": wsm, "Q² (WPM)": wpm}
    if method == "TOPSIS":
        score, details = topsis(x, weights, directions)
        return score, False, details
    score, details = vikor(x, weights, directions, v)
    return score, True, details


def make_ranking(alternatives, scores, ascending):
    table = pd.DataFrame({"Alternative": alternatives, "Score": scores})
    table = table.sort_values("Score", ascending=ascending, kind="stable").reset_index(drop=True)
    table.index += 1
    table.index.name = "Rang"
    return table


st.title("📊 Système d’aide à la décision multicritère")
st.caption("Entropie, CRITIC, WSM, WPM, WASPAS, TOPSIS et VIKOR — avec calculs et analyse de robustesse.")

with st.sidebar:
    st.header("1. Données")
    upload = st.file_uploader("Importer une matrice CSV", type="csv")

if upload is None:
    df = EXAMPLE.copy()
    st.info("Jeu d’exemple du cours : sélection d’une voiture. Importez un CSV pour utiliser vos données.")
else:
    try:
        df = pd.read_csv(upload)
    except (UnicodeDecodeError, pd.errors.ParserError):
        upload.seek(0)
        df = pd.read_csv(upload, encoding="latin-1", sep=None, engine="python")

if df.empty or len(df.columns) < 2:
    st.error("Le fichier doit contenir une colonne d’alternatives et au moins un critère.")
    st.stop()

with st.sidebar:
    alternative_column = st.selectbox("Colonne des alternatives", df.columns, index=0)

criteria = [column for column in df.columns if column != alternative_column]
numeric = df[criteria].apply(pd.to_numeric, errors="coerce")
invalid = numeric.columns[numeric.isna().any()].tolist()
if invalid:
    st.error("Critères non numériques ou incomplets : " + ", ".join(invalid))
    st.stop()
if df[alternative_column].isna().any() or df[alternative_column].duplicated().any():
    st.error("Les noms des alternatives doivent être présents et uniques.")
    st.stop()

alternatives, matrix = df[alternative_column].astype(str).to_numpy(), numeric.to_numpy(float)

with st.sidebar:
    st.header("2. Préférences")
    directions = []
    for criterion in criteria:
        cost_words = ("coût", "cout", "cost", "délai", "delai", "risque", "risk")
        default = 1 if any(word in criterion.lower() for word in cost_words) else 0
        directions.append(st.radio(criterion, ["À maximiser (+)", "À minimiser (−)"], index=default, key=f"direction_{criterion}"))

    st.header("3. Pondération")
    weight_method = st.selectbox("Méthode des poids", ["Manuelle", "Entropie", "CRITIC"])
    details = correlation = None
    if weight_method == "Manuelle":
        raw = np.array([st.number_input(f"Poids — {c}", min_value=0., value=1 / len(criteria), step=.01, key=f"weight_{c}") for c in criteria])
        if np.isclose(raw.sum(), 0):
            st.error("Au moins un poids doit être supérieur à zéro.")
            st.stop()
        weights = raw / raw.sum()
    elif weight_method == "Entropie":
        weights, details = entropy_weights(matrix, directions)
    else:
        weights, details, correlation = critic_weights(matrix, directions)

    st.header("4. Classement")
    method = st.selectbox("Méthode", ["TOPSIS", "VIKOR", "WSM", "WPM", "WASPAS"])
    lam = st.slider("λ WASPAS (WSM ↔ WPM)", 0., 1., .5, .05) if method == "WASPAS" else .5
    v = st.slider("v VIKOR (utilité de groupe)", 0., 1., .5, .05) if method == "VIKOR" else .5

st.subheader("Matrice de décision")
st.dataframe(df, use_container_width=True, hide_index=True)

weight_table = pd.DataFrame({"Critère": criteria, "Sens": directions, "Poids": weights})
st.subheader(f"Poids des critères — {weight_method}")
st.dataframe(weight_table.style.format({"Poids": "{:.4f}"}), use_container_width=True, hide_index=True)
with st.expander("Détails du calcul des poids"):
    if details is None:
        st.write("Les poids manuels sont normalisés automatiquement (somme = 1).")
    else:
        details.index = criteria
        st.dataframe(details.style.format("{:.4f}"), use_container_width=True)
    if correlation is not None:
        st.caption("Matrice de corrélation utilisée par CRITIC")
        st.dataframe(pd.DataFrame(correlation, index=criteria, columns=criteria).style.format("{:.3f}"), use_container_width=True)

scores, ascending, steps = calculate(method, matrix, weights, directions, lam, v)
result = make_ranking(alternatives, scores, ascending)
left, right = st.columns([3, 2])
with left:
    st.subheader(f"Classement — {method}")
    style = result.style.highlight_min("Score", color="#d9f2d9") if ascending else result.style.highlight_max("Score", color="#d9f2d9")
    st.dataframe(style.format({"Score": "{:.6f}"}), use_container_width=True)
with right:
    st.metric("Meilleure alternative", result.iloc[0]["Alternative"], f"score : {result.iloc[0]['Score']:.4f}")
    st.caption("VIKOR : le plus petit Q est préférable. Les autres méthodes : le plus grand score est préférable.")
    st.bar_chart(result.set_index("Alternative")["Score"])

with st.expander(f"Étapes de calcul — {method}"):
    for label, value in steps.items():
        st.markdown(f"**{label}**")
        if value.ndim == 2:
            st.dataframe(pd.DataFrame(value, index=alternatives, columns=criteria).style.format("{:.5f}"), use_container_width=True)
        else:
            index = alternatives if len(value) == len(alternatives) else criteria
            st.dataframe(pd.DataFrame({label: value}, index=index).style.format("{:.5f}"), use_container_width=True)

st.subheader("Robustesse : comparaison des méthodes")
rank_columns = {}
for candidate in ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]:
    candidate_scores, candidate_ascending, _ = calculate(candidate, matrix, weights, directions, lam, v)
    rank_columns[candidate] = make_ranking(alternatives, candidate_scores, candidate_ascending).reset_index().set_index("Alternative")["Rang"]
rankings = pd.DataFrame(rank_columns).loc[alternatives]
st.dataframe(rankings, use_container_width=True)
st.caption("Corrélation de Spearman : proche de 1 = classements similaires ; proche de −1 = classements opposés.")
st.dataframe(rankings.corr(method="spearman").style.format("{:.3f}"), use_container_width=True)

st.download_button("Télécharger le classement CSV", result.to_csv(index=True).encode("utf-8-sig"), "classement_mcdm.csv", "text/csv")
