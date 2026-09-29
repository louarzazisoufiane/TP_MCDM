import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Aide à la Décision MCDM", layout="wide")

st.title("Système d'Aide à la Décision Multicritères (MCDM)")
st.markdown("Ce système implémente les méthodes présentées dans le cours (Entropie, CRITIC, WSM, WASPAS, TOPSIS, VIKOR).")

# ---------------------------------------------------------
# 1. DONNÉES D'EXEMPLE ET CHARGEMENT
# ---------------------------------------------------------
st.sidebar.header("1. Paramétrage des Données")

# Exemple basé sur l'exercice TOPSIS du document
data_exemple = pd.DataFrame({
    "Alternatives": ["M1", "M2", "M3", "M4"],
    "Style": [9, 7, 9, 6],
    "Fiabilité": [7, 8, 6, 7],
    "Economie": [9, 7, 8, 8],
    "Coût": [8, 8, 9, 6]
})

uploaded_file = st.sidebar.file_uploader("Chargez votre fichier CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Utilisation des données d'exemple (Voitures).")
    df = data_exemple.copy()

st.write("### Matrice de Décision Initiale")
st.dataframe(df)

# Sélection de la colonne des alternatives
alt_col = st.sidebar.selectbox("Colonne des Alternatives", df.columns, index=0)
criteres = [col for col in df.columns if col != alt_col]

# Types de critères (Max/Min)
st.sidebar.subheader("Types de Critères")
types_criteres = {}
for crit in criteres:
    # Par défaut, on met Max, sauf si ça s'appelle "Coût"
    default_index = 1 if "coût" in crit.lower() or "cost" in crit.lower() else 0
    types_criteres[crit] = st.sidebar.radio(f"{crit}", ["Maximiser (+)", "Minimiser (-)"], index=default_index, key=f"type_{crit}")

matrice = df[criteres].values
alternatives = df[alt_col].values

# ---------------------------------------------------------
# 2. PONDÉRATION DES CRITÈRES
# ---------------------------------------------------------
st.sidebar.header("2. Méthode de Pondération")
methode_poids = st.sidebar.selectbox("Choisissez une méthode", ["Manuelle", "Entropie", "CRITIC"])

poids = np.zeros(len(criteres))

if methode_poids == "Manuelle":
    st.sidebar.write("Définissez les poids (la somme doit idéalement faire 1)")
    poids_list = []
    for crit in criteres:
        poids_list.append(st.sidebar.number_input(f"Poids {crit}", value=1.0/len(criteres), step=0.01))
    poids = np.array(poids_list)
    poids = poids / np.sum(poids) # Normalisation automatique

elif methode_poids == "Entropie":
    # Méthode d'Entropie
    # 1. Normalisation proportionnelle
    sommes = matrice.sum(axis=0)
    p_ij = matrice / sommes
    
    # 2. Calcul de l'entropie Ej
    m = matrice.shape[0]
    k = 1.0 / np.log(m)
    # Remplacer les 0 par 1e-9 pour éviter log(0)
    p_ij_safe = np.where(p_ij == 0, 1e-9, p_ij)
    E_j = -k * np.sum(p_ij_safe * np.log(p_ij_safe), axis=0)
    
    # 3. Calcul des poids
    d_j = 1 - E_j
    poids = d_j / np.sum(d_j)
    
elif methode_poids == "CRITIC":
    # 1. Normalisation Min-Max
    matrice_norm = np.zeros_like(matrice, dtype=float)
    for j in range(matrice.shape[1]):
        col = matrice[:, j]
        c_min, c_max = np.min(col), np.max(col)
        if "Maximiser" in types_criteres[criteres[j]]:
            matrice_norm[:, j] = (col - c_min) / (c_max - c_min) if c_max != c_min else 1
        else:
            matrice_norm[:, j] = (c_max - col) / (c_max - c_min) if c_max != c_min else 1
            
    # 2. Ecart-type
    std_j = np.std(matrice_norm, axis=0, ddof=1)
    
    # 3. Matrice de corrélation
    df_norm = pd.DataFrame(matrice_norm)
    corr_matrix = df_norm.corr().fillna(0).values
    
    # 4. Indice C et Poids
    C_j = std_j * np.sum(1 - np.abs(corr_matrix), axis=1)
    poids = C_j / np.sum(C_j)

st.write(f"### Poids des Critères ({methode_poids})")
df_poids = pd.DataFrame([poids], columns=criteres, index=["Poids"])
st.dataframe(df_poids)

# ---------------------------------------------------------
# 3. CLASSEMENT DES ALTERNATIVES
# ---------------------------------------------------------
st.sidebar.header("3. Méthode de Classement")
methode_class = st.sidebar.selectbox("Choisissez une méthode", ["TOPSIS", "VIKOR", "WSM (Somme Pondérée)"])

scores = np.zeros(len(alternatives))

if methode_class == "TOPSIS":
    # 1. Normalisation vectorielle
    norm_vect = np.sqrt(np.sum(matrice**2, axis=0))
    matrice_norm = matrice / norm_vect
    
    # 2. Matrice pondérée
    v_ij = matrice_norm * poids
    
    # 3. Solutions idéales positives et négatives
    ideal_pos = np.zeros(len(criteres))
    ideal_neg = np.zeros(len(criteres))
    
    for j, crit in enumerate(criteres):
        if "Maximiser" in types_criteres[crit]:
            ideal_pos[j] = np.max(v_ij[:, j])
            ideal_neg[j] = np.min(v_ij[:, j])
        else:
            ideal_pos[j] = np.min(v_ij[:, j])
            ideal_neg[j] = np.max(v_ij[:, j])
            
    # 4. Distances
    S_plus = np.sqrt(np.sum((v_ij - ideal_pos)**2, axis=1))
    S_moins = np.sqrt(np.sum((v_ij - ideal_neg)**2, axis=1))
    
    # 5. Proximité relative
    scores = S_moins / (S_plus + S_moins)
    ascending_sort = False # Pour TOPSIS, on cherche le score (RC) le plus grand

elif methode_class == "VIKOR":
    # 1. Meilleures et mauvaises valeurs
    f_star = np.zeros(len(criteres))
    f_moins = np.zeros(len(criteres))
    
    for j, crit in enumerate(criteres):
        if "Maximiser" in types_criteres[crit]:
            f_star[j] = np.max(matrice[:, j])
            f_moins[j] = np.min(matrice[:, j])
        else:
            f_star[j] = np.min(matrice[:, j])
            f_moins[j] = np.max(matrice[:, j])
            
    # 2. Utilité (S) et Regret (R)
    S_i = np.zeros(len(alternatives))
    R_i = np.zeros(len(alternatives))
    
    for i in range(len(alternatives)):
        valeurs = poids * (f_star - matrice[i, :]) / (f_star - f_moins)
        S_i[i] = np.sum(valeurs)
        R_i[i] = np.max(valeurs)
        
    S_star, S_moins_val = np.min(S_i), np.max(S_i)
    R_star, R_moins_val = np.min(R_i), np.max(R_i)
    
    # 3. Calcul de Q (v = 0.5 par défaut)
    v = 0.5
    Q_i = np.zeros(len(alternatives))
    for i in range(len(alternatives)):
        term1 = v * (S_i[i] - S_star) / (S_moins_val - S_star) if S_moins_val != S_star else 0
        term2 = (1 - v) * (R_i[i] - R_star) / (R_moins_val - R_star) if R_moins_val != R_star else 0
        Q_i[i] = term1 + term2
        
    scores = Q_i
    ascending_sort = True # Pour VIKOR, on cherche le score (Q) le plus petit

elif methode_class == "WSM (Somme Pondérée)":
    # Normalisation linéaire
    matrice_norm = np.zeros_like(matrice, dtype=float)
    for j, crit in enumerate(criteres):
        col = matrice[:, j]
        if "Maximiser" in types_criteres[crit]:
            matrice_norm[:, j] = col / np.max(col)
        else:
            matrice_norm[:, j] = np.min(col) / col
            
    # Somme pondérée
    scores = np.sum(matrice_norm * poids, axis=1)
    ascending_sort = False # On cherche le score le plus grand

# ---------------------------------------------------------
# 4. RÉSULTATS
# ---------------------------------------------------------
st.write(f"### Résultats du Classement ({methode_class})")

df_resultats = pd.DataFrame({
    "Alternatives": alternatives,
    "Score": scores
})

# Tri des résultats
df_resultats = df_resultats.sort_values(by="Score", ascending=ascending_sort).reset_index(drop=True)
df_resultats.index = df_resultats.index + 1
df_resultats.index.name = "Rang"

st.dataframe(df_resultats.style.highlight_max(subset=['Score'], color='lightgreen') if not ascending_sort else df_resultats.style.highlight_min(subset=['Score'], color='lightgreen'))

st.bar_chart(data=df_resultats.set_index("Alternatives")["Score"])

st.success(f"La meilleure alternative selon la méthode {methode_class} est **{df_resultats.iloc[0]['Alternatives']}**.")