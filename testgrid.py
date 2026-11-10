
import pandas as pd

from src.database.connection import MySQLConnector

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1. EXTRACCIÓN DE DATOS
# ==========================================================

print("\n--- EXTRACCIÓN DE DATOS ---")

db = MySQLConnector()

try:
    df = db.extract_data("SELECT * FROM players")
finally:
    db.close()


# ==========================================================
# 2. LIMPIEZA Y TRANSFORMACIÓN
# ==========================================================

print("\n--- LIMPIEZA DE DATOS ---")

df = df.copy()
df = df.replace(r"^\s*$", pd.NA, regex=True)

df["height_in_cm"] = pd.to_numeric(
    df["height_in_cm"], errors="coerce"
)

df["market_value_in_eur"] = pd.to_numeric(
    df["market_value_in_eur"], errors="coerce"
)

df["date_of_birth"] = pd.to_datetime(
    df["date_of_birth"], errors="coerce"
)

# Calcular edad
today = pd.Timestamp.today().normalize()

df["age"] = (
    today.year
    - df["date_of_birth"].dt.year
    - (
        (today.month < df["date_of_birth"].dt.month)
        | (
            (today.month == df["date_of_birth"].dt.month)
            & (today.day < df["date_of_birth"].dt.day)
        )
    ).astype(float)
)

# Normalizar variables categóricas
for column in ["position", "sub_position", "foot"]:
    df[column] = df[column].astype("string").str.strip()

df["position"] = df["position"].replace(
    ["Missing", "missing"], pd.NA
).fillna("Unknown")

df["sub_position"] = df["sub_position"].fillna("Unknown")
df["foot"] = df["foot"].fillna("Unknown")

# Eliminar jugadores sin valor de mercado
df = df.dropna(subset=["market_value_in_eur"])


# ==========================================================
# 3. CREACIÓN DE LA VARIABLE OBJETIVO
# ==========================================================

print("\n--- VARIABLE OBJETIVO ---")

q3 = df["market_value_in_eur"].quantile(0.75)

df["elite"] = (
    df["market_value_in_eur"] >= q3
).astype(int)

print(f"Percentil 75: €{q3:,.2f}")
print(df["elite"].value_counts())


# ==========================================================
# 4. SELECCIÓN DE CARACTERÍSTICAS
# ==========================================================

# No incluimos market_value_in_eur porque se utilizó
# para construir la variable objetivo.

features = [
    "position",
    "sub_position",
    "height_in_cm",
    "age",
    "foot"
]

X = df[features].copy()
y = df["elite"].copy()

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = X_train.copy()
X_test = X_test.copy()

print("\n--- DIVISIÓN DE DATOS ---")
print(f"Entrenamiento: {len(X_train)}")
print(f"Prueba: {len(X_test)}")


# ==========================================================
# 5. IMPUTACIÓN DE ALTURA POR POSICIÓN
# ==========================================================

# Las medianas se calculan únicamente con entrenamiento.
# Así evitamos utilizar información del conjunto de prueba.

medianas_altura = X_train.groupby("position")[
    "height_in_cm"
].median()

mediana_general = X_train["height_in_cm"].median()


def imputar_altura(data):
    data = data.copy()

    data["height_in_cm"] = data["height_in_cm"].fillna(
        data["position"].map(medianas_altura)
    )

    data["height_in_cm"] = data["height_in_cm"].fillna(
        mediana_general
    )

    return data


X_train = imputar_altura(X_train)
X_test = imputar_altura(X_test)


# ==========================================================
# 6. PREPROCESAMIENTO
# ==========================================================

numeric_features = [
    "height_in_cm",
    "age"
]

categorical_features = [
    "position",
    "sub_position",
    "foot"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================================
# 7. CONFIGURACIÓN DEL RANDOM FOREST
# ==========================================================

print("\n--- CONFIGURACIÓN DE GRIDSEARCHCV ---")

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )
        )
    ]
)


# ==========================================================
# 8. DEFINICIÓN DE HIPERPARÁMETROS
# ==========================================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 4]
}

# 24 combinaciones posibles


# ==========================================================
# 9. VALIDACIÓN CRUZADA Y BÚSQUEDA
# ==========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=random_forest,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("\n--- INICIANDO GRIDSEARCHCV ---")
print("Probando combinaciones de hiperparámetros...")

grid_search.fit(X_train, y_train)


# ==========================================================
# 10. MEJORES HIPERPARÁMETROS
# ==========================================================

print("\n--- MEJORES RESULTADOS DE GRIDSEARCHCV ---")

print("Mejores hiperparámetros:")
print(grid_search.best_params_)

print(
    f"\nMejor F1-score promedio en validación cruzada: "
    f"{grid_search.best_score_:.4f}"
)


# ==========================================================
# 11. EVALUACIÓN DEL MEJOR MODELO
# ==========================================================

best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print("\n--- EVALUACIÓN DEL MODELO OPTIMIZADO ---")

print(f"Accuracy entrenamiento: {accuracy_train:.4f}")
print(f"Accuracy prueba: {accuracy_test:.4f}")
print(f"Brecha: {accuracy_train - accuracy_test:.4f}")


# ==========================================================
# 12. MATRIZ DE CONFUSIÓN
# ==========================================================

print("\n--- MATRIZ DE CONFUSIÓN ---")

print(confusion_matrix(y_test, y_test_pred))


# ==========================================================
# 13. REPORTE DE CLASIFICACIÓN
# ==========================================================

print("\n--- REPORTE DE CLASIFICACIÓN ---")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["Estándar", "Élite"]
    )
)


# ==========================================================
# 14. RESULTADOS DE LAS MEJORES CONFIGURACIONES
# ==========================================================

resultados = pd.DataFrame(grid_search.cv_results_)

resultados = resultados[
    [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].sort_values("rank_test_score")

print("\n--- TOP 5 CONFIGURACIONES ---")
print(resultados.head(5).to_string(index=False))


print("\n--- PROCESO FINALIZADO ---")
