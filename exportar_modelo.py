
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.database.connection import MySQLConnector


def obtener_datos():
    """Extrae los datos de jugadores desde MySQL."""

    db = MySQLConnector()

    try:
        df = db.extract_data("SELECT * FROM players")
    finally:
        db.close()

    print(f"[OK] Registros extraídos: {len(df)}")

    return df


def limpiar_datos(df):
    """Aplicando limpieza todos los jugadores"""

    df = df.copy()

    # Convertir cadenas vacías en valores nulos
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

    # Convertir variables numéricas
    df["height_in_cm"] = pd.to_numeric(
        df["height_in_cm"], errors="coerce"
    )

    df["market_value_in_eur"] = pd.to_numeric(
        df["market_value_in_eur"], errors="coerce"
    )

    # Convertir fecha de nacimiento y calcular edad
    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"], errors="coerce"
    )

    fecha_actual = pd.Timestamp.today().normalize()

    df["age"] = (
        (fecha_actual - df["date_of_birth"]).dt.days / 365.25
    )

    df["age"] = df["age"].round().astype("Int64")

    # Normalizar variables categóricas
    for columna in ["position", "sub_position", "foot"]:
        df[columna] = (
            df[columna]
            .astype("string")
            .str.strip()
            .str.title()
        )

    # Tratar posiciones faltantes
    df["position"] = df["position"].replace(
        ["Missing", "Nan", "<NA>"], pd.NA
    )

    # Las categorías faltantes se identifican como Unknown
    for columna in ["position", "sub_position", "foot"]:
        df[columna] = df[columna].fillna("Unknown")

    # Eliminar registros sin valor de mercado
    df = df.dropna(subset=["market_value_in_eur"]).copy()

    print(f"[OK] Registros después de limpieza: {len(df)}")

    return df


def entrenar_modelo(df):
    """Entrena el Random Forest con los parámetros seleccionados."""

    # 1. Crear variable objetivo usando el percentil 75
    q3 = df["market_value_in_eur"].quantile(0.75)

    df["elite"] = (
        df["market_value_in_eur"] >= q3
    ).astype(int)

    print(f"\nPercentil 75 del valor de mercado: €{q3:,.2f}")

    # 2. Seleccionar las variables predictoras
    features = [
        "position",
        "sub_position",
        "height_in_cm",
        "age",
        "foot"
    ]

    X = df[features].copy()
    y = df["elite"].copy()

    # 3. División de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # 4. Calcular medianas de altura usando únicamente entrenamiento
    medianas_altura = X_train.groupby("position")[
        "height_in_cm"
    ].median()

    mediana_general = X_train["height_in_cm"].median()

    # Imputar alturas faltantes
    X_train["height_in_cm"] = (
        X_train["height_in_cm"]
        .fillna(X_train["position"].map(medianas_altura))
        .fillna(mediana_general)
    )

    X_test["height_in_cm"] = (
        X_test["height_in_cm"]
        .fillna(X_test["position"].map(medianas_altura))
        .fillna(mediana_general)
    )

    # 5. Definir preprocesamiento
    variables_numericas = ["height_in_cm", "age"]

    variables_categoricas = [
        "position",
        "sub_position",
        "foot"
    ]

    transformador_numerico = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    transformador_categorico = Pipeline(
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

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                transformador_numerico,
                variables_numericas
            ),
            (
                "cat",
                transformador_categorico,
                variables_categoricas
            )
        ]
    )

    # 6. Random Forest con los parámetros ganadores de GridSearchCV
    clasificador = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    modelo = Pipeline(
        steps=[
            ("preprocessor", preprocesador),
            ("classifier", clasificador)
        ]
    )

    # 7. Entrenar modelo
    print("\nEntrenando modelo Random Forest...")

    modelo.fit(X_train, y_train)

    # 8. Evaluación
    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)

    print("\n========== EVALUACIÓN DEL MODELO ==========")
    print(f"Exactitud en prueba: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            predicciones,
            target_names=["Standard", "Elite"]
        )
    )

    # 9. Construir paquete compatible con main_api.py
    paquete_modelo = {
        "modelo": modelo,
        "medianas_altura": medianas_altura.to_dict(),
        "mediana_general": mediana_general,
        "percentil_75": q3,
        "features": features
    }

    return paquete_modelo


def exportar_modelo(paquete):
    """Serializa el modelo y sus metadatos."""

    ruta_salida = "models/modelo_scouting_rf.joblib"

    os.makedirs("models", exist_ok=True)

    joblib.dump(paquete, ruta_salida)

    print("\n============================================")
    print("[OK] MODELO EXPORTADO CORRECTAMENTE")
    print(f"Archivo: {ruta_salida}")
    print(f"Variables: {paquete['features']}")
    print(f"Percentil 75: €{paquete['percentil_75']:,.2f}")
    print("============================================")


def main():
    print("============================================")
    print("  ENTRENAMIENTO Y EXPORTACIÓN")
    print("============================================")

    datos = obtener_datos()
    datos = limpiar_datos(datos)

    paquete = entrenar_modelo(datos)

    exportar_modelo(paquete)


if __name__ == "__main__":
    main()
