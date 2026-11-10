
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.database.connection import MySQLConnector


def main():

    print("=================================================================")
    print("  FASE 1: ENTRENAMIENTO Y PERSISTENCIA DEL MODELO (DEPLOYMENT)")
    print("=================================================================\n")

    # Conexión a MySQL
    db = MySQLConnector()

    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            TRIM(p.country_of_birth) AS pais_nacimiento,
            comp.competition_id AS liga_id,
            comp.name AS liga,
            comp.country_name AS pais_liga,
            p.position AS posicion,
            p.sub_position AS sub_posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.foot AS pie_dominante,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        INNER JOIN clubs c
            ON p.current_club_id = c.club_id
        INNER JOIN competitions comp
            ON c.domestic_competition_id = comp.competition_id
        WHERE p.country_of_birth IS NOT NULL
            AND TRIM(p.country_of_birth) <> ''
            AND p.market_value_in_eur IS NOT NULL
            AND TRIM(p.market_value_in_eur) <> ''
            AND p.date_of_birth IS NOT NULL
            AND p.position IS NOT NULL
            AND TRIM(p.position) <> '';
    """

    try:
        df = db.extract_data(query)
    finally:
        db.close()

    print(f"[OK] Registros extraídos: {len(df)}")

    # Preprocesamiento
    df['estatura_cm'] = pd.to_numeric(
        df['estatura_cm'], errors='coerce'
    )
    df['edad'] = pd.to_numeric(
        df['edad'], errors='coerce'
    )
    df['valor_mercado'] = pd.to_numeric(
        df['valor_mercado'], errors='coerce'
    )

    df['posicion'] = df['posicion'].fillna('Sin Posicion')
    df['sub_posicion'] = df['sub_posicion'].fillna('')

    # Guardar las medianas de imputación para reutilizar en producción
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = df['estatura_cm'].median()
    mediana_global_edad = df['edad'].median()

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(
        lambda x: x.fillna(x.median())
    )

    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)

    # Crear variable objetivo
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # Convertir variables categóricas en dummies
    df_dummies = pd.get_dummies(
        df[['pais_nacimiento', 'liga_id', 'posicion', 'sub_posicion', 'pie_dominante']],
        drop_first=True,
        dtype=int
    )

    X = pd.concat([
        df[['edad', 'estatura_cm']],
        df_dummies
    ], axis=1)

    y = df['es_elite']

    # Entrenar modelo final
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X, y)

    # Crear paquete de despliegue
    paquete_despliegue = {
        'modelo': rf_model,
        'columnas_X': X.columns.tolist(),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite,

        # Perfiles históricos para la recomendación de ligas
        'perfiles_historicos': df[
            [
                'player_id',
                'pais_nacimiento',
                'liga_id',
                'liga',
                'pais_liga',
                'posicion',
                'sub_posicion',
                'pie_dominante',
                'edad',
                'estatura_cm'
            ]
        ].copy()
    }

    # Exportar paquete de despliegue
    os.makedirs('models', exist_ok=True)

    ruta_modelo = 'models/modelo_recomendacion_ligas.joblib'

    joblib.dump(paquete_despliegue, ruta_modelo)

    print(f"Paquete de modelo exportado exitosamente en: '{ruta_modelo}'")
    print(f"[OK] Perfiles históricos almacenados: {len(df)}")


if __name__ == "__main__":
    main()
