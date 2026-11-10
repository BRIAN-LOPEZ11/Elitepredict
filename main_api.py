
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

# ==========================================
# CONFIGURACIÓN Y CARGA DEL MODELO
# ==========================================

RUTA_MODELO = "models/modelo_scouting_rf.joblib"

if not os.path.exists(RUTA_MODELO):
    raise FileNotFoundError(
        f"No se encontró el modelo: {RUTA_MODELO}"
    )

paquete = joblib.load(RUTA_MODELO)

modelo = paquete["modelo"]
medianas_altura = paquete["medianas_altura"]
mediana_general = paquete["mediana_general"]


# ==========================================
# CONFIGURACIÓN Y CARGA DEL MODELO DE LIGAS
# ==========================================

RUTA_MODELO_LIGAS = "models/modelo_recomendacion_ligas.joblib"

if not os.path.exists(RUTA_MODELO_LIGAS):
    raise FileNotFoundError(
        f"No se encontró el modelo: {RUTA_MODELO_LIGAS}"
    )

paquete_ligas = joblib.load(RUTA_MODELO_LIGAS)

modelo_ligas = paquete_ligas["modelo"]
columnas_X_ligas = paquete_ligas["columnas_X"]
medianas_estatura = paquete_ligas["medianas_estatura"]
mediana_global_estatura = paquete_ligas["mediana_global_estatura"]
mediana_global_edad = paquete_ligas["mediana_global_edad"]
perfiles_historicos = paquete_ligas["perfiles_historicos"]

app = FastAPI(
    title="Scouting API",
    description="API para clasificar jugadores de fútbol según su valor de mercado.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)


# ==========================================
# ESTRUCTURA DE LOS DATOS DE ENTRADA
# ==========================================

class Jugador(BaseModel):
    position: str = Field(..., min_length=1)
    sub_position: str = Field(default="Unknown")
    height_in_cm: float | None = Field(default=None, gt=0, le=250)
    age: float = Field(..., ge=15, le=60)
    foot: str = Field(default="Unknown")


class PaisJugador(BaseModel):
    pais_nacimiento: str = Field(..., min_length=1)

# ==========================================
# ENDPOINT PRINCIPAL
# ==========================================

@app.get("/")
def inicio():
    return {
        "mensaje": "Scouting API funcionando correctamente",
        
    }


# ==========================================
# VERIFICAR ESTADO DEL MODELO
# ==========================================

@app.get("/health")
def health():
    return {
        "estado": "activo",
        "modelo_cargado": modelo is not None
    }


# ==========================================
# PREDICCIÓN DE JUGADORES
# ==========================================

@app.post("/predict")
def predecir(jugador: Jugador):

    try:
        datos = jugador.model_dump()

        # Imputación de altura según la posición
        if datos["height_in_cm"] is None:
            datos["height_in_cm"] = medianas_altura.get(
                datos["position"],
                mediana_general
            )

        # Crear DataFrame con las variables esperadas
        entrada = pd.DataFrame([{
            "position": datos["position"],
            "sub_position": datos["sub_position"],
            "height_in_cm": datos["height_in_cm"],
            "age": datos["age"],
            "foot": datos["foot"]
        }])

        # Obtener predicción y probabilidades
        prediccion = int(modelo.predict(entrada)[0])
        probabilidades = modelo.predict_proba(entrada)[0]

        probabilidad_elite = float(probabilidades[1] * 100)

        categoria = "Élite" if prediccion == 1 else "Estándar"

        return {
            "jugador": datos,
            "prediccion": prediccion,
            "categoria": categoria,
            "probabilidad_elite": round(probabilidad_elite, 2),
            "mensaje": (
                "Jugador clasificado como Élite"
                if prediccion == 1
                else "Jugador clasificado como Estándar"
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(error)}"
        )



# ==========================================
# RECOMENDACIÓN DE LIGAS SEGÚN PAÍS
# ==========================================

@app.post("/recommend-leagues")
def recomendar_ligas(datos: PaisJugador):

    try:
        pais = datos.pais_nacimiento.strip()

        # Filtrar jugadores según el país de nacimiento
        jugadores_pais = perfiles_historicos[
            perfiles_historicos['pais_nacimiento'].str.strip().str.casefold()
            == pais.casefold()
        ].copy()

        if jugadores_pais.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron jugadores de {pais}."
            )

        # Identificar ligas con mínimo 10 jugadores
        ligas_candidatas = (
            jugadores_pais.groupby('liga_id')
            .agg(
                liga=('liga', 'first'),
                pais_liga=('pais_liga', 'first'),
                cantidad_jugadores=('player_id', 'count')
            )
            .reset_index()
        )

        ligas_candidatas = ligas_candidatas[
            ligas_candidatas['cantidad_jugadores'] >= 10
        ]

        if ligas_candidatas.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No hay ligas con al menos 10 jugadores de {pais}."
            )

        # Obtener perfiles históricos del país
        perfiles = jugadores_pais[
            [
                'pais_nacimiento',
                'liga_id',
                'posicion',
                'sub_posicion',
                'pie_dominante',
                'edad',
                'estatura_cm'
            ]
        ].copy()

        resultados = []

        # Evaluar los perfiles en cada liga candidata
        for _, liga_candidata in ligas_candidatas.iterrows():

            perfiles_liga = perfiles.copy()

            perfiles_liga['liga_id'] = liga_candidata['liga_id']

            # Convertir variables categóricas en dummies
            df_dummies = pd.get_dummies(
                perfiles_liga[
                    [
                        'pais_nacimiento',
                        'liga_id',
                        'posicion',
                        'sub_posicion',
                        'pie_dominante'
                    ]
                ],
                dtype=int
            )

            X_nuevo = pd.concat(
                [
                    perfiles_liga[['edad', 'estatura_cm']],
                    df_dummies
                ],
                axis=1
            )

            # Alinear columnas con las utilizadas en el entrenamiento
            X_nuevo = X_nuevo.reindex(
                columns=columnas_X_ligas,
                fill_value=0
            )

            # Obtener probabilidades de pertenecer a Élite
            probabilidades = modelo_ligas.predict_proba(X_nuevo)[:, 1]

            promedio_elite = probabilidades.mean()

            resultados.append({
                'liga': liga_candidata['liga'],
                'pais_liga': liga_candidata['pais_liga'],
                'cantidad_jugadores': int(
                    liga_candidata['cantidad_jugadores']
                ),
                'probabilidad_elite': round(
                    float(promedio_elite * 100), 2
                )
            })

        # Ordenar de mayor a menor probabilidad
        resultados.sort(
            key=lambda x: x['probabilidad_elite'],
            reverse=True
        )

        return {
            'pais_nacimiento': pais,
            'total_ligas': len(resultados),
            'recomendaciones': resultados
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al recomendar ligas: {str(error)}"
        )
