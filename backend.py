from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests


# Dirección interna de la API de predicción
API_INTERNA = "http://127.0.0.1:8000/predict"
API_INTERNA_LIGAS = "http://127.0.0.1:8000/recommend-leagues"

app = FastAPI(
    title="Scouting Backend",
    description="Servicio intermediario para el sistema de scouting.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

class Jugador(BaseModel):
    position: str = Field(..., min_length=1)
    sub_position: str = Field(default="Unknown")
    height_in_cm: float | None = Field(default=None, gt=0, le=250)
    age: float = Field(..., ge=15, le=60)
    foot: str = Field(default="Unknown")

class PaisJugador(BaseModel):
    pais_nacimiento: str = Field(..., min_length=1)


class Solicitud(BaseModel):
    tipo: str = Field(...)

    position: str | None = None
    sub_position: str = "Unknown"
    height_in_cm: float | None = Field(default=None, gt=0, le=250)
    age: float | None = Field(default=None, ge=15, le=60)
    foot: str = "Unknown"

    pais_nacimiento: str | None = None

@app.get("/")
def inicio():
    return {
        "mensaje": "Scouting Backend funcionando correctamente"
    }



@app.post("/api")
def procesar_solicitud(datos: Solicitud):

    try:

        # Identificar el tipo de operación
        if datos.tipo == "clasificacion":

            jugador = Jugador(
                position=datos.position,
                sub_position=datos.sub_position,
                height_in_cm=datos.height_in_cm,
                age=datos.age,
                foot=datos.foot
            )

            url = API_INTERNA
            payload = jugador.model_dump()

        elif datos.tipo == "recomendacion_ligas":

            pais = PaisJugador(
                pais_nacimiento=datos.pais_nacimiento
            )

            url = API_INTERNA_LIGAS
            payload = pais.model_dump()

        else:
            raise HTTPException(
                status_code=400,
                detail="Tipo de solicitud no válido."
            )

        # Enviar solicitud a la API principal
        respuesta = requests.post(
            url,
            json=payload,
            timeout=30
        )

        respuesta.raise_for_status()

        return respuesta.json()

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="La API interna tardó demasiado en responder."
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="El servicio de predicción no está disponible."
        )

    except requests.exceptions.HTTPError:
        if respuesta.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=respuesta.json().get("detail", "No se encontraron resultados.")
            )

        raise HTTPException(
            status_code=502,
            detail="Error en el servicio interno de predicción."
        )
