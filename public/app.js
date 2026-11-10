// Único servicio al que se conecta el navegador.
const API_URL = "";

// ==========================================
// ELEMENTOS DEL FORMULARIO
// ==========================================

const form = document.getElementById("scoutingForm");
const analyzeButton = document.getElementById("analyzeButton");

const positionSelect = document.getElementById("position");
const subPositionSelect = document.getElementById("sub_position");

// ==========================================
// ESTADOS DE LA INTERFAZ
// ==========================================

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const predictionResult = document.getElementById("predictionResult");

// Elementos del resultado
const errorMessage = document.getElementById("errorMessage");
const categoryBadge = document.getElementById("categoryBadge");
const probabilityValue = document.getElementById("probabilityValue");
const probabilityBar = document.getElementById("probabilityBar");
const resultTitle = document.getElementById("resultTitle");
const resultMessage = document.getElementById("resultMessage");

// ==========================================
// ELEMENTOS DE RECOMENDACIÓN DE LIGAS
// ==========================================

const leagueRecommendationSection =
    document.getElementById("leagueRecommendationSection");

const showLeagueButton =
    document.getElementById("showLeagueButton");

const leagueForm =
    document.getElementById("leagueForm");

const countryInput =
    document.getElementById("countryInput");

const recommendLeagueButton =
    document.getElementById("recommendLeagueButton");

const leagueLoadingState =
    document.getElementById("leagueLoadingState");

const leagueErrorState =
    document.getElementById("leagueErrorState");

const leagueErrorMessage =
    document.getElementById("leagueErrorMessage");

const leagueResults =
    document.getElementById("leagueResults");

const leagueResultTitle =
    document.getElementById("leagueResultTitle");

const leagueResultDescription =
    document.getElementById("leagueResultDescription");

const leagueTableBody =
    document.getElementById("leagueTableBody");

// ==========================================
// FORMATEAR ERRORES DE LA API
// ==========================================

function obtenerMensajeError(error) {

    if (typeof error === "string") {
        return error;
    }

    if (Array.isArray(error)) {

        return error.map(item => {

            if (typeof item === "string") {
                return item;
            }

            if (item && typeof item === "object") {

                const ubicacion = Array.isArray(item.loc)
                    ? item.loc.join(" → ")
                    : "";

                const mensaje = item.msg || item.message || JSON.stringify(item);

                return ubicacion
                    ? `${ubicacion}: ${mensaje}`
                    : mensaje;

            }

            return String(item);

        }).join(" | ");

    }

    if (error && typeof error === "object") {

        return error.msg ||
            error.message ||
            JSON.stringify(error);

    }

    return "Ocurrió un error inesperado.";

}

// ==========================================
// LEER RESPUESTA DE LA API
// ==========================================

async function leerRespuesta(respuesta) {

    const data = await respuesta.json();

    if (!respuesta.ok) {

        console.error("Respuesta de error de la API:", data);

        throw new Error(
            obtenerMensajeError(data.detail || data)
        );

    }

    return data;

}

// ==========================================
// CONTROL DE ESTADOS DE CLASIFICACIÓN
// ==========================================

function ocultarEstados() {

    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    errorState.classList.add("hidden");
    predictionResult.classList.add("hidden");

}

function mostrarCarga() {

    ocultarEstados();

    leagueRecommendationSection.classList.add("hidden");

    loadingState.classList.remove("hidden");

    analyzeButton.disabled = true;

    analyzeButton.querySelector("span:first-child").textContent =
        "Analizando...";

}

function mostrarError(mensaje) {

    ocultarEstados();

    errorMessage.textContent = mensaje;

    errorState.classList.remove("hidden");

}

function restaurarBoton() {

    analyzeButton.disabled = false;

    analyzeButton.querySelector("span:first-child").textContent =
        "Analizar jugador";

}

// ==========================================
// OBTENER DATOS DEL FORMULARIO
// ==========================================

function obtenerDatosJugador() {

    const altura = document.getElementById("height").value;

    return {

        nombre: document.getElementById("playerName").value.trim(),

        position: positionSelect.value.trim(),

        sub_position: subPositionSelect.value.trim() || "Unknown",

        height_in_cm: altura === "" ? null : Number(altura),

        age: Number(document.getElementById("age").value),

        foot: document.getElementById("foot").value

    };

}

// ==========================================
// MOSTRAR RESULTADO DE CLASIFICACIÓN
// ==========================================

function mostrarResultado(data) {

    ocultarEstados();

    const nombreJugador = obtenerDatosJugador().nombre;

    const esElite = Number(data.prediccion) === 1;

    const probabilidad = Number(data.probabilidad_elite);

    categoryBadge.textContent = esElite ? "ÉLITE" : "ESTÁNDAR";

    categoryBadge.classList.toggle("standard", !esElite);

    probabilityValue.textContent = `${probabilidad.toFixed(2)}%`;

    probabilityBar.style.width =
        `${Math.max(0, Math.min(100, probabilidad))}%`;

    resultTitle.textContent = `${nombreJugador} - ${
        esElite
            ? "Perfil clasificado como Élite"
            : "Perfil clasificado como Estándar"
    }`;

    resultMessage.textContent = data.mensaje;

    predictionResult.classList.remove("hidden");

    reiniciarRecomendacion();

    leagueRecommendationSection.classList.remove("hidden");

}

// ==========================================
// RESTABLECER RECOMENDACIÓN DE LIGAS
// ==========================================

function reiniciarRecomendacion() {

    leagueForm.classList.add("hidden");

    showLeagueButton.classList.remove("hidden");

    leagueLoadingState.classList.add("hidden");

    leagueErrorState.classList.add("hidden");

    leagueResults.classList.add("hidden");

    countryInput.value = "";

    leagueTableBody.innerHTML = "";

    recommendLeagueButton.disabled = false;

    recommendLeagueButton.querySelector("span:first-child").textContent =
        "Predecir probabilidades mejores ligas";

}

// ==========================================
// MOSTRAR FORMULARIO DE PAÍS
// ==========================================

showLeagueButton.addEventListener("click", () => {

    showLeagueButton.classList.add("hidden");

    leagueForm.classList.remove("hidden");

});

// ==========================================
// MOSTRAR ERROR DE RECOMENDACIÓN
// ==========================================

function mostrarErrorLigas(mensaje) {

    leagueLoadingState.classList.add("hidden");

    leagueResults.classList.add("hidden");

    leagueErrorMessage.textContent = obtenerMensajeError(mensaje);

    leagueErrorState.classList.remove("hidden");

}

// ==========================================
// MOSTRAR RESULTADOS DE RECOMENDACIÓN
// ==========================================

function mostrarResultadosLigas(data) {

    leagueLoadingState.classList.add("hidden");

    leagueErrorState.classList.add("hidden");

    leagueTableBody.innerHTML = "";

    const recomendaciones = data.recomendaciones || [];

    if (!Array.isArray(recomendaciones) || recomendaciones.length === 0) {

        mostrarErrorLigas(
            "No se encontraron recomendaciones para este país."
        );

        return;

    }

    leagueResultTitle.textContent =
        `Mejores ligas para jugadores de ${data.pais_nacimiento}`;

    leagueResultDescription.textContent =
        `Se encontraron ${data.total_ligas} ligas candidatas.`;

    recomendaciones.forEach((liga, indice) => {

        const fila = document.createElement("tr");

        const celdaPosicion = document.createElement("td");
        const celdaLiga = document.createElement("td");
        const celdaPais = document.createElement("td");
        const celdaProbabilidad = document.createElement("td");

        celdaPosicion.textContent = indice + 1;

        celdaLiga.textContent = liga.liga ?? "Sin información";

        celdaPais.textContent = liga.pais_liga ?? "Sin información";

        const probabilidad = Number(liga.probabilidad_elite);

        celdaProbabilidad.textContent =
            Number.isFinite(probabilidad)
                ? `${probabilidad.toFixed(2)}%`
                : "No disponible";

        fila.appendChild(celdaPosicion);
        fila.appendChild(celdaLiga);
        fila.appendChild(celdaPais);
        fila.appendChild(celdaProbabilidad);

        leagueTableBody.appendChild(fila);

    });

    leagueResults.classList.remove("hidden");

}

// ==========================================
// RECOMENDACIÓN DE LIGAS SEGÚN PAÍS
// ==========================================

recommendLeagueButton.addEventListener("click", async () => {

    const pais = countryInput.value.trim();

    if (!pais) {

        mostrarErrorLigas(
            "Introduce un país de nacimiento para continuar."
        );

        return;

    }

    leagueLoadingState.classList.remove("hidden");

    leagueErrorState.classList.add("hidden");

    leagueResults.classList.add("hidden");

    recommendLeagueButton.disabled = true;

    recommendLeagueButton.querySelector("span:first-child").textContent =
        "Consultando...";

    try {

        const respuesta = await fetch(`${API_URL}/recommend-leagues`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                tipo: "recomendacion_ligas",

                pais_nacimiento: pais

            })

        });

        const data = await leerRespuesta(respuesta);

        mostrarResultadosLigas(data);

    } catch (error) {

        console.error("Error al recomendar ligas:", error);

        mostrarErrorLigas(
            error.message.includes("Failed to fetch")
                ? "No se pudo conectar con el servidor. Comprueba que el backend esté funcionando."
                : error.message
        );

    } finally {

        recommendLeagueButton.disabled = false;

        recommendLeagueButton.querySelector("span:first-child").textContent =
            "Predecir probabilidades mejores ligas";

    }

});

// ==========================================
// ENVIAR SOLICITUD DE CLASIFICACIÓN
// ==========================================

form.addEventListener("submit", async (event) => {

    event.preventDefault();

    const jugador = obtenerDatosJugador();

    // Validación original del primer modelo
    if (
        !jugador.position ||
        !Number.isFinite(jugador.age) ||
        jugador.age < 18 ||
        jugador.age > 45
    ) {

        mostrarError(
            "Verifica la posición y la edad del jugador."
        );

        return;

    }

    if (
        jugador.height_in_cm !== null &&
        (
            !Number.isFinite(jugador.height_in_cm) ||
            jugador.height_in_cm <= 160 ||
            jugador.height_in_cm > 250
        )
    ) {

        mostrarError(
            "Introduce una altura válida entre 1 y 250 cm."
        );

        return;

    }

    mostrarCarga();

    try {

        const respuesta = await fetch(`${API_URL}/predict`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                tipo: "clasificacion",

                position: jugador.position,

                sub_position: jugador.sub_position,

                height_in_cm: jugador.height_in_cm,

                age: jugador.age,

                foot: jugador.foot

            })

        });

        const data = await leerRespuesta(respuesta);

        mostrarResultado(data);

    } catch (error) {

        console.error("Error de comunicación:", error);

        mostrarError(
            error.message.includes("Failed to fetch")
                ? "No se pudo conectar con el servidor. Comprueba que el backend esté funcionando."
                : error.message
        );

    } finally {

        restaurarBoton();

    }

});

console.log("ScoutVision iniciado correctamente.");
