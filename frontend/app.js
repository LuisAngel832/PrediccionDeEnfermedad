// El frontend habla con el backend a traves de /api (Nginx hace el proxy).
// Asi no importa en que host/puerto este el backend.
const API_URL = "/api";

const formulario = document.getElementById("formulario");
const seccionResultado = document.getElementById("resultado");
const tituloResultado = document.getElementById("resultado-titulo");
const detalleResultado = document.getElementById("resultado-detalle");
const barraRelleno = document.getElementById("barra-relleno");

formulario.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Recolectar los valores del formulario y convertirlos a numero
  const datos = {};
  new FormData(formulario).forEach((valor, clave) => {
    datos[clave] = Number(valor);
  });

  tituloResultado.textContent = "Calculando...";
  detalleResultado.textContent = "";
  seccionResultado.className = "";

  try {
    const respuesta = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(datos),
    });

    if (!respuesta.ok) {
      const err = await respuesta.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${respuesta.status}`);
    }

    const r = await respuesta.json();
    mostrarResultado(r);
  } catch (error) {
    seccionResultado.className = "";
    tituloResultado.textContent = "⚠️ Error al predecir";
    detalleResultado.textContent =
      "No se pudo contactar al backend. ¿Está corriendo? Detalle: " + error.message;
    barraRelleno.style.width = "0%";
  }
});

function mostrarResultado(r) {
  seccionResultado.className = r.tiene_riesgo ? "riesgo" : "sano";
  tituloResultado.textContent = r.tiene_riesgo
    ? "⚠️ " + r.resultado
    : "✅ " + r.resultado;
  detalleResultado.textContent =
    `Probabilidad estimada de enfermedad cardíaca: ${r.porcentaje}%`;
  barraRelleno.style.width = `${r.porcentaje}%`;
}
