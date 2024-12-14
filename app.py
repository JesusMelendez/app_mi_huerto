# UrbanGrow Lite MVP

# Importar librerías necesarias
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from fpdf import FPDF
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import geopandas as gpd
import contextily as ctx
from sklearn.decomposition import PCA
# ------------------------
# 1. Simulación de captación de agua de lluvia
# ------------------------
def calcular_captacion_agua(area_m2, precipitacion_mm):
    """
    Calcula el volumen de agua recolectada en litros.
    :param area_m2: área del techo en m2
    :param precipitacion_mm: precipitación promedio en mm
    :return: volumen en litros
    """
    volumen_litros = area_m2 * (precipitacion_mm / 1000) * 1000
    return volumen_litros

# ------------------------
# 2. Costos estimados
# ------------------------
def calcular_costos():
    costos = {
        "Capacitación": 2000,
        "Asesoría técnica": 1500,
        "Materiales iniciales": 5000,
        "Mantenimiento anual": 1000
    }
    return costos, sum(costos.values())

# ------------------------
# 3. Base de datos relacional
# ------------------------
def crear_base_datos():
    """Crea una base de datos SQLite con tablas simples."""
    conn = sqlite3.connect("urbangrow.db")
    cursor = conn.cursor()

    # Crear tablas
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Plantas (
        id INTEGER PRIMARY KEY,
        nombre TEXT UNIQUE,
        tipo TEXT,
        requerimientos_agua TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Recursos (
        id INTEGER PRIMARY KEY,
        nombre TEXT UNIQUE,
        cantidad TEXT
    )
    """)

    # Insertar datos simulados si no existen
    cursor.execute("INSERT OR IGNORE INTO Plantas (nombre, tipo, requerimientos_agua) VALUES (?, ?, ?)",
                   ("Lechuga", "Hoja verde", "Moderado"))
    cursor.execute("INSERT OR IGNORE INTO Plantas (nombre, tipo, requerimientos_agua) VALUES (?, ?, ?)",
                   ("Tomate", "Fruto", "Alto"))

    cursor.execute("INSERT OR IGNORE INTO Recursos (nombre, cantidad) VALUES (?, ?)",
                   ("Agua recolectada", "0 L"))
    cursor.execute("INSERT OR IGNORE INTO Recursos (nombre, cantidad) VALUES (?, ?)",
                   ("Compost", "10 kg"))

    conn.commit()
    conn.close()

crear_base_datos()

# ------------------------
# 4. Reporte de calidad del agua y mapa de la CDMX
# ------------------------
def generar_reporte_calidad_agua_y_mapa():
    """Genera un reporte PDF con gráficos simulados y un mapa de la CDMX."""
    # Simulación de datos
    calidad_agua = {
        "pH": np.random.uniform(6.5, 7.5, 10),
        "Contaminantes": np.random.uniform(0, 50, 10)
    }

    # Crear gráficos
    plt.figure()
    plt.plot(calidad_agua["pH"], label="pH")
    plt.plot(calidad_agua["Contaminantes"], label="Contaminantes")
    plt.legend()
    plt.savefig("calidad_agua.png")

    # Crear un mapa de la CDMX con datos simulados
   


    # URL del dataset de países

    # Descargar el archivo

 

    # Filtrar CDMX

    # Leer el dataset
  




    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Reporte de Calidad del Agua y Mapa", ln=True, align="C")
    pdf.image("calidad_agua.png", x=10, y=30, w=190)
    pdf.output("reporte_calidad_agua_mapa.pdf")

    return "reporte_calidad_agua_mapa.pdf"

# ------------------------
# 5. Análisis de factibilidad con Machine Learning
# ------------------------
def generar_analisis_factibilidad():
    """Genera un gráfico 3D de factibilidad usando SVM y regresión logística."""
    # Datos simulados
    X, y = make_classification(n_features=3, n_informative=3, n_redundant=0, n_classes=2, n_samples=100)

    # Reducir dimensionalidad
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Modelos
    svm_model = SVC(kernel='linear', probability=True).fit(X_pca, y)
    lr_model = LogisticRegression().fit(X_pca, y)

    # Predicciones
    pred_svm = svm_model.predict_proba(X_pca)[:, 1]
    pred_lr = lr_model.predict_proba(X_pca)[:, 1]

    # Graficar
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], pred_svm, c=y, label="SVM", alpha=0.7)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], pred_lr, c=y, label="Logistic Regression", marker='x', alpha=0.7)
    ax.set_title("Análisis de Factibilidad")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("Probabilidad")
    plt.legend()
    plt.savefig("factibilidad_3d.png")

    return "factibilidad_3d.png"

factibilidad_imagen = generar_analisis_factibilidad()

# ------------------------
# 6. Interfaz Streamlit
# ------------------------
with st.sidebar:
    st.title("Mi huertito")
    st.write("Explora las opciones de tu huerto urbano")

st.header("Resultados del análisis")

# Inputs de usuario
area_input = st.number_input("Área del techo (m2):", min_value=1, value=50)
precipitacion_input = st.number_input("Precipitación anual promedio (mm):", min_value=1, value=800)

# Cálculo de captación de agua
volumen_recolectado = calcular_captacion_agua(area_input, precipitacion_input)
st.subheader("1. Captación de Agua de Lluvia")
st.write(f"Volumen estimado recolectado: {volumen_recolectado:.2f} litros")

# Mostrar costos
costos, costo_total = calcular_costos()
st.subheader("2. Costos Estimados")
st.write(pd.DataFrame.from_dict(costos, orient="index", columns=["Costo (MXN)"]))
st.write(f"**Costo Total:** {costo_total} MXN")

# Mostrar base de datos
st.subheader("3. Base de Datos")
conn = sqlite3.connect("urbangrow.db")
st.write(pd.read_sql_query("SELECT * FROM Plantas", conn))
st.write(pd.read_sql_query("SELECT * FROM Recursos", conn))

# Descargar reporte


# Mostrar factibilidad
st.subheader("5. Factibilidad del Proyecto")
st.image(factibilidad_imagen, caption="Análisis de Factibilidad 3D")
