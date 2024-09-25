import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Función para obtener la lista de archivos de un repositorio en GitHub usando la API
@st.cache_data
def obtener_lista_archivos_github(repo_url, subdirectorio=""):
    api_url = repo_url.replace("github.com", "api.github.com/repos") + f"/contents/{subdirectorio}"
    response = requests.get(api_url)
    if response.status_code == 200:
        archivos = [archivo['download_url'] for archivo in response.json() if archivo['name'].endswith(".snana.dat")]
        return archivos
    else:
        return []

# Función para descargar y leer el contenido de un archivo desde GitHub
@st.cache_data
def descargar_archivo_desde_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Función para intentar convertir un valor a float de forma segura
def convertir_a_float(valor, valor_default=None):
    try:
        return float(valor)
    except ValueError:
        return valor_default

# Función para leer el archivo descargado y extraer los datos relevantes
def leer_archivo_supernova_contenido(contenido):
    # Variables y listas para almacenar los datos
    mjd, mag, magerr, flx, flxerr, filtros = [], [], [], [], [], []
    snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv = None, None, None, None, None, None, None
    observaciones_antes_pico, observaciones_pico, observaciones_despues_pico = None, None, None

    # Procesar línea por línea el contenido del archivo
    for linea in contenido.splitlines():
        if linea.startswith("SNID:"):
            snid = linea.split()[1]
        elif linea.startswith("RA:"):
            ra = convertir_a_float(linea.split()[1])
        elif linea.startswith("DECL:"):
            decl = convertir_a_float(linea.split()[1])
        elif linea.startswith("REDSHIFT_FINAL:"):
            redshift = convertir_a_float(linea.split()[1])
        elif linea.startswith("MWEBV:"):
            mwebv = convertir_a_float(linea.split()[1])
        elif linea.startswith("NOBS_BEFORE_PEAK:"):
            observaciones_antes_pico = convertir_a_float(linea.split()[1])
        elif linea.startswith("NOBS_TO_PEAK:"):
            observaciones_pico = convertir_a_float(linea.split()[1])
        elif linea.startswith("NOBS_AFTER_PEAK:"):
            observaciones_despues_pico = convertir_a_float(linea.split()[1])
        elif linea.startswith("PARSNIP_PRED:"):
            parsnip_pred = ' '.join(linea.split()[1:])
        elif linea.startswith("SUPERRAENN_PRED:"):
            superraenn_pred = ' '.join(linea.split()[1:])
        elif linea.startswith("OBS:"):  # Extraer observaciones
            datos = linea.split()
            mjd.append(convertir_a_float(datos[1]))  # MJD (Modified Julian Date)
            filtros.append(datos[2])     # Filtro (g, r, i, z, etc.)
            flx.append(convertir_a_float(datos[4]))  # Flujo (FLUXCAL)
            flxerr.append(convertir_a_float(datos[5]))  # Error en el flujo (FLUXCALERR)
            mag.append(convertir_a_float(datos[6]))  # Magnitud (MAG)
            magerr.append(convertir_a_float(datos[7]))  # Error en la magnitud (MAGERR)

    return mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observaciones_antes_pico, observaciones_pico, observaciones_despues_pico

# Función para guardar las curvas de luz como un DataFrame
def guardar_curvas_como_vectores(lista_vectores, nombre_archivo, mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observaciones_antes_pico, observaciones_pico, observaciones_despues_pico):
    for i in range(len(mjd)):
        curva_vector = {
            'nombre_archivo': nombre_archivo,
            'snid': snid,
            'mjd': mjd[i],
            'filtro': filtros[i],
            'mag': mag[i],
            'magerr': magerr[i],
            'flx': flx[i],
            'flxerr': flxerr[i],
            'parsnip_pred': parsnip_pred,
            'superraenn_pred': superraenn_pred,
            'ra': ra,
            'decl': decl,
            'redshift': redshift,
            'mwebv': mwebv,
            'observaciones_antes_pico': observaciones_antes_pico,
            'observaciones_pico': observaciones_pico,
            'observaciones_despues_pico': observaciones_despues_pico
        }
        lista_vectores.append(curva_vector)

# Descargar y procesar los archivos de supernovas desde GitHub
@st.cache_data
def descargar_y_procesar_supernovas(repo_url, subdirectorio=""):
    lista_archivos = obtener_lista_archivos_github(repo_url, subdirectorio)
    lista_vectores = []

    for archivo_url in lista_archivos:
        nombre_archivo = archivo_url.split("/")[-1]
        contenido = descargar_archivo_desde_github(archivo_url)

        if contenido:
            mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observaciones_antes_pico, observaciones_pico, observaciones_despues_pico = leer_archivo_supernova_contenido(contenido)
            guardar_curvas_como_vectores(lista_vectores, nombre_archivo, mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observaciones_antes_pico, observaciones_pico, observaciones_despues_pico)

    return pd.DataFrame(lista_vectores)

# Cargar los datos de supernovas desde GitHub
st.write("Descargando y procesando archivos de supernovas...")
repo_url = "https://github.com/SArcD/supernovaIA"
df_curvas_luz = descargar_y_procesar_supernovas(repo_url)

# Guardar los datos en un archivo CSV
df_curvas_luz.to_csv('curvas_de_luz_con_parsnip_y_ra_decl_redshift_snid.csv', index=False)
st.write("Datos guardados en 'curvas_de_luz_con_parsnip_y_ra_decl_redshift_snid.csv'.")

# Función para calcular días relativos al pico de luminosidad
def calcular_dias_relativos(df_supernova):
    # Calcular el MJD del pico de luminosidad (mínima magnitud)
    mjd_pico = df_supernova.loc[df_supernova['mag'].idxmin(), 'mjd']
    df_supernova['dias_relativos'] = df_supernova['mjd'] - mjd_pico
    return df_supernova

# Función para graficar la curva de luz de una supernova específica con información en el título
def graficar_curva_de_luz(df_supernova):
    fig = go.Figure()

    # Calcular días relativos al pico de luminosidad
    df_supernova = calcular_dias_relativos(df_supernova)

    for filtro in df_supernova['filtro'].unique():
        df_filtro = df_supernova[df_supernova['filtro'] == filtro]
        fig.add_trace(go.Scatter(
            x=df_filtro['dias_relativos'],  # Usar días relativos al pico como eje X
            y=df_filtro['mag'],
            mode='lines+markers',
            name=filtro
        ))

    # Extraer la información relevante para el título
    snid = df_supernova['snid'].iloc[0]
    tipo_supernova = df_supernova['parsnip_pred'].iloc[0]
    ra = df_supernova['ra'].iloc[0]
    decl = df_supernova['decl'].iloc[0]
    redshift = df_supernova['redshift'].iloc[0]
    observaciones_antes_pico = df_supernova['observaciones_antes_pico'].iloc[0]
    observaciones_pico = df_supernova['observaciones_pico'].iloc[0]
    observaciones_despues_pico = df_supernova['observaciones_despues_pico'].iloc[0]

    # Invertir el eje Y porque las magnitudes menores son más brillantes y añadir la información al título
    fig.update_layout(
        title=(
            f'Curva de luz de {snid} ({tipo_supernova})\n'
            f'RA: {ra}°, Dec: {decl}°, Redshift: {redshift} - '
            f'Antes del pico: {observaciones_antes_pico}, Pico: {observaciones_pico}, Después del pico: {observaciones_despues_pico}'
        ),
        xaxis_title='Días relativos al pico de luminosidad',
        yaxis_title='Magnitud',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
        showlegend=True
    )

    return fig

# Seleccionar supernova
snid_seleccionado = st.selectbox("Selecciona una supernova para ver su curva de luz:", df_curvas_luz['snid'].unique())

# Filtrar los datos de la supernova seleccionada y mostrar la curva de luz
df_supernova_seleccionada = df_curvas_luz[df_curvas_luz['snid'] == snid_seleccionado]
st.plotly_chart(graficar_curva_de_luz(df_supernova_seleccionada))

# --- NUEVA FUNCIONALIDAD ---

# Caja de texto para especificar el tipo de supernova
tipo_supernova = st.text_input("Ingresa el tipo de supernova (ej. 'SN Ia', 'SN Ib', 'SN II'):")

# Entrada para el número mínimo de observaciones
min_observaciones = st.number_input("Especifica el número mínimo de observaciones:", min_value=1, value=5)

# Función para filtrar supernovas por tipo y número mínimo de observaciones
def filtrar_supernovas_por_tipo(df, tipo_supernova, min_observaciones):
    # Filtrar por tipo de supernova (PARSNIP_PRED)
    df_filtrado = df[df['parsnip_pred'] == tipo_supernova]

    # Agrupar por SNID y contar el número de observaciones por supernova
    supernovas_con_observaciones = df_filtrado.groupby('snid').filter(lambda x: len(x) >= min_observaciones)
    
    return supernovas_con_observaciones

# Filtrar las supernovas por el tipo y número mínimo de observaciones
df_supernovas_filtradas = filtrar_supernovas_por_tipo(df_curvas_luz, tipo_supernova, min_observaciones)

# Mostrar los resultados si hay supernovas que cumplan con los criterios
if not df_supernovas_filtradas.empty:
    supernovas_filtradas_por_snid = df_supernovas_filtradas['snid'].unique()
    st.write(f"Se encontraron {len(supernovas_filtradas_por_snid)} supernovas del tipo '{tipo_supernova}' con al menos {min_observaciones} observaciones.")
    
    # Graficar todas las supernovas que cumplan con los criterios
    for snid in supernovas_filtradas_por_snid:
        st.write(f"Graficando la supernova: {snid}")
        df_supernova_seleccionada = df_supernovas_filtradas[df_supernovas_filtradas['snid'] == snid]
        st.plotly_chart(graficar_curva_de_luz(df_supernova_seleccionada))

else:
    st.write(f"No se encontraron supernovas del tipo '{tipo_supernova}' con al menos {min_observaciones} observaciones.")



# --- CLUSTERING JERÁRQUICO CON PCA ---
st.write("Clustering de supernovas usando PCA")

# Entrada para el tipo de supernova
tipo_supernova = st.text_input("Ingresar el tipo de supernova (ej. 'SN Ia', 'SN Ib', 'SN II'):")

# Entrada para el número mínimo de observaciones
min_observaciones = st.number_input("El número mínimo de observaciones:", min_value=1, value=5)

# Filtrar supernovas por tipo y número mínimo de observaciones
df_supernovas_filtradas = df_curvas_luz[df_curvas_luz['parsnip_pred'] == tipo_supernova]
df_supernovas_filtradas = df_supernovas_filtradas.groupby('snid').filter(lambda x: len(x) >= min_observaciones)

# Normalizar las características para el clustering
scaler = StandardScaler()
features = ['ra', 'decl', 'redshift']  # Parámetros que se usarán para el clustering
X = scaler.fit_transform(df_supernovas_filtradas[features])

# Aplicar PCA para reducir las dimensiones a 2 componentes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Convertir el resultado de PCA en un DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['snid'] = df_supernovas_filtradas['snid'].values
df_pca['parsnip_pred'] = df_supernovas_filtradas['parsnip_pred'].values

# Graficar los resultados de PCA
fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='parsnip_pred', hover_data=['snid'], title='Clustering de Supernovas con PCA')
st.plotly_chart(fig_pca)
