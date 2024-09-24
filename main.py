import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import plotly.express as px  # Importación de Plotly Express
import plotly.graph_objects as go

# Función para obtener la lista de archivos de un repositorio en GitHub usando la API
@st.cache_data
def obtener_lista_archivos_github(repo_url, subdirectorio=""):
    api_url = repo_url.replace("github.com", "api.github.com/repos") + f"/contents/{subdirectorio}"
    response = requests.get(api_url)
    if response.status_code == 200:
        archivos = [archivo['download_url'] for archivo in response.json() if archivo['name'].endswith(".snana.dat")]
        return archivos
    else:
        st.write(f"Error al obtener la lista de archivos de {repo_url}")
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
    mjd, mag, magerr, flx, flxerr, filtros = [], [], [], [], [], []
    snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv = None, None, None, None, None, None, None

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
        elif linea.startswith("PARSNIP_PRED:"):
            parsnip_pred = ' '.join(linea.split()[1:])
        elif linea.startswith("SUPERRAENN_PRED:"):
            superraenn_pred = ' '.join(linea.split()[1:])
        elif linea.startswith("OBS:"):
            datos = linea.split()
            mjd.append(convertir_a_float(datos[1]))  # MJD
            filtros.append(datos[2])  # Filtro (g, r, i, z, etc.)
            flx.append(convertir_a_float(datos[4]))  # Flujo
            flxerr.append(convertir_a_float(datos[5]))  # Error en el flujo
            mag.append(convertir_a_float(datos[6]))  # Magnitud
            magerr.append(convertir_a_float(datos[7]))  # Error en la magnitud

    return mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv

# Función para guardar las curvas de luz como un DataFrame
def guardar_curvas_como_vectores(lista_vectores, nombre_archivo, mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv):
    for i in range(len(mjd)):
        curva_vector = {
            'nombre_archivo': nombre_archivo,
            'snid': snid,
            'mjd': mjd[i],
            'filtro': filtros[i] if filtros else None,
            'mag': mag[i],
            'magerr': magerr[i],
            'flx': flx[i],
            'flxerr': flxerr[i],
            'parsnip_pred': parsnip_pred,
            'superraenn_pred': superraenn_pred,
            'ra': ra,
            'decl': decl,
            'redshift': redshift,
            'mwebv': mwebv
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
            mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv = leer_archivo_supernova_contenido(contenido)
            guardar_curvas_como_vectores(lista_vectores, nombre_archivo, mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv)

    df = pd.DataFrame(lista_vectores)
    return df

# Cargar los datos de supernovas desde GitHub
repo_url = "https://github.com/SArcD/supernovaIA"
df_curvas_luz = descargar_y_procesar_supernovas(repo_url)

# Crear el gráfico de posiciones de supernovas
def crear_grafico_posiciones():
    fig = px.scatter_polar(df_curvas_luz, r='redshift', theta='ra', color='parsnip_pred',
                           hover_data=['snid', 'redshift'], title='Posiciones Polares de Supernovas')
    return fig

# Mostrar el gráfico de posiciones en Streamlit
st.plotly_chart(crear_grafico_posiciones())

# Seleccionar supernova
snid_seleccionado = st.selectbox("Selecciona una supernova para ver su curva de luz:", df_curvas_luz['snid'].unique())

# Función para calcular días relativos al pico
def calcular_dias_relativos_con_pico(df_supernova, nobs_before_peak, nobs_to_peak):
    if df_supernova['mag'].isnull().all() or nobs_to_peak is None:
        return df_supernova['mjd']
    mjd_pico = df_supernova['mjd'].iloc[nobs_before_peak]
    df_supernova['dias_relativos'] = df_supernova['mjd'] - mjd_pico
    return df_supernova

# Función para graficar la curva de luz
def graficar_curva_de_luz(df_supernova, nobs_before_peak, nobs_to_peak):
    # Verificar si la columna 'filtro' existe y si tiene datos válidos
    if 'filtro' not in df_supernova.columns or df_supernova['filtro'].isnull().all():
        st.warning(f"La supernova {df_supernova['snid'].iloc[0]} no tiene datos en la columna 'filtro'.")
        return go.Figure()

    # Calcular días relativos al pico usando la información de observaciones antes del pico
    df_supernova = calcular_dias_relativos_con_pico(df_supernova, nobs_before_peak, nobs_to_peak)
    
    fig = go.Figure()

    # Graficar solo los filtros que tienen datos
    for filtro in df_supernova['filtro'].dropna().unique():
        df_filtro = df_supernova[df_supernova['filtro'] == filtro]
        fig.add_trace(go.Scatter(
            x=df_filtro['dias_relativos'],
            y=df_filtro['mag'],
            mode='lines+markers',
            name=filtro
        ))

    # Invertir el eje Y porque las magnitudes menores son más brillantes
    fig.update_layout(
        title=f'Curva de luz de {df_supernova["snid"].iloc[0]} (días relativos al pico)',
        xaxis_title='Días relativos al pico de luminosidad',
        yaxis_title='Magnitud',
        yaxis=dict(autorange='reversed'),
        showlegend=True
    )
    return fig

# Filtrar los datos de la supernova seleccionada
df_supernova_seleccionada = df_curvas_luz[df_curvas_luz['snid'] == snid_seleccionado]
nobs_before_peak = df_supernova_seleccionada['nobs_before_peak'].iloc[0] if 'nobs_before_peak' in df_supernova_seleccionada.columns else None
nobs_to_peak = df_supernova_seleccionada['nobs_to_peak'].iloc[0] if 'nobs_to_peak' in df_supernova_seleccionada.columns else None
st.plotly_chart(graficar_curva_de_luz(df_supernova_seleccionada, nobs_before_peak, nobs_to_peak))

# Filtrar supernovas por tipo y observaciones
tipo_supernova = st.text_input("Ingresa el tipo de supernova (ej. 'SN Ia', 'SN Ib', 'SN II'):")
min_observaciones = st.number_input("Especifica el número mínimo de observaciones:", min_value=1, value=5)
df_supernovas_filtradas = df_curvas_luz[df_curvas_luz['parsnip_pred'] == tipo_supernova].groupby('snid').filter(lambda x: len(x) >= min_observaciones)

# Graficar todas las supernovas filtradas
if not df_supernovas_filtradas.empty:
    for snid in df_supernovas_filtradas['snid'].unique():
        df_supernova_seleccionada = df_supernovas_filtradas[df_supernovas_filtradas['snid'] == snid]
        nobs_before_peak = df_supernova_seleccionada['nobs_before_peak'].iloc[0] if 'nobs_before_peak' in df_supernova_seleccionada.columns else None
        nobs_to_peak = df_supernova_seleccionada['nobs_to_peak'].iloc[0] if 'nobs_to_peak' in df_supernova_seleccionada.columns else None
        st.plotly_chart(graficar_curva_de_luz(df_supernova_seleccionada, nobs_before_peak, nobs_to_peak))
else:
    st.write(f"No se encontraron supernovas del tipo '{tipo_supernova}' con al menos {min_observaciones} observaciones.")
