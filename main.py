import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Función para obtener la lista de archivos de un repositorio en GitHub usando la API
@st.cache_data
def obtener_lista_archivos_github(repo_url, subdirectorio=""):
    api_url = repo_url.replace("github.com", "api.github.com/repos") + f"/contents/{subdirectorio}"
    st.write(f"Obteniendo lista de archivos desde: {api_url}")

    response = requests.get(api_url)
    if response.status_code == 200:
        archivos = [archivo['download_url'] for archivo in response.json() if archivo['name'].endswith(".snana.dat")]
        st.write(f"Se encontraron {len(archivos)} archivos .snana.dat en {subdirectorio}")
        return archivos
    else:
        st.write(f"Error al obtener la lista de archivos de {repo_url}")
        st.write(f"Código de error: {response.status_code}")
        return []


# Función para descargar y leer el contenido de un archivo desde GitHub
@st.cache_data
def descargar_archivo_desde_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Función para leer el archivo descargado y extraer los datos relevantes
def leer_archivo_supernova_contenido(contenido):
    # Variables y listas para almacenar los datos
    mjd, mag, magerr, flx, flxerr, filtros = [], [], [], [], [], []
    snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv = None, None, None, None, None, None, None

    for linea in contenido.splitlines():
        if linea.startswith("SNID:"):
            snid = linea.split()[1]
        elif linea.startswith("RA:"):
            ra = float(linea.split()[1])
        elif linea.startswith("DECL:"):
            decl = float(linea.split()[1])
        elif linea.startswith("REDSHIFT_FINAL:"):
            redshift = float(linea.split()[1])
        elif linea.startswith("MWEBV:"):
            mwebv = float(linea.split()[1])
        elif linea.startswith("PARSNIP_PRED:"):
            parsnip_pred = ' '.join(linea.split()[1:])
        elif linea.startswith("SUPERRAENN_PRED:"):
            superraenn_pred = ' '.join(linea.split()[1:])
        elif linea.startswith("OBS:"):
            datos = linea.split()
            mjd.append(float(datos[1]))
            filtros.append(datos[2])
            flx.append(float(datos[4]))
            flxerr.append(float(datos[5]))
            mag.append(float(datos[6]))
            magerr.append(float(datos[7]))

    return mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv

# Función para guardar las curvas de luz como un DataFrame
def guardar_curvas_como_vectores(lista_vectores, nombre_archivo, mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv):
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

    return pd.DataFrame(lista_vectores)

# Cargar los datos de supernovas desde GitHub
repo_url = "https://api.github.com/repos/SArcD/supernovaIA/contents/"
df_curvas_luz = descargar_y_procesar_supernovas(repo_url)

# Guardar los datos en un archivo CSV
df_curvas_luz.to_csv('curvas_de_luz_con_parsnip_y_ra_decl_redshift_snid.csv', index=False)

# Mostrar el gráfico de posiciones en Streamlit
def crear_grafico_posiciones():
    fig = px.scatter_polar(df_curvas_luz, r='redshift', theta='ra', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Posiciones Polares de Supernovas')
    return fig

st.plotly_chart(crear_grafico_posiciones())

# Seleccionar supernova
snid_seleccionado = st.selectbox("Selecciona una supernova para ver su curva de luz:", df_curvas_luz['snid'].unique())

# Función para graficar la curva de luz de una supernova específica
def graficar_curva_de_luz(df_supernova):
    fig = go.Figure()
    for filtro in df_supernova['filtro'].unique():
        df_filtro = df_supernova[df_supernova['filtro'] == filtro]
        fig.add_trace(go.Scatter(x=df_filtro['mjd'], y=df_filtro['mag'], mode='lines+markers', name=filtro))

    fig.update_layout(title=f'Curva de luz de {snid_seleccionado}', xaxis_title='MJD', yaxis_title='Magnitud')
    return fig

# Filtrar los datos de la supernova seleccionada y mostrar la curva de luz
df_supernova_seleccionada = df_curvas_luz[df_curvas_luz['snid'] == snid_seleccionado]
st.plotly_chart(graficar_curva_de_luz(df_supernova_seleccionada))
