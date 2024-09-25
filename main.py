import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

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

# Función para convertir a float de manera segura
def convertir_a_float(valor, valor_default=None):
    try:
        return float(valor)
    except ValueError:
        return valor_default

# Función para procesar el contenido de las supernovas
def leer_archivo_supernova_contenido(contenido):
    # Variables para almacenar datos
    mjd, mag, magerr, flx, flxerr, filtros = [], [], [], [], [], []
    snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv = None, None, None, None, None, None, None
    observaciones_antes_pico, observaciones_pico, observaciones_despues_pico = None, None, None

    # Procesar línea por línea
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
        elif linea.startswith("OBS:"):  # Extraer observaciones
            datos = linea.split()
            mjd.append(convertir_a_float(datos[1]))
            filtros.append(datos[2])
            flx.append(convertir_a_float(datos[4]))
            flxerr.append(convertir_a_float(datos[5]))
            mag.append(convertir_a_float(datos[6]))
            magerr.append(convertir_a_float(datos[7]))

    return mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv

# Función para guardar los datos como un DataFrame
def guardar_curvas_como_vectores(lista_vectores, nombre_archivo, mjd, mag, magerr, flx, flxerr, filtros, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv):
    for i in range(len(mjd)):
        curva_vector = {
            'nombre_archivo': nombre_archivo,
            'SNID': snid,
            'mjd': mjd[i],
            'filtro': filtros[i],
            'mag': mag[i],
            'magerr': magerr[i],
            'flx': flx[i],
            'flxerr': flxerr[i],
            'parsnip_pred': parsnip_pred,
            'superraenn_pred': superraenn_pred,
            'RA': ra,
            'Dec': decl,
            'Redshift': redshift,
            'MWEBV': mwebv
        }
        lista_vectores.append(curva_vector)

# Descargar y procesar supernovas
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

# Descargando y procesando supernovas
st.write("Descargando y procesando archivos de supernovas...")
repo_url = "https://github.com/SArcD/supernovaIA"
df_curvas_luz = descargar_y_procesar_supernovas(repo_url)

# Guardar los datos en un archivo CSV
df_curvas_luz.to_csv('curvas_de_luz_con_parsnip_y_ra_decl_redshift_snid.csv', index=False)
st.write("Datos guardados en 'curvas_de_luz_con_parsnip_y_ra_decl_redshift_snid.csv'.")

# Graficar curva de luz
def graficar_curva_de_luz(df_supernova):
    fig = go.Figure()
    df_supernova['dias_relativos'] = df_supernova['mjd'] - df_supernova['mjd'].min()

    for filtro in df_supernova['filtro'].unique():
        df_filtro = df_supernova[df_supernova['filtro'] == filtro]
        fig.add_trace(go.Scatter(x=df_filtro['dias_relativos'], y=df_filtro['mag'], mode='lines+markers', name=filtro))

    # Información de la supernova para el título
    snid = df_supernova['SNID'].iloc[0]
    tipo_supernova = df_supernova['parsnip_pred'].iloc[0]
    ra = df_supernova['RA'].iloc[0]
    decl = df_supernova['Dec'].iloc[0]
    redshift = df_supernova['Redshift'].iloc[0]

    fig.update_layout(
        title=f'Curva de luz de {snid} ({tipo_supernova})\nRA: {ra}°, Dec: {decl}°, Redshift: {redshift}',
        xaxis_title='Días relativos al pico de luminosidad',
        yaxis_title='Magnitud',
        yaxis=dict(autorange='reversed'),
        showlegend=True
    )

    return fig

# Análisis de Clustering
st.write(f"Se generaron {num_clusters} clusters.")
for cluster_id in range(num_clusters):
    st.write(f"Cluster {cluster_id}:")
    
    df_supernovas_cluster = df_supernovas_clustering[df_supernovas_clustering['cluster'] == cluster_id]
    supernovas_filtradas_por_snid = df_supernovas_cluster['SNID'].unique()

    if len(supernovas_filtradas_por_snid) > 0:
        st.write(f"Se encontraron {len(supernovas_filtradas_por_snid)} supernovas en el Cluster {cluster_id}.")
        
        index_seleccionado = st.slider(f'Selecciona una supernova para ver su curva de luz (Cluster {cluster_id}):',
                                       min_value=0, max_value=len(supernovas_filtradas_por_snid)-1, step=1)
        snid_seleccionado = supernovas_filtradas_por_snid[index_seleccionado]
        df_supernova_seleccionada = df_supernovas_cluster[df_supernovas_cluster['SNID'] == snid_seleccionado]
        
        if 'filtro' in df_supernova_seleccionada.columns and 'mjd' in df_supernova_seleccionada.columns and 'mag' in df_supernova_seleccionada.columns:
            fig = graficar_curva_de_luz(df_supernova_seleccionada)
            st.plotly_chart(fig)
        else:
            st.write(f"Las columnas necesarias para graficar no están presentes en los datos.")
    else:
        st.write(f"No se encontraron supernovas en el Cluster {cluster_id}.")

# Aplicar PCA y t-SNE en secuencia
pca = PCA(n_components=20)
pca_data_cluster = pca.fit_transform(columnas_numericas_scaled)
tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=10, learning_rate=200, random_state=42)
tsne_data_cluster = tsne.fit_transform(pca_data_cluster)

# Crear el gráfico de t-SNE
fig_tsne_subcluster = go.Figure()

for subcluster_id in np.unique(df_supernovas_clustering['cluster']):
    indices = df_supernovas_clustering['cluster'] == subcluster_id

    scatter_trace = go.Scatter(
        x=tsne_data_cluster[indices, 0],
        y=tsne_data_cluster[indices, 1],
        mode='markers',
        text=df_supernovas_clustering.loc[indices, ['SNID', 'RA', 'Dec', 'Redshift']].apply(lambda x: '<br>'.join(x.astype(str)), axis=1),
        marker=dict(size=7, line=dict(width=0.5, color='black')),
        name=f'Cluster {subcluster_id}'
    )
    fig_tsne_subcluster.add_trace(scatter_trace)

fig_tsne_subcluster.update_layout(
    title='Gráfico de Dispersión de t-SNE',
    xaxis_title='Dimensión 1',
    yaxis_title='Dimensión 2',
    showlegend=True
)

st.plotly_chart(fig_tsne_subcluster)
