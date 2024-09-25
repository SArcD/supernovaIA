import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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


#########3

# Crear el gráfico de posiciones de supernovas
def crear_grafico_posiciones():
    fig = px.scatter_polar(df_curvas_luz, r='redshift', theta='ra', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Posiciones Polares de Supernovas')
    return fig

# Mostrar el gráfico de posiciones en Streamlit
#st.plotly_chart(crear_grafico_posiciones())

def crear_grafico_posiciones_rectangulares():
    fig = px.scatter(df_curvas_luz,
                     x='ra',
                     y='decl',
                     color='parsnip_pred',  # Colorear por el valor de PARSNIP_PRED
                     hover_data=['snid', 'redshift', 'superraenn_pred'],  # Mostrar SNID, redshift y SUPERRAENN al pasar el cursor
                     title='Posición de las Supernovas en el Cielo (RA vs Dec)',
                     labels={'ra': 'Ascensión Recta (RA)', 'decl': 'Declinación (Dec)'}
                     #color_discrete_sequence=px.colors.sequential.Viridis  # Usar la paleta de colores Viridis
                     )
    return fig

# Crear el gráfico de posiciones Declinación vs Redshift
def crear_grafico_decl_vs_redshift():
    fig = px.scatter_polar(df_curvas_luz, r='redshift', theta='decl', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Posiciones Polares de Supernovas (Dec) vs Redshift')
    return fig




# Mostrar un selector para que el usuario elija el tipo de gráfico
opcion_grafico = st.selectbox("Selecciona el tipo de gráfico para mostrar:", ["Ascensión Recta vs Corrimiento al Rojo", "Declinación vs Corrimiento al Rojo", "Declinación vs Ascensión Recta"])

# Mostrar el gráfico según la opción seleccionada
if opcion_grafico == "Ascensión Recta vs Corrimiento al Rojo":
    st.plotly_chart(crear_grafico_posiciones())
elif opcion_grafico == "Declinación vs Ascensión Recta" :
    st.plotly_chart(crear_grafico_posiciones_rectangulares())
elif opcion_grafico == "Declinación vs Corrimiento al Rojo" :
    st.plotly_chart(crear_grafico_decl_vs_redshift())



#######


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

    # Invertir el eje Y porque las magnitudes menores son más brillantes y añadir la información al título
    fig.update_layout(
        title=(
            f'Curva de luz de {snid} ({tipo_supernova})\n'
            f'RA: {ra}°, Dec: {decl}°, Redshift: {redshift}'
        ),
        xaxis_title='Días relativos al pico de luminosidad',
        yaxis_title='Magnitud',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
        showlegend=True
    )

    return fig

# Seleccionar el tipo de supernova y el número mínimo de observaciones con un deslizador
tipo_supernova = st.text_input("Ingresa el tipo de supernova (ej. 'SN Ia', 'SN Ib', 'SN II'):")
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
    
    # Deslizador horizontal para seleccionar una supernova y mostrar su curva de luz
    index_seleccionado = st.slider('Selecciona la supernova para ver su curva de luz:', 
                                   min_value=0, max_value=len(supernovas_filtradas_por_snid)-1, step=1)
    snid_seleccionado = supernovas_filtradas_por_snid[index_seleccionado]
    df_supernova_seleccionada = df_supernovas_filtradas[df_supernovas_filtradas['snid'] == snid_seleccionado]
    st.plotly_chart(graficar_curva_de_luz(df_supernova_seleccionada))

else:
    st.write(f"No se encontraron supernovas del tipo '{tipo_supernova}' con al menos {min_observaciones} observaciones.")

# --- Mostrar DataFrame con detalles de las supernovas filtradas ---

# Función para calcular Δm15 para supernovas tipo Ia con la opción de filtro alternativo
def calcular_delta_m15(df_supernova, filtro_preferido='g', filtro_alternativo='i'):
    df_filtro = df_supernova[df_supernova['filtro'] == filtro_preferido]
    
    # Si no hay datos en el filtro preferido, usar el filtro alternativo
    if df_filtro.empty and filtro_alternativo:
        df_filtro = df_supernova[df_supernova['filtro'] == filtro_alternativo]
    
    # Si aún no hay datos, devolver NA
    if df_filtro.empty:
        return 'NA'  # No hay datos ni en el filtro preferido ni en el alternativo
    
    # Obtener el MJD del pico de luminosidad
    mjd_pico = df_filtro.loc[df_filtro['mag'].idxmin(), 'mjd']
    
    # Filtrar observaciones de 15 días después del pico
    df_15_dias_despues = df_filtro[(df_filtro['mjd'] > mjd_pico) & (df_filtro['mjd'] <= mjd_pico + 15)]
    
    if not df_15_dias_despues.empty:
        mag_pico = df_filtro.loc[df_filtro['mag'].idxmin(), 'mag']
        mag_15_dias_despues = df_15_dias_despues['mag'].mean()
        delta_m15 = mag_15_dias_despues - mag_pico
        return delta_m15
    else:
        return 'NA'

# Función para calcular la duración de la meseta para supernovas tipo II o Ibc
def calcular_duracion_meseta(df_supernova, filtro='r'):
    df_filtro = df_supernova[df_supernova['filtro'] == filtro]
    
    if df_filtro.empty:
        return 'NA'  # No hay datos en este filtro
    
    # Encontrar el MJD del pico
    mjd_pico = df_filtro.loc[df_filtro['mag'].idxmin(), 'mjd']
    
    # Definir la meseta como el tiempo entre el pico y cuando la magnitud cae en 1 o más (adaptación de la meseta)
    df_meseta = df_filtro[df_filtro['mag'] <= (df_filtro['mag'].min() + 1)]
    
    if not df_meseta.empty:
        duracion_meseta = df_meseta['mjd'].max() - mjd_pico
        return duracion_meseta
    else:
        return 'NA'


# Función para calcular la velocidad de caída de la luminosidad después de la meseta
def calcular_velocidad_caida(df_supernova, filtro='r'):
    df_filtro = df_supernova[df_supernova['filtro'] == filtro]
    
    if df_filtro.empty:
        return 'NA'
    
    # Encontrar el MJD del final de la meseta (pico)
    mjd_pico = df_filtro.loc[df_filtro['mag'].idxmin(), 'mjd']
    
    # Encontrar la última observación de la supernova
    mjd_final = df_filtro['mjd'].max()
    mag_final = df_filtro.loc[df_filtro['mjd'] == mjd_final, 'mag'].values[0]
    
    # Magnitud al final de la meseta
    mag_pico = df_filtro['mag'].min()
    
    # Calcular la velocidad de caída de la luminosidad
    velocidad_caida = (mag_final - mag_pico) / (mjd_final - mjd_pico)
    
    return velocidad_caida

# Función para calcular la magnitud promedio durante la meseta
def calcular_magnitud_meseta(df_supernova, filtro='r'):
    df_filtro = df_supernova[df_supernova['filtro'] == filtro]
    
    if df_filtro.empty:
        return 'NA'
    
    # Magnitudes durante la meseta (magnitud <= min(magnitud) + 1)
    df_meseta = df_filtro[df_filtro['mag'] <= df_filtro['mag'].min() + 1]
    
    if df_meseta.empty:
        return 'NA'
    
    # Promediar las magnitudes durante la meseta
    mag_promedio = df_meseta['mag'].mean()
    
    return mag_promedio

# Modificación en el resumen de supernovas para incluir SN II
def crear_dataframe_parametros(df_supernovas, tipo_supernova):
    resultados = []
    
    for snid in df_supernovas['snid'].unique():
        df_supernova = df_supernovas[df_supernovas['snid'] == snid]
        tipo_supernova = df_supernova['parsnip_pred'].iloc[0]
        ra = df_supernova['ra'].iloc[0]
        decl = df_supernova['decl'].iloc[0]
        redshift = df_supernova['redshift'].iloc[0]
        
        # Calcular la magnitud del pico para cada filtro disponible
        magnitudes_pico = {}
        for filtro in df_supernova['filtro'].unique():
            grupo_filtro = df_supernova[df_supernova['filtro'] == filtro]
            magnitud_pico_filtro = grupo_filtro['mag'].min() if not grupo_filtro['mag'].isnull().all() else 'NA'
            magnitudes_pico[f'magnitud_pico_{filtro}'] = magnitud_pico_filtro
        
        # Calcular la duración del evento
        mjd_pico = df_supernova.loc[df_supernova['mag'].idxmin(), 'mjd']
        duracion_evento = df_supernova['mjd'].max() - mjd_pico
        
        # Calcular parámetros específicos para SN Ia, SN II y SN Ibc
        delta_m15 = 'NA'
        duracion_meseta = 'NA'
        velocidad_caida = 'NA'
        mag_promedio_meseta = 'NA'
        
        if tipo_supernova == 'SN Ia':
            delta_m15 = calcular_delta_m15(df_supernova, filtro_preferido='g', filtro_alternativo='i')
        elif tipo_supernova in ['SN II', 'SN Ibc']:
            duracion_meseta = calcular_duracion_meseta(df_supernova, filtro='r')
            velocidad_caida = calcular_velocidad_caida(df_supernova, filtro='r')
            mag_promedio_meseta = calcular_magnitud_meseta(df_supernova, filtro='r')
        
        # Agregar los datos al resumen
        resumen = {
            'SNID': snid,
            **magnitudes_pico,  # Añadir magnitudes de pico por filtro
            'Duración del evento': duracion_evento,
            'RA': ra,
            'Dec': decl,
            'Redshift': redshift
        }
        
        # Agregar Δm15 solo si el tipo de supernova es Ia
        if tipo_supernova == 'SN Ia':
            resumen['Δm15 (g/i)'] = delta_m15
        
        # Agregar parámetros solo si el tipo de supernova es SN II o Ibc
        if tipo_supernova in ['SN II', 'SN Ibc']:
            resumen['Duración Meseta (r)'] = duracion_meseta
            resumen['Velocidad Caída (r)'] = velocidad_caida
            resumen['Magnitud Promedio Meseta (r)'] = mag_promedio_meseta
        
        resultados.append(resumen)

    return pd.DataFrame(resultados)
    
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import plotly.express as px

# Corregir la llamada a la función `crear_dataframe_parametros`
df_parametros = crear_dataframe_parametros(df_supernovas_filtradas, tipo_supernova)

# Mostrar el DataFrame para verificar que tiene datos
st.write("Verificando contenido de df_parametros:")
st.write(df_parametros)
# Eliminar filas con valores NaN
df_supernovas_clustering = df_parametros.dropna()

# Seleccionar las columnas numéricas para el clustering
columnas_numericas = df_supernovas_clustering.select_dtypes(include=['number'])

# Normalizar los datos
scaler = StandardScaler()
columnas_numericas_scaled = scaler.fit_transform(columnas_numericas)

# Clustering jerárquico
num_clusters = 5
clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
df_supernovas_clustering['cluster'] = clustering.fit_predict(columnas_numericas_scaled)

# Aplicar PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(columnas_numericas_scaled)
df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['cluster'] = df_supernovas_clustering['cluster']

# Visualización de clusters con PCA
fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='cluster', title='Clusters visualizados con PCA')
st.plotly_chart(fig_pca)

# Ajustar perplexity para t-SNE
n_samples = len(df_supernovas_clustering)
perplexity_value = min(30, n_samples // 3)

# Aplicar t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity_value, early_exaggeration=10, learning_rate=5)
tsne_data = tsne.fit_transform(pca_data)
df_tsne = pd.DataFrame(tsne_data, columns=['t-SNE1', 't-SNE2'])
df_tsne['cluster'] = df_supernovas_clustering['cluster']

# Resetear los índices para asegurar que estén alineados
df_supernovas_clustering = df_supernovas_clustering.reset_index(drop=True)
df_tsne = df_tsne.reset_index(drop=True)

# Visualización de clusters con t-SNE y curvas de densidad de kernel
fig_tsne = go.Figure()

# Agregar los contornos de densidad de kernel y los puntos
for cluster_id in np.unique(df_tsne['cluster']):
    indices = np.where(df_tsne['cluster'] == cluster_id)

    # Calcular la densidad de kernel
    kde = gaussian_kde(tsne_data[indices].T)
    x_range = np.linspace(np.min(tsne_data[:, 0])-5, np.max(tsne_data[:, 0])+5, 100)
    y_range = np.linspace(np.min(tsne_data[:, 1])-5, np.max(tsne_data[:, 1])+5, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions).T, xx.shape)

    # Añadir curvas de densidad de kernel
    contour_trace = go.Contour(
        x=x_range,
        y=y_range,
        z=zz,
        colorscale='Blues',
        opacity=0.3,
        showscale=False,
        name=f'Contour {cluster_id}'
    )
    fig_tsne.add_trace(contour_trace)

    # Añadir los puntos del scatter
    scatter_trace = go.Scatter(
        x=tsne_data[indices, 0].flatten(),
        y=tsne_data[indices, 1].flatten(),
        mode='markers',
        text=df_supernovas_clustering.loc[indices[0], ['SNID', 'RA', 'Dec', 'Redshift']].apply(lambda x: '<br>'.join(x.astype(str)), axis=1),
        hovertemplate="%{text}",
        marker=dict(size=7, line=dict(width=0.5, color='black')),
        name=f'Cluster {cluster_id}'
    )
    fig_tsne.add_trace(scatter_trace)

# Actualizar el layout de la visualización de t-SNE
fig_tsne.update_layout(
    title='Gráfico de Dispersión de t-SNE con Curvas de Densidad de Kernel',
    xaxis_title='Dimensión 1',
    yaxis_title='Dimensión 2',
    showlegend=True,
    legend_title='Clusters',
    width=1084
)

# Mostrar el gráfico de t-SNE
st.plotly_chart(fig_tsne)

# Mostrar el DataFrame final con el cluster asignado
st.write(df_supernovas_clustering)


