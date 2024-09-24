import requests
import pandas as pd
import streamlit as st
import plotly.express as px
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
st.write("Descargando y procesando archivos de supernovas...")
repo_url = "https://github.com/SArcD/supernovaIA"
df_curvas_luz = descargar_y_procesar_supernovas(repo_url)

# Guardar los datos en un archivo CSV
df_curvas_luz.to_csv('curvas_de_luz_con_parsnip_y_ra_decl_redshift_snid.csv', index=False)
st.write("Datos guardados en 'curvas_de_luz_con_parsnip_y_ra_decl_redshift_snid.csv'.")

# Crear el gráfico de posiciones de supernovas
def crear_grafico_posiciones():
    fig = px.scatter_polar(df_curvas_luz, r='redshift', theta='ra', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Posiciones Polares de Supernovas')
    return fig

# Mostrar el gráfico de posiciones en Streamlit
st.plotly_chart(crear_grafico_posiciones())

# Seleccionar supernova
snid_seleccionado = st.selectbox("Selecciona una supernova para ver su curva de luz:", df_curvas_luz['snid'].unique())

# Función para graficar la curva de luz de una supernova específica
def graficar_curva_de_luz(df_supernova):
    fig = go.Figure()
    for filtro in df_supernova['filtro'].unique():
        df_filtro = df_supernova[df_supernova['filtro'] == filtro]
        fig.add_trace(go.Scatter(x=df_filtro['mjd'], y=df_filtro['mag'], mode='lines+markers', name=filtro))

    # Invertir el eje Y porque las magnitudes menores son más brillantes
    fig.update_layout(
        title=f'Curva de luz de {snid_seleccionado}',
        xaxis_title='MJD (Modified Julian Date)',
        yaxis_title='Magnitud',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
        showlegend=True
    )

    return fig

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

    # Ordenar por el número de observaciones, de mayor a menor
    supernovas_ordenadas = supernovas_con_observaciones.groupby('snid').apply(lambda x: x if len(x) >= min_observaciones else None).reset_index(drop=True)
    supernovas_ordenadas['num_observaciones'] = supernovas_ordenadas.groupby('snid')['snid'].transform('count')
    return supernovas_ordenadas.sort_values(by='num_observaciones', ascending=False)

# Filtrar las supernovas por el tipo y número mínimo de observaciones
df_supernovas_filtradas = filtrar_supernovas_por_tipo(df_curvas_luz, tipo_supernova, min_observaciones)

# Mostrar los resultados si hay supernovas que cumplan con los criterios
if not df_supernovas_filtradas.empty:
    # Ordenar por el número de puntos de observación
    supernovas_filtradas_por_num_obs = df_supernovas_filtradas['snid'].unique()
    st.write(f"Se encontraron {len(supernovas_filtradas_por_num_obs)} supernovas del tipo '{tipo_supernova}' con al menos {min_observaciones} observaciones.")
    
    # Graficar todas las supernovas que cumplan con los criterios
    for snid in supernovas_filtradas_por_num_obs:
        st.write(f"Graficando la supernova: {snid} con {len(df_supernovas_filtradas[df_supernovas_filtradas['snid'] == snid])} observaciones.")
        df_supernova_seleccionada = df_supernovas_filtradas[df_supernovas_filtradas['snid'] == snid]
        st.plotly_chart(graficar_curva_de_luz(df_supernova_seleccionada))

else:
    st.write(f"No se encontraron supernovas del tipo '{tipo_supernova}' con al menos {min_observaciones} observaciones.")





