import numpy as np
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

# Configurar logging para mostrar solo advertencias y errores
logging.basicConfig(level=logging.WARNING)


# Function to obtain the list of files from a GitHub repository using the API
@st.cache_data
def get_github_file_list(repo_url, subdirectory=""):
    api_url = repo_url.replace("github.com", "api.github.com/repos") + f"/contents/{subdirectory}"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = [file['download_url'] for file in response.json() if file['name'].endswith(".snana.dat")]
        return files
    else:
        return []

# Function to download and read the content of a file from GitHub
@st.cache_data
def download_file_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Function to safely convert a value to float
def convert_to_float(value, default_value=None):
    try:
        return float(value)
    except ValueError:
        return default_value

# Function to read the downloaded supernova file and extract relevant data
def read_supernova_file_content(content):
    # Variables and lists to store data
    mjd, mag, magerr, flx, flxerr, filters = [], [], [], [], [], []
    snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv = None, None, None, None, None, None, None
    observations_before_peak, peak_observations, observations_after_peak = None, None, None

    # Process the file line by line
    for line in content.splitlines():
        if line.startswith("SNID:"):
            snid = line.split()[1]
        elif line.startswith("RA:"):
            ra = convert_to_float(line.split()[1])
        elif line.startswith("DECL:"):
            decl = convert_to_float(line.split()[1])
        elif line.startswith("REDSHIFT_FINAL:"):
            redshift = convert_to_float(line.split()[1])
        elif line.startswith("MWEBV:"):
            mwebv = convert_to_float(line.split()[1])
        elif line.startswith("NOBS_BEFORE_PEAK:"):
            observations_before_peak = convert_to_float(line.split()[1])
        elif line.startswith("NOBS_TO_PEAK:"):
            peak_observations = convert_to_float(line.split()[1])
        elif line.startswith("NOBS_AFTER_PEAK:"):
            observations_after_peak = convert_to_float(line.split()[1])
        elif line.startswith("PARSNIP_PRED:"):
            parsnip_pred = ' '.join(line.split()[1:])
        elif line.startswith("SUPERRAENN_PRED:"):
            superraenn_pred = ' '.join(line.split()[1:])
        elif line.startswith("OBS:"):  # Extract observations
            data = line.split()
            mjd.append(convert_to_float(data[1]))  # MJD (Modified Julian Date)
            filters.append(data[2])     # Filter (g, r, i, z, etc.)
            flx.append(convert_to_float(data[4]))  # Flux (FLUXCAL)
            flxerr.append(convert_to_float(data[5]))  # Flux error (FLUXCALERR)
            mag.append(convert_to_float(data[6]))  # Magnitude (MAG)
            magerr.append(convert_to_float(data[7]))  # Magnitude error (MAGERR)

    return mjd, mag, magerr, flx, flxerr, filters, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observations_before_peak, peak_observations, observations_after_peak

# Function to store light curves as a DataFrame
def save_light_curves_as_vectors(vector_list, file_name, mjd, mag, magerr, flx, flxerr, filters, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observations_before_peak, peak_observations, observations_after_peak):
    for i in range(len(mjd)):
        curve_vector = {
            'file_name': file_name,
            'snid': snid,
            'mjd': mjd[i],
            'filter': filters[i],
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
            'observations_before_peak': observations_before_peak,
            'peak_observations': peak_observations,
            'observations_after_peak': observations_after_peak
        }
        vector_list.append(curve_vector)

# Download and process supernova files from GitHub
@st.cache_data
def download_and_process_supernovas(repo_url, subdirectory=""):
    file_list = get_github_file_list(repo_url, subdirectory)
    vector_list = []

    for file_url in file_list:
        file_name = file_url.split("/")[-1]
        content = download_file_from_github(file_url)

        if content:
            mjd, mag, magerr, flx, flxerr, filters, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observations_before_peak, peak_observations, observations_after_peak = read_supernova_file_content(content)
            save_light_curves_as_vectors(vector_list, file_name, mjd, mag, magerr, flx, flxerr, filters, snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv, observations_before_peak, peak_observations, observations_after_peak)

    return pd.DataFrame(vector_list)

# Load supernova data from GitHub
st.write("Downloading and processing supernova files...")
repo_url = "https://github.com/SArcD/supernovaIA"
df_light_curves = download_and_process_supernovas(repo_url)

# Save data to a CSV file
df_light_curves.to_csv('light_curves_with_parsnip_and_ra_decl_redshift_snid.csv', index=False)
st.write("Data saved in 'light_curves_with_parsnip_and_ra_decl_redshift_snid.csv'.")

#########

import streamlit as st

# Title
st.title("Young Supernova Experiment (YSE) Database")

# Justified text using unsafe_allow_html
st.write("""
    <div style='text-align: justify;'>
    The <strong>Young Supernova Experiment (YSE)</strong> focuses on discovering and studying supernovae in their early stages. This allows scientists to understand the explosions of massive stars and other transient phenomena.
    
    <h3>Key Features:</h3>
    <ul>
    <li><strong>Early Detection</strong>: Captures supernovae within days or hours of explosion.</li>
    <li><strong>Light Curves</strong>: Provides detailed light curves across multiple wavelengths, helping track the evolution of each supernova.</li>
    <li><strong>Multi-band Photometry</strong>: Includes data in different filters (g, r, i, z) for comprehensive spectral analysis.</li>
    <li><strong>Classification</strong>: Supernovae are categorized by type, including Type Ia (thermonuclear), Type II, and Type Ibc (collapsing massive stars).</li>
    <li><strong>Redshift & Astrometry</strong>: Contains redshift info to estimate distances and precise coordinates (RA, Dec).</li>
    <li><strong>Milky Way Extinction</strong>: Accounts for interstellar dust that affects the observed brightness.</li>
    <li><strong>Spectroscopy</strong>: Select supernovae also include spectroscopic data for detailed chemical composition analysis.</li>
    </ul>

    <h3>Applications:</h3>
    <ul>
    <li><strong>Cosmology</strong>: Type Ia supernovae are vital for measuring cosmic distances and studying the universe’s expansion.</li>
    <li><strong>Stellar Evolution</strong>: Helps refine models of how stars end their lives.</li>
    <li><strong>Transient Events</strong>: Aids in understanding short-lived cosmic phenomena beyond supernovae.</li>
    </ul>

    This database is publicly available and supports cutting-edge research on the life cycle of stars and cosmic structures.
    </div>
    """, unsafe_allow_html=True)



# Function to create a scatter plot for supernova positions
def create_position_plot():
    fig = px.scatter_polar(df_light_curves, r='redshift', theta='ra', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Polar Positions of Supernovae')
    return fig

# Show the position plot in Streamlit
#st.plotly_chart(create_position_plot())

def create_rectangular_position_plot():
    fig = px.scatter(df_light_curves,
                     x='ra',
                     y='decl',
                     color='parsnip_pred',  # Color by PARSNIP_PRED
                     hover_data=['snid', 'redshift', 'superraenn_pred'],  # Show SNID, redshift, and SUPERRAENN on hover
                     title='Position of Supernovae in the Sky (RA vs Dec)',
                     labels={'ra': 'Right Ascension (RA)', 'decl': 'Declination (Dec)'}
                     #color_discrete_sequence=px.colors.sequential.Viridis  # Use the Viridis color palette
                     )
    return fig

# Create a Declination vs Redshift plot
def create_decl_vs_redshift_plot():
    fig = px.scatter_polar(df_light_curves, r='redshift', theta='decl', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Polar Positions of Supernovae (Dec) vs Redshift')
    return fig

# Let the user select the type of plot to display
plot_option = st.selectbox("Select the type of plot to display:", ["Right Ascension vs Redshift", "Declination vs Redshift", "Declination vs Right Ascension"])

# Show the plot based on the selected option
if plot_option == "Right Ascension vs Redshift":
    st.plotly_chart(create_position_plot())
elif plot_option == "Declination vs Right Ascension" :
    st.plotly_chart(create_rectangular_position_plot())
elif plot_option == "Declination vs Redshift" :
    st.plotly_chart(create_decl_vs_redshift_plot())

#######

# Function to calculate days relative to the luminosity peak
def calculate_days_relative_to_peak(df_supernova):
    # Calculate the MJD of the luminosity peak (minimum magnitude)
    mjd_peak = df_supernova.loc[df_supernova['mag'].idxmin(), 'mjd']
    df_supernova['days_relative'] = df_supernova['mjd'] - mjd_peak
    return df_supernova

# Function to plot the light curve of a specific supernova with relevant information in the title
#def plot_light_curve(df_supernova):
#    fig = go.Figure()
#
#    # Calculate days relative to the luminosity peak
#    df_supernova = calculate_days_relative_to_peak(df_supernova)
#
#    for filter in df_supernova['filter'].unique():
#        df_filter = df_supernova[df_supernova['filter'] == filter]
#        fig.add_trace(go.Scatter(
#            x=df_filter['days_relative'],  # Use days relative to the peak as the X axis
#            y=df_filter['mag'],
#            mode='lines+markers',
#            name=filter
#        ))

#    # Extract relevant information for the title
#    snid = df_supernova['snid'].iloc[0]
#    supernova_type = df_supernova['parsnip_pred'].iloc[0]
#    ra = df_supernova['ra'].iloc[0]
#    decl = df_supernova['decl'].iloc[0]
#    redshift = df_supernova['redshift'].iloc[0]

#    # Invert the Y axis since lower magnitudes are brighter and add the information to the title
#    fig.update_layout(
#        title=(
#            f'Light Curve of {snid} ({supernova_type})\n'
#            f'RA: {ra}°, Dec: {decl}°, Redshift: {redshift}'
#        ),
#        xaxis_title='Days Relative to Luminosity Peak',
#        yaxis_title='Magnitude',
#        yaxis=dict(autorange='reversed'),  # Invert the Y axis
#        showlegend=True
#    )

#    return fig

# Constantes de extinción para diferentes filtros

def corregir_magnitud_extincion(m, MWEBV, filtro='g'):
    # Constantes de extinción para diferentes filtros
    
    extincion_filtros = {
        'g': 3.303,
        'r': 2.285,
        'i': 1.698,
        'z': 1.263,
        'X': 2.000,  # Valor ajustado para el filtro 'x'
        'Y': 1.000   # Valor ajustado para el filtro 'Y'
    }
    
    
    #"""
    #Corrige la magnitud aparente por la extinción debido al polvo galáctico.
    
    #:param m: Magnitud aparente sin corregir.
    #:param MWEBV: Valor de extinción por polvo galáctico (MWEBV).
    #:param filtro: Filtro utilizado (g, r, i, z).
    #:return: Magnitud corregida por extinción.
    #"""
    if filtro in extincion_filtros:
        A_lambda = extincion_filtros[filtro] * MWEBV
        m_corregida = m - A_lambda
    else:
        raise ValueError("Filtro no válido. Usa 'g', 'r', 'i' o 'z', 'X', 'Y'.")
    
    return m_corregida



def corregir_magnitud_redshift(m_corregida, z):
    """
    Corrige la magnitud aparente por el efecto del redshift (corrimiento al rojo).
    
    :param m_corregida: Magnitud corregida por extinción.
    :param z: Redshift de la supernova.
    :return: Magnitud corregida por redshift.
    """
    # La corrección por redshift es básicamente una corrección de distancia.
    D_L = (3e5 * z / 70) * (1 + z)  # Distancia de luminosidad aproximada en Mpc
    D_L_parsecs = D_L * 1e6  # Convertir a parsecs
    m_redshift_corregida = m_corregida - 5 * (np.log10(D_L_parsecs) - 1)
    
    return m_redshift_corregida



def plot_light_curve(df_supernova):
    fig = go.Figure()

    # Calcular días relativos al pico de luminosidad
    df_supernova = calculate_days_relative_to_peak(df_supernova)
    
    # Obtener valores generales para la corrección
    MWEBV = df_supernova['mwebv'].iloc[0]  # Extinción por polvo
    redshift = df_supernova['redshift'].iloc[0]  # Redshift de la supernova

    # Filtros permitidos, ahora incluyendo 'Y'
    extincion_filtros = {
        'g': 3.303,
        'r': 2.285,
        'i': 1.698,
        'z': 1.263,
        'X': 2.000,  # Valor ajustado para el filtro 'x'
        'Y': 1.000   # Valor ajustado para el filtro 'Y'
    }
    filtros_permitidos = extincion_filtros.keys()

    # Bandera para verificar si algún filtro fue graficado
    filtros_graficados = False

    # Iterar sobre cada filtro único
    for filtro in df_supernova['filter'].unique():
        if filtro not in filtros_permitidos:
            st.write(f"Filtro '{filtro}' no es válido. Solo se permiten: {filtros_permitidos}.")
            continue  # Saltar filtros no válidos

        df_filtro = df_supernova[df_supernova['filter'] == filtro]

        # Comprobar si hay suficientes mediciones en el filtro
        if len(df_filtro) < 1:
            st.write(f"Advertencia: No hay suficientes mediciones para el filtro {filtro}.")
            continue

        # Verificar si hay valores nulos en la magnitud o redshift
        if df_filtro['mag'].isnull().all():
            st.write(f"Advertencia: Todas las mediciones de magnitud están ausentes para el filtro {filtro}.")
            continue

        if pd.isnull(MWEBV):
            st.write(f"Advertencia: valor nulo en 'MWEBV' para el filtro {filtro}")
            continue

        if pd.isnull(redshift):
            st.write(f"Advertencia: valor nulo en 'redshift' para el filtro {filtro}")
            continue

        # Corregir la magnitud por extinción y redshift
        try:
            df_filtro['mag_corregida'] = df_filtro['mag'].apply(lambda m: corregir_magnitud_redshift(
                corregir_magnitud_extincion(m, MWEBV, filtro), redshift))
        except Exception as e:
            st.write(f"Error en la corrección de la magnitud para el filtro {filtro}: {str(e)}")
            continue

        # Añadir la curva corregida al gráfico si hay más de un punto
        if len(df_filtro['mag_corregida']) > 1:
            fig.add_trace(go.Scatter(
                x=df_filtro['days_relative'],
                y=df_filtro['mag_corregida'],
                mode='lines+markers',
                name=filtro
            ))
            filtros_graficados = True
        else:
            st.write(f"Advertencia: No hay suficientes puntos para graficar en el filtro {filtro}.")

    # Verificar si se ha añadido algún trazo al gráfico
    if not filtros_graficados:
        st.write("No se pudo generar una curva de luz: No hay datos suficientes después de aplicar las correcciones.")
        return None

    # Extraer información relevante para el título
    snid = df_supernova['snid'].iloc[0]
    tipo_supernova = df_supernova['parsnip_pred'].iloc[0]
    ra = df_supernova['ra'].iloc[0]
    decl = df_supernova['decl'].iloc[0]

    # Invertir el eje Y porque las magnitudes menores son más brillantes
    fig.update_layout(
        title=(
            f'Curva de Luz de {snid} ({tipo_supernova})\n'
            f'RA: {ra}°, Dec: {decl}°, Redshift: {redshift}'
        ),
        xaxis_title='Días Relativos al Pico de Luminosidad',
        yaxis_title='Magnitud Corregida',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
        showlegend=True
    )

    return fig


# Select the supernova type and minimum number of observations with a slider
supernova_type = st.text_input("Enter the supernova type (e.g., 'SN Ia', 'SN Ib', 'SN II'):")
min_observations = st.number_input("Specify the minimum number of observations:", min_value=1, value=5)

# Function to filter supernovae by type and minimum number of observations
def filter_supernovae_by_type(df, supernova_type, min_observations):
    # Filter by supernova type (PARSNIP_PRED)
    filtered_df = df[df['parsnip_pred'] == supernova_type]

    # Group by SNID and count the number of observations per supernova
    supernovae_with_observations = filtered_df.groupby('snid').filter(lambda x: len(x) >= min_observations)
    
    return supernovae_with_observations

# Filter supernovae by type and minimum number of observations
df_filtered_supernovae = filter_supernovae_by_type(df_light_curves, supernova_type, min_observations)

# Display the results if there are supernovae that meet the criteria
if not df_filtered_supernovae.empty:
    filtered_supernovae_by_snid = df_filtered_supernovae['snid'].unique()
    st.write(f"Found {len(filtered_supernovae_by_snid)} supernovae of type '{supernova_type}' with at least {min_observations} observations.")
    
    # Horizontal slider to select a supernova and display its light curve
    selected_index = st.slider('Select a supernova to view its light curve:', 
                               min_value=0, max_value=len(filtered_supernovae_by_snid)-1, step=1)
    selected_snid = filtered_supernovae_by_snid[selected_index]
    df_selected_supernova = df_filtered_supernovae[df_filtered_supernovae['snid'] == selected_snid]
    st.plotly_chart(plot_light_curve(df_selected_supernova))

else:
    st.write(f"No supernovae of type '{supernova_type}' found with at least {min_observations} observations.")

# --- Show DataFrame with filtered supernova details ---

# Function to calculate Δm15 for supernovae of type Ia with an option for an alternative filter
def calculate_delta_m15(df_supernova, preferred_filter='g', alternative_filter='i'):
    df_filter = df_supernova[df_supernova['filter'] == preferred_filter]
    
    # If there is no data in the preferred filter, use the alternative filter
    if df_filter.empty and alternative_filter:
        df_filter = df_supernova[df_supernova['filter'] == alternative_filter]
    
    # If still no data, return NA
    if df_filter.empty:
        return 'NA'  # No data in the preferred or alternative filter
    
    # Get the MJD of the luminosity peak
    mjd_peak = df_filter.loc[df_filter['mag'].idxmin(), 'mjd']
    
    # Filter observations 15 days after the peak
    df_15_days_after = df_filter[(df_filter['mjd'] > mjd_peak) & (df_filter['mjd'] <= mjd_peak + 15)]
    
    if not df_15_days_after.empty:
        mag_peak = df_filter.loc[df_filter['mag'].idxmin(), 'mag']
        mag_15_days_after = df_15_days_after['mag'].mean()
        delta_m15 = mag_15_days_after - mag_peak
        return delta_m15
    else:
        return 'NA'

# Function to calculate the plateau duration for type II or Ibc supernovae
def calculate_plateau_duration(df_supernova, filter='r'):
    df_filter = df_supernova[df_supernova['filter'] == filter]
    
    if df_filter.empty:
        return 'NA'  # No data in this filter
    
    # Find the MJD of the peak
    mjd_peak = df_filter.loc[df_filter['mag'].idxmin(), 'mjd']
    
    # Define the plateau as the time between the peak and when the magnitude drops by 1 or more
    df_plateau = df_filter[df_filter['mag'] <= (df_filter['mag'].min() + 1)]
    
    if not df_plateau.empty:
        plateau_duration = df_plateau['mjd'].max() - mjd_peak
        return plateau_duration
    else:
        return 'NA'

# Function to calculate the luminosity drop rate after the plateau
def calculate_fall_rate(df_supernova, filter='r'):
    df_filter = df_supernova[df_supernova['filter'] == filter]
    
    if df_filter.empty:
        return 'NA'
    
    # Find the MJD of the end of the plateau (peak)
    mjd_peak = df_filter.loc[df_filter['mag'].idxmin(), 'mjd']
    
    # Find the last observation of the supernova
    mjd_end = df_filter['mjd'].max()
    mag_end = df_filter.loc[df_filter['mjd'] == mjd_end, 'mag'].values[0]
    
    # Magnitude at the end of the plateau
    mag_peak = df_filter['mag'].min()
    
    # Calculate the luminosity drop rate
    fall_rate = (mag_end - mag_peak) / (mjd_end - mjd_peak)
    
    return fall_rate

# Function to calculate the average magnitude during the plateau
def calculate_plateau_average_magnitude(df_supernova, filter='r'):
    df_filter = df_supernova[df_supernova['filter'] == filter]
    
    if df_filter.empty:
        return 'NA'
    
    # Magnitudes during the plateau (magnitude <= min(magnitude) + 1)
    df_plateau = df_filter[df_filter['mag'] <= df_filter['mag'].min() + 1]
    
    if df_plateau.empty:
        return 'NA'
    
    # Average the magnitudes during the plateau
    avg_magnitude = df_plateau['mag'].mean()
    
    return avg_magnitude

# Modification in the supernova summary to include SN II
def create_summary_dataframe(df_supernovae, supernova_type):
    results = []
    
    for snid in df_supernovae['snid'].unique():
        df_supernova = df_supernovae[df_supernovae['snid'] == snid]
        supernova_type = df_supernova['parsnip_pred'].iloc[0]
        ra = df_supernova['ra'].iloc[0]
        decl = df_supernova['decl'].iloc[0]
        redshift = df_supernova['redshift'].iloc[0]
        
        # Calculate the peak magnitude for each available filter
        peak_magnitudes = {}
        for filter in df_supernova['filter'].unique():
            filter_group = df_supernova[df_supernova['filter'] == filter]
            peak_magnitude_filter = filter_group['mag'].min() if not filter_group['mag'].isnull().all() else 'NA'
            peak_magnitudes[f'peak_magnitude_{filter}'] = peak_magnitude_filter
        
        # Calculate event duration
        mjd_peak = df_supernova.loc[df_supernova['mag'].idxmin(), 'mjd']
        event_duration = df_supernova['mjd'].max() - mjd_peak
        
        # Calculate specific parameters for SN Ia, SN II, and SN Ibc
        delta_m15 = 'NA'
        plateau_duration = 'NA'
        fall_rate = 'NA'
        avg_plateau_magnitude = 'NA'
        
        if supernova_type == 'SN Ia':
            delta_m15 = calculate_delta_m15(df_supernova, preferred_filter='g', alternative_filter='i')
        elif supernova_type in ['SN II', 'SN Ibc']:
            plateau_duration = calculate_plateau_duration(df_supernova, filter='r')
            fall_rate = calculate_fall_rate(df_supernova, filter='r')
            avg_plateau_magnitude = calculate_plateau_average_magnitude(df_supernova, filter='r')
        
        # Add the data to the summary
        summary = {
            'SNID': snid,
            **peak_magnitudes,  # Add peak magnitudes by filter
            'Event Duration': event_duration,
            'RA': ra,
            'Dec': decl,
            'Redshift': redshift
        }
        
        # Add Δm15 only if the supernova type is Ia
        if supernova_type == 'SN Ia':
            summary['Δm15 (g/i)'] = delta_m15
        
        # Add parameters only if the supernova type is SN II or Ibc
        if supernova_type in ['SN II', 'SN Ibc']:
            summary['Plateau Duration (r)'] = plateau_duration
            summary['Fall Rate (r)'] = fall_rate
            summary['Average Plateau Magnitude (r)'] = avg_plateau_magnitude
        
        results.append(summary)

    return pd.DataFrame(results)
    
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import plotly.express as px

# We assume that df_supernova_clustering has already been generated after clustering
df_parameters = create_summary_dataframe(df_filtered_supernovae, supernova_type)

# Show the DataFrame to verify it has data
st.write("Verifying df_parameters content:")
st.write(df_parameters)

# Remove rows with NaN values
df_supernova_clustering = df_parameters.dropna()

# Select numerical columns for clustering
numerical_columns = df_supernova_clustering.select_dtypes(include=['number'])

# Normalize the data
scaler = StandardScaler()
numerical_columns_scaled = scaler.fit_transform(numerical_columns)

# Hierarchical clustering
num_clusters = st.number_input('Select the number of clusters', min_value=2, max_value=10, value=5, step=1)
clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
df_supernova_clustering['cluster'] = clustering.fit_predict(numerical_columns_scaled)

# Collect the names of supernovae in each cluster and store them
supernova_names_clusters = {}

for cluster_id in range(num_clusters):
    # Filter the supernovae by cluster
    supernovae_in_cluster = df_supernova_clustering[df_supernova_clustering['cluster'] == cluster_id]['SNID'].tolist()
    
    # Store in the dictionary with the name 'cluster_X'
    supernova_names_clusters[f'cluster_{cluster_id}'] = supernovae_in_cluster

# Show the names of the supernovae in each cluster
for cluster_id, supernovae in supernova_names_clusters.items():
    st.write(f"Supernovae in {cluster_id}:")
    st.write(supernovae)

##############

# Show a dropdown menu for the user to choose the cluster
selected_cluster = st.selectbox(
    "Select the cluster to view supernovae:",
    list(supernova_names_clusters.keys())  # Show available clusters
)

# Get the supernovae in the selected cluster
supernovae_in_cluster = supernova_names_clusters[selected_cluster]

if len(supernovae_in_cluster) > 0:
    # Show the number of supernovae in the cluster
    st.write(f"Found {len(supernovae_in_cluster)} supernovae in {selected_cluster}.")
    
    # Horizontal slider to select a specific supernova in the cluster
    selected_index = st.slider('Select a supernova to view its light curve:',
                               min_value=0, max_value=len(supernovae_in_cluster)-1, step=1)
    
    # Get the SNID of the selected supernova
    selected_snid = supernovae_in_cluster[selected_index]
    
    # Filter the light curve DataFrame (df_light_curves) by the selected SNID
    df_selected_supernova = df_light_curves[df_light_curves['snid'] == selected_snid]
    
    # Verify that the supernova has data to plot
    if not df_selected_supernova.empty:
        # Plot the light curve of the selected supernova
        st.plotly_chart(plot_light_curve(df_selected_supernova))
    else:
        st.write(f"No light curve data found for supernova {selected_snid}.")
else:
    st.write(f"No supernovae in {selected_cluster}.") 

st.write(df_supernova_clustering.columns)

########
import pandas as pd
import numpy as np

# Supongamos que df_light_curves ya contiene los datos de las supernovas

# Función para calcular las magnitudes de pico por cada filtro en df_light_curves
def calcular_picos(df_light_curves):
    df_peaks = df_light_curves.groupby(['snid', 'filter']).apply(
        lambda x: x.loc[x['mag'].idxmin()])  # Selecciona la fila con la menor magnitud (pico)
    df_peaks = df_peaks.reset_index(drop=True)
    
    # Pivotear para obtener una columna por filtro con la magnitud de pico
    df_peaks = df_peaks.pivot(index='snid', columns='filter', values='mag').reset_index()
    
    # Renombrar las columnas para que reflejen los filtros
    df_peaks.columns = ['snid'] + [f'peak_magnitude_{filtro}' for filtro in df_peaks.columns[1:]]
    
    return df_peaks

# Calculamos los picos a partir de df_light_curves
df_peaks = calcular_picos(df_light_curves)

# Ahora renombramos la columna 'snid' a 'SNID' para hacer el merge con df_supernova_clustering
df_peaks = df_peaks.rename(columns={'snid': 'SNID'})

# Función para combinar los picos con el DataFrame de clustering
def combinar_picos_con_clustering(df_supernova_clustering, df_peaks):
    # Realizar el merge con la columna 'SNID'
    df_supernova_clustering = pd.merge(df_supernova_clustering, df_peaks, on='SNID', how='left')
    return df_supernova_clustering

# Aplicar la corrección de magnitudes absolutas
def corregir_magnitudes_abs(df, extincion_filtros):
    """
    Corrige las magnitudes aparentes para convertirlas en magnitudes absolutas.
    """
    # Iteramos sobre cada filtro disponible en extincion_filtros
    for filtro in extincion_filtros.keys():
        # Columnas correspondientes a la magnitud de pico en el DataFrame
        peak_col = f'peak_magnitude_{filtro}'
        
        # Comprobamos si la columna de pico existe
        if peak_col in df.columns:
            # Corregir magnitudes con extinción y redshift
            df[f'abs_magnitude_{filtro}'] = df.apply(
                lambda row: corregir_magnitud_redshift(
                    corregir_magnitud_extincion(row[peak_col], row['mwebv'], filtro), row['redshift']
                ) if not pd.isnull(row[peak_col]) else np.nan, axis=1)
        else:
            print(f"Advertencia: La columna {peak_col} no existe en el DataFrame.")
    
    return df

# Función para corregir magnitud por extinción
def corregir_magnitud_extincion(m, MWEBV, filtro):
    extincion_filtros = {
        'g': 3.303,
        'r': 2.285,
        'i': 1.698,
        'z': 1.263,
        'X': 2.000,  # Valor ajustado para el filtro 'X'
        'Y': 1.000   # Valor ajustado para el filtro 'Y'
    }
    
    # Corrección por extinción galáctica
    if filtro in extincion_filtros:
        A_lambda = extincion_filtros[filtro] * MWEBV
        m_corregida = m - A_lambda
    else:
        raise ValueError(f"Filtro no válido: {filtro}")
    
    return m_corregida

# Función para corregir magnitud por redshift
def corregir_magnitud_redshift(m_corregida, z):
    # Corrección por redshift (corrimiento al rojo)
    D_L = (3e5 * z / 70) * (1 + z)  # Distancia de luminosidad aproximada en Mpc
    D_L_parsecs = D_L * 1e6  # Convertir a parsecs
    m_redshift_corregida = m_corregida - 5 * (np.log10(D_L_parsecs) - 1)
    
    return m_redshift_corregida

# Supongamos que df_supernova_clustering ya ha sido creado en pasos anteriores
# Combinamos los picos con el DataFrame de clustering
df_supernova_clustering = combinar_picos_con_clustering(df_supernova_clustering, df_peaks)

# Definimos los valores de extinción para cada filtro
extincion_filtros = {
    'g': 3.303,
    'r': 2.285,
    'i': 1.698,
    'z': 1.263,
    'X': 2.000,  # Valor ajustado para el filtro 'X'
    'Y': 1.000   # Valor ajustado para el filtro 'Y'
}

# Aplicamos la corrección de magnitudes absolutas
df_supernova_clustering = corregir_magnitudes_abs(df_supernova_clustering, extincion_filtros)

########
st.write(df_supernova_clustering.columns)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get the names of the numerical columns, excluding the 'cluster' column
numerical_columns = df_supernova_clustering.select_dtypes(include='number').drop(columns=['cluster']).columns

# Calculate the number of rows and columns in the panel (one column per parameter)
num_rows = len(numerical_columns)
num_cols = 1  # One column for each parameter

# Adjust vertical spacing and the height of the subplots
subplot_height = 400  # Adjust the height as preferred
vertical_spacing = 0.01  # Adjust the vertical spacing as preferred

# Create subplots for each parameter
fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=numerical_columns, vertical_spacing=vertical_spacing)

# Create a box plot for each parameter and compare clusters
for i, column in enumerate(numerical_columns):
    # Get the data for each cluster for the current parameter
    cluster_data = [df_supernova_clustering[df_supernova_clustering['cluster'] == cluster][column] for cluster in range(num_clusters)]

    # Add the box plot to the corresponding subplot
    for j in range(num_clusters):
        box = go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'Cluster {j}')
        box.hovertemplate = 'id: %{text}'  # Add the value of the 'SNID' column to the hovertemplate
        box.text = df_supernova_clustering[df_supernova_clustering['cluster'] == j]['SNID']  # Assign 'SNID' values to the text
        fig.add_trace(box, row=i+1, col=1)

# Update the layout and show the panel of box plots
fig.update_layout(showlegend=False, height=subplot_height*num_rows, width=800,
                  title_text='Cluster Comparison - Box Plot',
                  margin=dict(t=100, b=100, l=50, r=50))  # Adjust the margins of the layout

# Show the box plot in Streamlit
st.plotly_chart(fig)


#########################

##########3

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(numerical_columns_scaled)
df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['cluster'] = df_supernova_clustering['cluster']

# Cluster visualization using PCA
#fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='cluster', title='Clusters visualized with PCA')
#st.plotly_chart(fig_pca)

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# Create a t-SNE instance with the desired hyperparameters
tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=10, learning_rate=5)

# Fit t-SNE to the PCA data (assuming pca_data has already been calculated)
tsne_data = tsne.fit_transform(pca_data)

# Create a Plotly figure
fig = go.Figure()

for cluster_id in np.unique(df_supernova_clustering['cluster']):
    indices = df_supernova_clustering['cluster'] == cluster_id

    scatter_trace = go.Scatter(
        x=tsne_data[indices, 0],
        y=tsne_data[indices, 1],
        mode='markers',
        text=df_supernova_clustering.loc[indices, ['SNID', 'RA', 'Dec', 'Redshift']].apply(lambda x: '<br>'.join(x.astype(str)), axis=1),
        hovertemplate="%{text}",
        marker=dict(
            size=7,
            line=dict(width=0.5, color='black')
        ),
        name=f'Cluster {cluster_id}'
    )
    fig.add_trace(scatter_trace)

# Configure the plot layout
fig.update_layout(
    title='t-SNE Scatter Plot',
    xaxis_title='Dimension 1',
    yaxis_title='Dimension 2',
    showlegend=True,
    legend_title='Clusters',
    width=1084  # Adjust the width of the plot
)

# Show the t-SNE plot in Streamlit
st.plotly_chart(fig)

###########

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import streamlit as st

# Select numerical columns, excluding 'RA' and 'Dec'
numerical_columns = df_supernova_clustering.select_dtypes(include=['number']).drop(columns=['RA', 'Dec'])

# Show a dropdown menu for the user to select the variables to use
selected_variables = st.multiselect(
    'Select the variables you want to use for the decision tree:',
    numerical_columns.columns.tolist(),
    default=numerical_columns.columns.tolist()  # By default, select all
)

# Ensure that at least one variable is selected
if len(selected_variables) > 0:
    # Filter the selected columns
    X = numerical_columns[selected_variables]
    
    # Select the cluster column as the target variable
    y = df_supernova_clustering['cluster']
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a decision tree adjusting hyperparameters
    clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=5, min_samples_leaf=2)
    clf.fit(X_train, y_train)
    
    # Show the decision rules learned by the tree
    tree_rules = export_text(clf, feature_names=selected_variables)
    st.text("Decision Tree Rules:")
    st.text(tree_rules)
    
    # Evaluate the model
    score = clf.score(X_test, y_test)
    st.write(f"Model accuracy on test set: {score:.2f}")
else:
    st.write("Please select at least one variable to train the model.")


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Select the cluster
selected_cluster = st.selectbox(
    "Select the cluster to analyze subclusters:",
    df_supernova_clustering['cluster'].unique()
)

# Filter the supernovae from the selected cluster
df_filtered_cluster = df_supernova_clustering[df_supernova_clustering['cluster'] == selected_cluster]

# Select numerical columns excluding RA, Dec, and cluster
filtered_numerical_columns = df_filtered_cluster.select_dtypes(include=['number']).drop(columns=['RA', 'Dec', 'cluster'])

# Normalize the data
scaler = StandardScaler()
scaled_filtered_numerical_columns = scaler.fit_transform(filtered_numerical_columns)

# Select the number of subclusters
num_subclusters = st.number_input('Select the number of subclusters within the selected cluster:', min_value=2, max_value=10, value=3, step=1)

# Apply agglomerative clustering within the selected cluster
clustering_subclusters = AgglomerativeClustering(n_clusters=num_subclusters, linkage='ward')
df_filtered_cluster['subcluster'] = clustering_subclusters.fit_predict(scaled_filtered_numerical_columns)

# --- Apply PCA and then t-SNE ---

# Apply PCA to reduce dimensionality to 50 components, for example
pca = PCA(n_components=2)  # Increase the number of PCA components to retain more information
pca_data_cluster = pca.fit_transform(scaled_filtered_numerical_columns)

# Now apply t-SNE over the result of PCA with adjustments in the hyperparameters
tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=10, learning_rate=5)
tsne_data_cluster = tsne.fit_transform(pca_data_cluster)

# Create a DataFrame with the t-SNE results and the subclusters
df_tsne_cluster = pd.DataFrame(tsne_data_cluster, columns=['t-SNE1', 't-SNE2'])
df_tsne_cluster['subcluster'] = df_filtered_cluster['subcluster']

# Visualize the subclusters within the selected cluster using t-SNE
fig_tsne_subcluster = go.Figure()

for subcluster_id in np.unique(df_tsne_cluster['subcluster']):
    indices = df_tsne_cluster['subcluster'] == subcluster_id
    
    scatter_trace = go.Scatter(
        x=df_tsne_cluster.loc[indices, 't-SNE1'],
        y=df_tsne_cluster.loc[indices, 't-SNE2'],
        mode='markers',
        marker=dict(size=7, line=dict(width=0.5, color='black')),
        name=f'Subcluster {subcluster_id}'
    )
    fig_tsne_subcluster.add_trace(scatter_trace)

# Configure the layout of the t-SNE plot
fig_tsne_subcluster.update_layout(
    title=f'Subclusters within Cluster {selected_cluster} using t-SNE after PCA',
    xaxis_title='t-SNE1',
    yaxis_title='t-SNE2',
    showlegend=True,
    legend_title='Subclusters',
    width=1084  # Adjust the width of the plot
)

# Show the t-SNE plot in Streamlit
#st.plotly_chart(fig_tsne_subcluster)


# Create box plots to compare variables between subclusters within the selected cluster

# Get the names of the numerical columns
filtered_numerical_columns = df_filtered_cluster.select_dtypes(include=['number']).drop(columns=['subcluster']).columns

# Calculate the number of rows for the subplot
num_rows = len(filtered_numerical_columns)

# Create subplots for each numerical parameter, similar to the first clustering routine
fig_box = make_subplots(rows=num_rows, cols=1, subplot_titles=filtered_numerical_columns)

# Add box plots for each numerical column, comparing subclusters within the selected cluster
for i, column in enumerate(filtered_numerical_columns):
    for subcluster in range(num_subclusters):
        cluster_data = df_filtered_cluster[df_filtered_cluster['subcluster'] == subcluster][column]
        box = go.Box(y=cluster_data, boxpoints='all', notched=True, name=f'Subcluster {subcluster}')
        box.hovertemplate = 'id: %{text}'  # Add the value of the 'SNID' column to the hovertemplate
        box.text = df_filtered_cluster[df_filtered_cluster['subcluster'] == subcluster]['SNID']  # Assign 'SNID' values to the text
        fig_box.add_trace(box, row=i+1, col=1)

# Adjust the layout to make it similar to the original plot
fig_box.update_layout(showlegend=False, height=400*num_rows, width=800,
                      title_text=f'Comparison of Variables between Subclusters within Cluster {selected_cluster}',
                      margin=dict(t=100, b=100, l=50, r=50))

# Show the box plots
st.plotly_chart(fig_box)

# Show the DataFrame with subclusters assigned within the selected cluster
st.write(f"DataFrame with subclusters assigned within Cluster {selected_cluster}:")
st.write(df_filtered_cluster[['SNID', 'subcluster']])
