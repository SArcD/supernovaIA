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

import streamlit as st

st.markdown("""
# Young Supernova Experiment (YSE) Database

The **Young Supernova Experiment (YSE)** focuses on discovering and studying supernovae in their early stages. This allows scientists to understand the explosions of massive stars and other transient phenomena.

## Key Features:
- **Early Detection**: Captures supernovae within days or hours of explosion.
- **Light Curves**: Provides detailed light curves across multiple wavelengths, helping track the evolution of each supernova.
- **Multi-band Photometry**: Includes data in different filters (g, r, i, z) for comprehensive spectral analysis.
- **Classification**: Supernovae are categorized by type, including Type Ia (thermonuclear), Type II, and Type Ibc (collapsing massive stars).
- **Redshift & Astrometry**: Contains redshift info to estimate distances and precise coordinates (RA, Dec).
- **Milky Way Extinction**: Accounts for interstellar dust that affects the observed brightness.
- **Spectroscopy**: Select supernovae also include spectroscopic data for detailed chemical composition analysis.

## About this app:
The primary goal of this application is to provide an interactive platform for analyzing and visualizing data on supernovae, specifically those obtained from the Young Supernova Experiment (YSE). The tool allows users to access relevant data about supernovae, perform statistical analyses, and visualize light curves, thus facilitating a better understanding of these complex astronomical phenomena.

## Main Sections
- **Data Download**: Allows users to access data files from a GitHub repository, filtering and processing the necessary information for analysis.
- **Data Visualization**: Offers multiple interactive plots that represent the evolution of supernova magnitude over time, as well as their positions in the sky. Users can select different filters and features to customize their visualizations.
- **Parameter Calculation**: Includes functions to calculate metrics such as plateau duration, magnitude drop rate, and absolute magnitudes, providing deeper insight into the behavior of supernovae.
- **Classification and Clustering**: Implements machine learning models that allow for the classification of supernovae into different clusters based on user-selected parameters, facilitating comparative analysis between different groups of supernovae.
- **Interactivity**: Provides a user-friendly interface that allows users to interact with the data, perform queries, and visualize results intuitively.

In summary, this application is a comprehensive tool for exploring and analyzing supernova data, designed for researchers and astronomy enthusiasts who wish to delve deeper into the study of these cosmic events.
""")

st.write("""

### **Position of Supernovae as a Function of Celestial Coordinates and Redshift**

In this section, you can plot the position of each type of supernova based on celestial coordinates and redshift. The available plots are Right Ascension vs Redshift, Declination vs Redshift, and Declination vs Right Ascension.
""")

def create_position_plot():
    # Obtener el valor máximo del redshift en los datos
    max_redshift = df_light_curves['redshift'].max()

    fig = px.scatter_polar(df_light_curves, r='redshift', theta='ra', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Polar Positions of Supernovae')

    # Habilitar el botón de reset y zoom out desde la barra de herramientas
    fig.update_layout(
        title='Polar Positions of Supernovae',
        autosize=True,
        polar=dict(
            radialaxis=dict(range=[0, max_redshift], showticklabels=True),  # Ajustar rango según el máximo redshift
            angularaxis=dict(showticklabels=True)
        )
    )

    # Añadir botones personalizados para resetear el gráfico con un botón más pequeño y color naranja
    fig.update_layout(
        updatemenus=[dict(type="buttons",
                          direction="left",
                          buttons=[dict(args=["polar.radialaxis.range", [0, max_redshift * 1.1]], 
                                        label="Reset Zoom", 
                                        method="relayout"
                                        )],
                          pad={"r": 10, "t": 10},
                          showactive=True,
                          x=0.8,
                          xanchor="left",
                          y=1.15,
                          # Personalizar estilo del botón
                          font=dict(size=10, color="black"),
                          bgcolor="orange",  # Color de fondo del botón
                          bordercolor="black",
                          borderwidth=1
                         )]
    )

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

##########-----#############


st.write("""
### **Types of Supernovae as a Function of Redshift with extintion (MWEB)**

This code allows filtering and visualizing supernovae based on a selected redshift range and then plots the positions of the supernovae as a function of their celestial coordinates (Right Ascension and Declination), along with an extinction map. A dropdown menu is provided for the user to select a redshift range. If supernovae are found within the selected range, a heatmap is created with the coordinates and extinction values, as well as a scatter plot to represent the supernovae by type. Additionally, a pie chart is generated to show the percentage of each supernova type within the selected range.
""")



import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
import streamlit as st

# Filtrar el DataFrame para obtener solo una fila por supernova
df_single = df_light_curves.drop_duplicates(subset=['snid', 'ra', 'decl', 'mwebv', 'parsnip_pred'])

# Extraer coordenadas, valores de extinción y tipo de supernova
ra = df_single['ra'].values
decl = df_single['decl'].values
mwebv = df_single['mwebv'].values
supernova_types = df_single['parsnip_pred'].values  # Tipo de supernova

# Crear un menú desplegable para seleccionar el rango de redshift
redshift_option = st.selectbox(
    "Select the redshift range:",
    options=["Range 1 (0.0024 - 0.1)", "Range 2 (0.1 - 0.2)", "Range 3 (0.2 - 0.3073)", "Total Range (0.0024 - 0.3073)"]
)

# Definir los límites de los rangos de redshift
if redshift_option == "Range 1 (0.0024 - 0.1)":
    selected_redshift = (0.0024, 0.1)
elif redshift_option == "Range 2 (0.1 - 0.2)":
    selected_redshift = (0.1, 0.2)
elif redshift_option == "Range 3 (0.2 - 0.3073)":
    selected_redshift = (0.2, 0.3073)
else:  # Total Range
    selected_redshift = (0.0024, 0.3073)

# Filtrar el DataFrame según el rango de redshift seleccionado
filtered_supernovae = df_light_curves[
    (df_light_curves['redshift'] >= selected_redshift[0]) & 
    (df_light_curves['redshift'] <= selected_redshift[1])
]

# Comprobar si hay supernovas filtradas
if not filtered_supernovae.empty:
    # Filtrar el DataFrame para las coordenadas y valores de extinción de las supernovas filtradas
    ra_filtered = filtered_supernovae['ra'].values
    decl_filtered = filtered_supernovae['decl'].values
    mwebv_filtered = filtered_supernovae['mwebv'].values
    supernova_types_filtered = filtered_supernovae['parsnip_pred'].values

    # Definir una malla de coordenadas para interpolación
    ra_grid = np.linspace(ra_filtered.min(), ra_filtered.max(), 100)
    decl_grid = np.linspace(decl_filtered.min(), decl_filtered.max(), 100)
    ra_mesh, decl_mesh = np.meshgrid(ra_grid, decl_grid)

    # Interpolación de los valores de extinción
    mwebv_interp = griddata((ra_filtered, decl_filtered), mwebv_filtered, (ra_mesh, decl_mesh), method='cubic')

    # Crear el gráfico de Plotly
    fig = go.Figure()

    # Agregar el mapa de extinción
    fig.add_trace(go.Heatmap(
        z=mwebv_interp,
        x=ra_grid,
        y=decl_grid,
        colorscale='Viridis',
        colorbar=dict(title='Extinction (MWEBV)'),
        zmin=mwebv.min(),
        zmax=mwebv.max(),
        opacity=0.7,
        showscale=True
    ))

    # Colorear los puntos según el tipo de supernova
    unique_types = np.unique(supernova_types_filtered)
    
    for t in unique_types:
        mask = supernova_types_filtered == t
        fig.add_trace(go.Scatter(
            x=ra_filtered[mask],
            y=decl_filtered[mask],
            mode='markers',
            marker=dict(size=5),
            name=t,  # Asignar el nombre del tipo de supernova a la leyenda
            text=t,  # Texto de hover para mostrar el tipo de supernova
            hoverinfo='text'
        ))

    # Actualizar el layout
    fig.update_layout(
        title=f'Extinction (MWEB) as a function of Right Ascension and Declination - Redshift: {selected_redshift[0]} to {selected_redshift[1]}',
        xaxis_title='Right Ascension (RA)',
        yaxis_title='Declination (Dec)',
        showlegend=True,  # Asegurarse de que la leyenda se muestre
        legend=dict(title='Supernova Type', orientation='h', xanchor='center', x=0.5, yanchor='bottom', y=0.95)
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

    # Pie chart for the percentage of supernovae of each type
    type_counts = filtered_supernovae['parsnip_pred'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(labels=type_counts.index, values=type_counts.values, hole=.3)])
    fig_pie.update_layout(title_text='Percentage of Supernovae by Type in the Selected Range')

    # Display the pie chart in Streamlit
    st.plotly_chart(fig_pie)

else:
    st.write("There are no supernovas within the selected range ")


############################

st.write("""

### **Light curves for the supernovae**

The horizontal axis of each light curve corresponds to the number of days relative to the peak in the light curve (calculated from the Modified Julian Date). The vertical axis represents the absolute magnitude in each filter, corrected for extinction due to interstellar dust, using constants for different filters (g, r, i, z, X, and Y) and for redshift. 

The user can select the type of supernova they want to display, as well as the minimum number of observations a light curve must have to be plotted. Additionally, a DataFrame is included showing additional parameters such as peak magnitudes, event duration, and specific parameters based on supernova type.

""")


# Function to calculate days relative to the luminosity peak
def calculate_days_relative_to_peak(df_supernova):
    # Calculate the MJD of the luminosity peak (minimum magnitude)
    mjd_peak = df_supernova.loc[df_supernova['mag'].idxmin(), 'mjd']
    df_supernova['days_relative'] = df_supernova['mjd'] - mjd_peak
    return df_supernova

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
    
    if filtro in extincion_filtros:
        A_lambda = extincion_filtros[filtro] * MWEBV
        m_corregida = m - A_lambda
    else:
        raise ValueError("Filtro no válido. Usa 'g', 'r', 'i' o 'z', 'X', 'Y'.")
    
    return m_corregida



def corregir_magnitud_redshift(m_corregida, z):

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

##############


#############

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
st.write("**Summary**")

st.write(df_parameters)

with st.expander("Description of Columns in df_parameters"):
    st.write("""
    **SNID**: The unique identifier for each supernova event. This identifier is used to track and reference supernovae across various datasets.

    **peak_magnitude_g**: The peak apparent magnitude observed in the 'g' (green) filter. This value indicates the brightness of the supernova at its brightest point in this specific wavelength range.

    **peak_magnitude_r**: The peak apparent magnitude observed in the 'r' (red) filter. Similar to the 'g' filter, this measurement reflects the supernova's brightness in the red wavelength range.

    **peak_magnitude_i**: The peak apparent magnitude observed in the 'i' (infrared) filter, indicating brightness at infrared wavelengths.

    **peak_magnitude_z**: The peak apparent magnitude observed in the 'z' (near-infrared) filter, reflecting the supernova's brightness in near-infrared wavelengths.

    **peak_magnitude_X**: The peak apparent magnitude observed in the 'X' filter, capturing brightness in a specific wavelength range tailored for certain observations.

    **peak_magnitude_Y**: The peak apparent magnitude observed in the 'Y' filter, reflecting the supernova's brightness in another specific wavelength range.

    **Event Duration**: The total duration of the supernova event, calculated as the time between the first observation and the last observation of the supernova.

    **RA**: The right ascension of the supernova, measured in degrees. This celestial coordinate helps locate the supernova in the sky.

    **Dec**: The declination of the supernova, also measured in degrees. It complements the right ascension for precise location in the celestial coordinate system.

    **Redshift**: The redshift value of the supernova, which indicates how much the light from the supernova has been stretched due to the expansion of the universe. It is essential for determining the distance to the supernova.

    **Δm15 (g/i)**: The decline in magnitude (Δm) measured 15 days after the peak brightness in the 'g' or 'i' filter. This metric is often used to classify supernovae types and assess their explosion mechanisms.

    **Plateau Duration (r)**: The duration for which the supernova remains at a constant brightness level (the plateau) in the 'r' filter. This is significant for understanding the characteristics of Type II supernovae.

    **Fall Rate (r)**: The rate at which the brightness of the supernova decreases after the plateau phase, calculated in the 'r' filter.

    **Average Plateau Magnitude (r)**: The average magnitude of the supernova during the plateau phase in the 'r' filter, providing insight into the supernova's brightness stability.

    **Wavelength ranges for each filter**:
    - **g (green)**: 400 - 550 nm
    - **r (red)**: 550 - 700 nm
    - **i (infrared)**: 700 - 850 nm
    - **z (infrared)**: 850 - 1000 nm
    - **X**: Specific to the YSE database, with a defined range between 500 - 800 nm.
    - **Y (infrared)**: 970 - 1070 nm
    """)


###############33

import pandas as pd
import numpy as np

def corregir_magnitud_extincion(m, MWEBV, filtro):
    # Define los valores de extinción para cada filtro
    extincion_filtros = {
        'g': 3.303,
        'r': 2.285,
        'i': 1.698,
        'z': 1.263,
        'X': 2.000,  # Valor ajustado para el filtro 'x'
        'Y': 1.000   # Valor ajustado para el filtro 'Y'
    }
    
    if filtro in extincion_filtros:
        A_lambda = extincion_filtros[filtro] * MWEBV
        return m - A_lambda
    else:
        raise ValueError(f"Filtro no válido: {filtro}")

def corregir_magnitud_redshift(m_corregida, z):
    # Corrección por redshift
    D_L = (3e5 * z / 70) * (1 + z)  # Distancia de luminosidad en Mpc
    D_L_parsecs = D_L * 1e6  # Convertir a parsecs
    return m_corregida - 5 * (np.log10(D_L_parsecs) - 1)

def calcular_picos_y_magnitudes_absolutas(df_light_curves, df_parametros):
    # Agrupa por SNID
    for snid, group in df_light_curves.groupby('snid'):
        for filtro in group['filter'].unique():
            # Filtra por el filtro actual
            df_filtro = group[group['filter'] == filtro]
            
            # Verifica si hay suficientes datos
            if df_filtro['mag'].count() > 1:  # Se requiere más de un dato para calcular el pico
                # Encuentra el índice del mínimo de magnitud
                idx_pico = df_filtro['mag'].idxmin()
                
                # Obtener los valores necesarios para la corrección
                m_aparente = df_filtro.loc[idx_pico, 'mag']
                MWEBV = df_filtro['mwebv'].iloc[0]
                redshift = df_filtro['redshift'].iloc[0]
                
                # Aplica las correcciones
                m_corregida = corregir_magnitud_extincion(m_aparente, MWEBV, filtro)
                m_absoluta = corregir_magnitud_redshift(m_corregida, redshift)
                
                # Almacena el resultado en df_parametros
                df_parametros.loc[df_parametros['SNID'] == snid, f'peak_magnitude_{filtro}'] = m_absoluta
                # Agregar otros campos como filtros si es necesario

    return df_parametros

# Ejemplo de uso
df_parametros = pd.DataFrame({'SNID': df_light_curves['snid'].unique()})
df_parametros = calcular_picos_y_magnitudes_absolutas(df_light_curves, df_parametros)


# Mostrar el DataFrame de parámetros
#st.write("PI")
#st.write(df_parametros.head())


###############
##############

# Ahora, para agregar los resultados de df_parametros a df_parameters
df_parameters = pd.merge(df_parameters, df_parametros, on='SNID', how='left')

# Muestra los resultados
#st.write(df_parameters.head())

###############
################


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import streamlit as st

# Suponiendo que df_parameters ya está definido y tiene las columnas necesarias
# Elimina las filas con valores NaN en df_parameters
df_supernova_clustering = df_parameters.dropna()

# Permitir que el usuario seleccione columnas específicas para el clustering
selected_columns = st.multiselect(
    'Select columns for clustering:',
    options=df_supernova_clustering.columns.tolist(),  # Todas las columnas del DataFrame
    default=df_supernova_clustering.columns.tolist()    # Por defecto, selecciona todas las columnas
)

# Verificar que al menos una columna esté seleccionada
if selected_columns:
    # Selecciona solo las columnas elegidas por el usuario
    numerical_columns = df_supernova_clustering[selected_columns].select_dtypes(include=['number'])

    # Normaliza los datos
    scaler = StandardScaler()
    numerical_columns_scaled = scaler.fit_transform(numerical_columns)

    # Clustering jerárquico
    num_clusters = st.number_input('Select the number of clusters', min_value=2, max_value=10, value=5, step=1)
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    df_supernova_clustering['cluster'] = clustering.fit_predict(numerical_columns_scaled)

    # Recopilar los nombres de las supernovas en cada clúster y almacenarlos
    supernova_names_clusters = {}

    for cluster_id in range(num_clusters):
        # Filtrar las supernovas por clúster
        supernovae_in_cluster = df_supernova_clustering[df_supernova_clustering['cluster'] == cluster_id]['SNID'].tolist()
        
        # Almacenar en el diccionario con el nombre 'cluster_X'
        supernova_names_clusters[f'cluster_{cluster_id}'] = supernovae_in_cluster

    # Mostrar los resultados
    st.write("Supernovae in each cluster:")
    for cluster, supernovae in supernova_names_clusters.items():
        st.write(f"{cluster}: {supernovae}")

else:
    st.write("Please select at least one column for clustering.")



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

st.write(df_supernova_clustering)

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
#df_supernova_clustering = corregir_magnitudes_abs(df_supernova_clustering, extincion_filtros)
st.write(df_supernova_clustering)
########
st.write(df_supernova_clustering.columns)


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Get the names of the numerical columns, excluding the 'cluster' column
numerical_columns = df_supernova_clustering.select_dtypes(include='number').drop(columns=['cluster']).columns

# Allow the user to select columns for the boxplots
selected_columns = st.multiselect(
    'Select columns to include in the boxplots:',
    options=numerical_columns.tolist(),  # List of numerical columns
    default=numerical_columns.tolist()     # Default selection is all columns
)

# Check if any columns were selected
if selected_columns:
    # Calculate the number of rows and columns in the panel (one column per selected parameter)
    num_rows = len(selected_columns)
    num_cols = 1  # One column for each parameter

    # Adjust vertical spacing and the height of the subplots
    subplot_height = 400  # Adjust the height as preferred
    vertical_spacing = 0.01  # Adjust the vertical spacing as preferred

    # Create subplots for each selected parameter
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=selected_columns, vertical_spacing=vertical_spacing)

    # Create a box plot for each selected parameter and compare clusters
    for i, column in enumerate(selected_columns):
        # Get the data for each cluster for the current parameter
        cluster_data = [df_supernova_clustering[df_supernova_clustering['cluster'] == cluster][column] for cluster in range(num_clusters)]

        # Add the box plot to the corresponding subplot
        for j in range(num_clusters):
            box = go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'Cluster {j}')
            box.hovertemplate = 'id: %{text}'  # Add the value of the 'SNID' column to the hovertemplate
            box.text = df_supernova_clustering[df_supernova_clustering['cluster'] == j]['SNID']  # Assign 'SNID' values to the text
            fig.add_trace(box, row=i + 1, col=1)

    # Update the layout and show the panel of box plots
    fig.update_layout(showlegend=False, height=subplot_height * num_rows, width=800,
                      title_text='Cluster Comparison - Box Plot',
                      margin=dict(t=100, b=100, l=50, r=50))  # Adjust the margins of the layout

    # Show the box plot in Streamlit
    st.plotly_chart(fig)
else:
    st.write("Please select at least one column to display the boxplots.")


#import plotly.graph_objects as go
#from plotly.subplots import make_subplots

# Get the names of the numerical columns, excluding the 'cluster' column
#numerical_columns = df_supernova_clustering.select_dtypes(include='number').drop(columns=['cluster']).columns

# Calculate the number of rows and columns in the panel (one column per parameter)
#num_rows = len(numerical_columns)
#num_cols = 1  # One column for each parameter

# Adjust vertical spacing and the height of the subplots
#subplot_height = 400  # Adjust the height as preferred
#vertical_spacing = 0.01  # Adjust the vertical spacing as preferred

# Create subplots for each parameter
#fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=numerical_columns, vertical_spacing=vertical_spacing)

# Create a box plot for each parameter and compare clusters
#for i, column in enumerate(numerical_columns):
#    # Get the data for each cluster for the current parameter
#    cluster_data = [df_supernova_clustering[df_supernova_clustering['cluster'] == cluster][column] for cluster in range(num_clusters)]#

#    # Add the box plot to the corresponding subplot
#    for j in range(num_clusters):
#        box = go.Box(y=cluster_data[j], boxpoints='all', notched=True, name=f'Cluster {j}')
#        box.hovertemplate = 'id: %{text}'  # Add the value of the 'SNID' column to the hovertemplate
#        box.text = df_supernova_clustering[df_supernova_clustering['cluster'] == j]['SNID']  # Assign 'SNID' values to the text
#        fig.add_trace(box, row=i+1, col=1)

# Update the layout and show the panel of box plots
#fig.update_layout(showlegend=False, height=subplot_height*num_rows, width=800,
#                  title_text='Cluster Comparison - Box Plot',
#                  margin=dict(t=100, b=100, l=50, r=50))  # Adjust the margins of the layout

# Show the box plot in Streamlit
#st.plotly_chart(fig)


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


####################################################################################################33








#import pandas as pd
#import numpy as np
#import plotly.graph_objects as go
#from sklearn.preprocessing import StandardScaler
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA

# Select the cluster
#selected_cluster = st.selectbox(
#    "Select the cluster to analyze subclusters:",
#    df_supernova_clustering['cluster'].unique()
#)

# Filter the supernovae from the selected cluster
#df_filtered_cluster = df_supernova_clustering[df_supernova_clustering['cluster'] == selected_cluster]

# Select numerical columns excluding RA, Dec, and cluster
#filtered_numerical_columns = df_filtered_cluster.select_dtypes(include=['number']).drop(columns=['RA', 'Dec', 'cluster'])

# Normalize the data
#scaler = StandardScaler()
#scaled_filtered_numerical_columns = scaler.fit_transform(filtered_numerical_columns)

# Select the number of subclusters
#num_subclusters = st.number_input('Select the number of subclusters within the selected cluster:', min_value=2, max_value=10, value=3, step=1)

# Apply agglomerative clustering within the selected cluster
#clustering_subclusters = AgglomerativeClustering(n_clusters=num_subclusters, linkage='ward')
#df_filtered_cluster['subcluster'] = clustering_subclusters.fit_predict(scaled_filtered_numerical_columns)

# --- Apply PCA and then t-SNE ---

# Apply PCA to reduce dimensionality to 50 components, for example
#pca = PCA(n_components=2)  # Increase the number of PCA components to retain more information
#pca_data_cluster = pca.fit_transform(scaled_filtered_numerical_columns)

# Now apply t-SNE over the result of PCA with adjustments in the hyperparameters
#tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=10, learning_rate=5)
#tsne_data_cluster = tsne.fit_transform(pca_data_cluster)

# Create a DataFrame with the t-SNE results and the subclusters
#df_tsne_cluster = pd.DataFrame(tsne_data_cluster, columns=['t-SNE1', 't-SNE2'])
#df_tsne_cluster['subcluster'] = df_filtered_cluster['subcluster']

# Visualize the subclusters within the selected cluster using t-SNE
#fig_tsne_subcluster = go.Figure()

#for subcluster_id in np.unique(df_tsne_cluster['subcluster']):
#    indices = df_tsne_cluster['subcluster'] == subcluster_id
    
#    scatter_trace = go.Scatter(
#        x=df_tsne_cluster.loc[indices, 't-SNE1'],
#        y=df_tsne_cluster.loc[indices, 't-SNE2'],
#        mode='markers',
#        marker=dict(size=7, line=dict(width=0.5, color='black')),
#        name=f'Subcluster {subcluster_id}'
#    )
#    fig_tsne_subcluster.add_trace(scatter_trace)

# Configure the layout of the t-SNE plot
#fig_tsne_subcluster.update_layout(
#    title=f'Subclusters within Cluster {selected_cluster} using t-SNE after PCA',
#    xaxis_title='t-SNE1',
#    yaxis_title='t-SNE2',
#    showlegend=True,
#    legend_title='Subclusters',
#    width=1084  # Adjust the width of the plot
#)

## Show the t-SNE plot in Streamlit
##st.plotly_chart(fig_tsne_subcluster)


## Create box plots to compare variables between subclusters within the selected cluster

## Get the names of the numerical columns
#filtered_numerical_columns = df_filtered_cluster.select_dtypes(include=['number']).drop(columns=['subcluster']).columns

# Calculate the number of rows for the subplot
#num_rows = len(filtered_numerical_columns)

# Create subplots for each numerical parameter, similar to the first clustering routine
#fig_box = make_subplots(rows=num_rows, cols=1, subplot_titles=filtered_numerical_columns)

# Add box plots for each numerical column, comparing subclusters within the selected cluster
#for i, column in enumerate(filtered_numerical_columns):
#    for subcluster in range(num_subclusters):
#        cluster_data = df_filtered_cluster[df_filtered_cluster['subcluster'] == subcluster][column]
#        box = go.Box(y=cluster_data, boxpoints='all', notched=True, name=f'Subcluster {subcluster}')
#        box.hovertemplate = 'id: %{text}'  # Add the value of the 'SNID' column to the hovertemplate
#        box.text = df_filtered_cluster[df_filtered_cluster['subcluster'] == subcluster]['SNID']  # Assign 'SNID' values to the text
#        fig_box.add_trace(box, row=i+1, col=1)

# Adjust the layout to make it similar to the original plot
#fig_box.update_layout(showlegend=False, height=400*num_rows, width=800,
#                      title_text=f'Comparison of Variables between Subclusters within Cluster {selected_cluster}',
#                      margin=dict(t=100, b=100, l=50, r=50))

# Show the box plots
#st.plotly_chart(fig_box)

# Show the DataFrame with subclusters assigned within the selected cluster
#st.write(f"DataFrame with subclusters assigned within Cluster {selected_cluster}:")
#st.write(df_filtered_cluster[['SNID', 'subcluster']])

######################

#L#

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Paso 1: Filtrar las supernovas del clúster seleccionado
selected_cluster = st.selectbox("Select the cluster to analyze:", df_supernova_clustering['cluster'].unique())
df_clustered_supernovae = df_supernova_clustering[df_supernova_clustering['cluster'] == selected_cluster]

# Paso 2: Preparar los datos
if not df_clustered_supernovae.empty:
    # Unir todos los datos de supernovas en el clúster
    supernova_ids = df_clustered_supernovae['SNID'].unique()
    df_light_curves_cluster = df_light_curves[df_light_curves['snid'].isin(supernova_ids)]

    # Calcular días relativos al pico de luminosidad
    df_light_curves_cluster = calculate_days_relative_to_peak(df_light_curves_cluster)

    # Paso 2.1: Filtrar para usar solo los datos desde el pico hasta el final
    df_light_curves_cluster = df_light_curves_cluster[df_light_curves_cluster['days_relative'] >= 0]

    # Normalizar los días relativos al pico
    df_light_curves_cluster['days_relative_normalized'] = df_light_curves_cluster.groupby('snid')['days_relative'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())  # Normalización entre 0 y 1
    )

    # Crear el conjunto de entrenamiento
    X = df_light_curves_cluster[['days_relative_normalized']]  # Días relativos normalizados
    y = df_light_curves_cluster['mag']  # Magnitudes observadas

    # Paso 3: Entrenar el modelo de árbol de regresión
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Paso 4: Predecir las magnitudes para un rango de días relativos normalizados
    days_range = np.linspace(X['days_relative_normalized'].min(), X['days_relative_normalized'].max(), 100).reshape(-1, 1)
    predicted_magnitudes = model.predict(days_range)

    # Paso 5: Graficar la curva de luz ajustada
    fig = go.Figure()
    
    # Gráfica de los datos originales
    for snid in supernova_ids:
        df_supernova_data = df_light_curves_cluster[df_light_curves_cluster['snid'] == snid]
        fig.add_trace(go.Scatter(
            x=df_supernova_data['days_relative_normalized'],
            y=df_supernova_data['mag'],
            mode='markers',
            name=f'SNID: {snid}',
            hoverinfo='text',
            text=df_supernova_data['snid']  # Información al pasar el mouse
        ))

    # Gráfica de la curva de ajuste
    fig.add_trace(go.Scatter(
        x=days_range.flatten(),
        y=predicted_magnitudes,
        mode='lines',
        name='Curva de Ajuste',
        line=dict(color='red')
    ))

    # Actualizar el layout
    fig.update_layout(
        title=f'Curva de Luz Ajustada para el Clúster {selected_cluster}',
        xaxis_title='Días Relativos Normalizados al Pico',
        yaxis_title='Magnitud',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
        showlegend=True
    )

    st.plotly_chart(fig)
else:
    st.write("No hay supernovas en este clúster.")



#=#

from sklearn.tree import DecisionTreeRegressor




###---###---###

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeRegressor


# Comprobar cuántas filas hay para cada SNID
counts = df_light_curves['snid'].value_counts()
st.write(counts)  # Muestra cuántas observaciones hay para cada supernova
# Paso 1: Filtrar las supernovas del clúster seleccionado

selected_cluster = st.selectbox("Please, select the cluster to analyze:", df_supernova_clustering['cluster'].unique())
df_clustered_supernovae = df_supernova_clustering[df_supernova_clustering['cluster'] == selected_cluster]

# Paso 2: Preparar los datos
if not df_clustered_supernovae.empty:
    # Unir todos los datos de supernovas en el clúster
    supernova_ids = df_clustered_supernovae['SNID'].unique()
    df_light_curves_cluster = df_light_curves[df_light_curves['snid'].isin(supernova_ids)]

    # Calcular días relativos al pico de luminosidad
    df_light_curves_cluster = calculate_days_relative_to_peak(df_light_curves_cluster)

    # Filtrar para usar solo los datos desde el pico hasta el final
    df_light_curves_cluster = df_light_curves_cluster[df_light_curves_cluster['days_relative'] >= 0]

    # Calcular magnitudes corregidas
    df_light_curves_cluster['mag_corregida'] = df_light_curves_cluster.apply(
        lambda row: corregir_magnitud_redshift(corregir_magnitud_extincion(row['mag'], row['mwebv'], row['filter']), row['redshift']),
        axis=1
    )

    # Normalizar los días relativos al pico
    df_light_curves_cluster['days_relative_normalized'] = df_light_curves_cluster.groupby('snid')['days_relative'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())  # Normalización entre 0 y 1
    )

    # Crear el conjunto de entrenamiento
    X = df_light_curves_cluster[['days_relative_normalized']]  # Días relativos normalizados
    y = df_light_curves_cluster['mag_corregida']  # Magnitudes corregidas

    # Paso 3: Entrenar el modelo de árbol de regresión
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Paso 4: Predecir las magnitudes para un rango de días relativos normalizados
    days_range = np.linspace(X['days_relative_normalized'].min(), X['days_relative_normalized'].max(), 100).reshape(-1, 1)
    predicted_magnitudes = model.predict(days_range)

    # Paso 5: Graficar la curva de luz ajustada
    fig = go.Figure()
    
    # Gráfica de los datos originales
    for snid in supernova_ids:
        df_supernova_data = df_light_curves_cluster[df_light_curves_cluster['snid'] == snid]
        fig.add_trace(go.Scatter(
            x=df_supernova_data['days_relative_normalized'],
            y=df_supernova_data['mag_corregida'],
            mode='markers',
            name=f'SNID: {snid}',
            hoverinfo='text',
            text=df_supernova_data['snid']  # Información al pasar el mouse
        ))

    # Gráfica de la curva de ajuste
    fig.add_trace(go.Scatter(
        x=days_range.flatten(),
        y=predicted_magnitudes,
        mode='lines',
        name='Curva de Ajuste',
        line=dict(color='red')
    ))

    # Actualizar el layout
    fig.update_layout(
        title=f'Curva de Luz Ajustada para el Clúster {selected_cluster}',
        xaxis_title='Días Relativos Normalizados al Pico',
        yaxis_title='Magnitud Corregida',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
        showlegend=True
    )

    st.plotly_chart(fig)
else:
    st.write("No hay supernovas en este clúster.")
