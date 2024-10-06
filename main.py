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

def create_decl_vs_redshift_plot():
    # Obtener el valor máximo del redshift en los datos
    max_redshift = df_light_curves['redshift'].max()

    # Crear la gráfica de dispersión polar para Declinación vs Redshift
    fig = px.scatter_polar(df_light_curves, r='redshift', theta='decl', color='parsnip_pred', 
                           hover_data=['snid', 'redshift'], title='Polar Positions of Supernovae (Dec) vs Redshift')

    # Habilitar el botón de reset y zoom out desde la barra de herramientas
    fig.update_layout(
        title='Polar Positions of Supernovae (Dec) vs Redshift',
        autosize=True,
        polar=dict(
            radialaxis=dict(range=[0, max_redshift * 1.1], showticklabels=True),  # Ajustar rango según el máximo redshift
            angularaxis=dict(showticklabels=True)
        )
    )

    # Añadir botón personalizado para resetear el gráfico con un botón más pequeño y color naranja
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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Load your DataFrame (df_light_curves) here
# df_light_curves = ...

# Filter the MJD values for the slider
min_mjd = df_light_curves['mjd'].min()
max_mjd = df_light_curves['mjd'].max()

# Create a slider to select the Julian date
selected_mjd = st.slider("Select Modified Julian Date (MJD):", min_value=min_mjd, max_value=max_mjd)

# Prepare data for plotting
filtered_data = df_light_curves[df_light_curves['mjd'] <= selected_mjd]

# Visualización del gráfico
view_option = st.selectbox("Select visualization type:", ["Positions", "Heatmap", "Line Chart"])

# Inicializar un diccionario para contar las supernovas por tipo
type_counts = {'SN Ia': 0, 'SN II': 0, 'SN Ibc': 0}
counted_snids = set()

if not filtered_data.empty:
    # Conteo de supernovas por tipo
    for _, row in filtered_data.iterrows():
        snid = row['snid']
        supernova_type = row['parsnip_pred']

        if snid not in counted_snids:
            if supernova_type in type_counts:
                type_counts[supernova_type] += 1
            counted_snids.add(snid)

    # Histogram Data
    type_counts_df = pd.DataFrame(list(type_counts.items()), columns=['Supernova Type', 'Count'])

    if view_option == "Positions":
        # Crear gráfico de posiciones
        fig = go.Figure()

        for _, row in filtered_data.iterrows():
            ra = row['ra']
            decl = row['decl']
            supernova_type = row['parsnip_pred']
            color_map = {'SN Ia': 'blue', 'SN II': 'red', 'SN Ibc': 'green'}
            color = color_map.get(supernova_type, 'black')

            fig.add_trace(go.Scatter(
                x=[ra],
                y=[decl],
                mode='markers',
                marker=dict(size=10, color=color, symbol='star'),  # Use asterisks
                name=row['snid'],
                hoverinfo='text',
                text=f'SNID: {row["snid"]}, MJD: {selected_mjd}'  # Show SNID on hover
            ))

        fig.update_layout(title='Positions of Supernovae (RA vs Declination)',
                          xaxis_title='Right Ascension (RA)',
                          yaxis_title='Declination (Dec)',
                          showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    elif view_option == "Heatmap":
        # Crear un mapa de calor
        fig_density = px.density_heatmap(
            filtered_data,
            x='ra',
            y='decl',
            color_continuous_scale='Viridis',
            title='Densidad de Supernovas en RA y Dec',
            labels={'ra': 'Right Ascension (RA)', 'decl': 'Declination (Dec)'}
        )

        #fig_density.update_layout(
        #    title='Densidad de Supernovas en RA y Dec (hasta MJD seleccionada)',
        #    xaxis_title='Right Ascension (RA)',
        #    yaxis_title='Declination (Dec)',
        #)

        fig_density = px.density_heatmap(
            filtered_data,
            x='ra',
            y='decl',
            color_continuous_scale='Viridis',
            title='Densidad de Supernovas en RA y Dec',
            labels={'ra': 'Right Ascension (RA)', 'decl': 'Declination (Dec)'},
            histnorm='probability'  # Normalizar los conteos
        )
        
        st.plotly_chart(fig_density, use_container_width=True)

    elif view_option == "Line Chart":
        # Crear un gráfico de líneas
        mjd_counts = filtered_data['mjd'].value_counts().sort_index()

        fig_lines = go.Figure()
        fig_lines.add_trace(go.Scatter(
            x=mjd_counts.index,
            y=mjd_counts.values,
            mode='lines',
            name='Cantidad de Supernovas',
            line=dict(color='blue')
        ))

        fig_lines.update_layout(
            title='Evolución de Supernovas a lo Largo del Tiempo',
            xaxis_title='MJD',
            yaxis_title='Cantidad de Supernovas',
        )

        st.plotly_chart(fig_lines, use_container_width=True)

    # Mostrar el histograma
    fig_hist = go.Figure(data=[go.Bar(
        x=type_counts_df['Supernova Type'],
        y=type_counts_df['Count'],
        marker=dict(color=[{'SN Ia': 'blue', 'SN II': 'red', 'SN Ibc': 'green'}.get(t, 'black') for t in type_counts_df['Supernova Type']])
    )])

    fig_hist.update_layout(title='Count of Supernovae by Type (up to selected MJD)',
                           xaxis_title='Supernova Type',
                           yaxis_title='Count')

    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.write("No supernovae found for the selected MJD.")

##########______#############

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
#supernova_type = st.text_input("Enter the supernova type (e.g., 'SN Ia', 'SN Ib', 'SN II'):")
# Cambiar el input de texto a un menú desplegable para seleccionar el tipo de supernova
supernova_type = st.selectbox(
    "Select the supernova type:",
    options=['SN Ia', 'SN Ibc', 'SN II'],  # Lista de tipos de supernova
    index=0  # Índice por defecto seleccionado (SN Ia)
)

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

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st  # Importar Streamlit

# Supongamos que los dataframes df_parameters y df_light_curves ya están cargados
# Columnas en df_parameters: 'Redshift', 'SNID', 'peak_magnitude_r', 'peak_magnitude_z', 'peak_magnitude_X', 'peak_magnitude_Y', 'peak_magnitude_g'
# Columnas en df_light_curves: 'snid', 'parsnip_pred' (tipo de supernova)

# Constantes
c = 3e5  # Velocidad de la luz en km/s
H0 = 70  # Constante de Hubble en km/s/Mpc

# Paso 1: Calcular el módulo de la distancia para cada supernova
def calcular_modulo_distancia(redshift):
    # Calcular la distancia de luminosidad en Mpc
    DL_mpc = (c * redshift / H0) * (1 + redshift)
    # Convertir a parsecs
    DL_parsecs = DL_mpc * 1e6
    # Calcular el módulo de la distancia
    return 5 * np.log10(DL_parsecs) - 5

# Verificar que el dataframe tiene la columna 'Redshift'
if 'Redshift' in df_parameters.columns:
    # Aplicar la función para calcular el módulo de la distancia
    df_parameters['distance_modulus'] = df_parameters['Redshift'].apply(calcular_modulo_distancia)

    # Paso 2: Crear un diccionario que relacione 'snid' con 'parsnip_pred' en df_light_curves
    snid_to_type = dict(zip(df_light_curves['snid'], df_light_curves['parsnip_pred']))

    # Paso 3: Crear la columna SN_type en df_parameters usando el diccionario
    df_parameters['SN_type'] = df_parameters['SNID'].map(snid_to_type)

    # Verificar si hay valores nulos en la nueva columna SN_type
    if df_parameters['SN_type'].isnull().sum() > 0:
        st.write("Existen valores en df_parameters que no tienen un tipo de supernova asociado.")
        st.write(df_parameters[df_parameters['SN_type'].isnull()][['SNID']].head())  # Muestra algunas filas problemáticas

    # Menú desplegable para seleccionar el filtro de magnitud
    filtro_seleccionado = st.selectbox(
        'Seleccione el filtro de magnitud para graficar:',
        ('peak_magnitude_r', 'peak_magnitude_z', 'peak_magnitude_X', 'peak_magnitude_Y', 'peak_magnitude_g')
    )

    # Verificar si la columna seleccionada existe
    if filtro_seleccionado in df_parameters.columns:
        # Verificar si hay valores nulos en la columna de magnitud seleccionada
        st.write(f"Total de valores nulos en la magnitud {filtro_seleccionado}: {df_parameters[filtro_seleccionado].isnull().sum()}")

        # Filtrar las filas donde la magnitud seleccionada no sea nula
        df_filtrado = df_parameters.dropna(subset=[filtro_seleccionado, 'SN_type'])

        # Verificar cuántas supernovas de cada tipo hay
        st.write("Distribución de tipos de supernovas después del filtrado:")
        st.write(df_filtrado['SN_type'].value_counts())

        # Paso 4: Calcular la magnitud absoluta para el filtro seleccionado
        df_filtrado[f'absolute_magnitude_{filtro_seleccionado}'] = df_filtrado[filtro_seleccionado] - df_filtrado['distance_modulus']

        # Verificar si hay valores nulos en la magnitud absoluta
        if df_filtrado[f'absolute_magnitude_{filtro_seleccionado}'].isnull().any():
            st.write(f"Hay valores nulos en la columna de magnitud absoluta '{f'absolute_magnitude_{filtro_seleccionado}'}'.")
        else:
            # Paso 5: Crear el gráfico con Plotly
            fig = px.scatter(
                df_filtrado,
                x='distance_modulus',
                y=f'absolute_magnitude_{filtro_seleccionado}',
                color='SN_type',  # Usar diferentes colores según el tipo de supernova
                hover_data=['SNID', 'Redshift', 'SN_type'],
                labels={'distance_modulus': 'Distance Modulus', f'absolute_magnitude_{filtro_seleccionado}': f'Absolute Magnitude ({filtro_seleccionado})'},
                title=f'Absolute Magnitude ({filtro_seleccionado}) vs Distance Modulus for Supernovae'
            )

            # Invertir el eje Y porque las magnitudes menores son más brillantes
            fig.update_layout(
                yaxis=dict(autorange='reversed'),
                legend_title="Supernova Type",  # Título de la leyenda
                legend=dict(itemsizing='constant')  # Ajuste para la leyenda
            )

            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig)
    else:
        st.write(f"La columna seleccionada '{filtro_seleccionado}' no existe en el DataFrame.")
else:
    st.write("La columna 'Redshift' no está presente en el DataFrame.")


##---------------------------------

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st  # Importar Streamlit

# Supongamos que los dataframes df_parameters y df_light_curves ya están cargados
# Columnas en df_parameters: 'Redshift', 'SNID', 'peak_magnitude_r', 'peak_magnitude_z', 'peak_magnitude_X', 'peak_magnitude_Y', 'peak_magnitude_g'
# Columnas en df_light_curves: 'snid', 'parsnip_pred' (tipo de supernova)

# Constantes
c = 3e5  # Velocidad de la luz en km/s
H0 = 70  # Constante de Hubble en km/s/Mpc

# Paso 1: Calcular el módulo de la distancia para cada supernova
def calcular_modulo_distancia(redshift):
    # Calcular la distancia de luminosidad en Mpc
    DL_mpc = (c * redshift / H0) * (1 + redshift)
    # Convertir a parsecs
    DL_parsecs = DL_mpc * 1e6
    # Calcular el módulo de la distancia
    return 5 * np.log10(DL_parsecs) - 5

# Verificar que el dataframe tiene la columna 'Redshift'
if 'Redshift' in df_parameters.columns:
    # Aplicar la función para calcular el módulo de la distancia
    df_parameters['distance_modulus'] = df_parameters['Redshift'].apply(calcular_modulo_distancia)

    # Paso 2: Crear un diccionario que relacione 'snid' con 'parsnip_pred' en df_light_curves
    snid_to_type = dict(zip(df_light_curves['snid'], df_light_curves['parsnip_pred']))

    # Paso 3: Crear la columna SN_type en df_parameters usando el diccionario
    df_parameters['SN_type'] = df_parameters['SNID'].map(snid_to_type)

    # Verificar si hay valores nulos en la nueva columna SN_type
    if df_parameters['SN_type'].isnull().sum() > 0:
        st.write("Existen valores en df_parameters que no tienen un tipo de supernova asociado.")
        st.write(df_parameters[df_parameters['SN_type'].isnull()][['SNID']].head())  # Muestra algunas filas problemáticas

    # Menú desplegable para seleccionar el filtro de magnitud
    #filtro_seleccionado = st.selectbox(
    #    'Seleccione el filtro de magnitud para graficar:',
    #    ('peak_magnitude_r', 'peak_magnitude_z', 'peak_magnitude_X', 'peak_magnitude_Y', 'peak_magnitude_g')
    #)

    # Verificar si la columna seleccionada existe
    if filtro_seleccionado in df_parameters.columns:
        # Verificar si hay valores nulos en la columna de magnitud seleccionada
        st.write(f"Total de valores nulos en la magnitud {filtro_seleccionado}: {df_parameters[filtro_seleccionado].isnull().sum()}")

        # Filtrar las filas donde la magnitud seleccionada no sea nula
        df_filtrado = df_parameters.dropna(subset=[filtro_seleccionado, 'SN_type'])

        # Paso 4: Calcular la magnitud absoluta para el filtro seleccionado
        df_filtrado[f'absolute_magnitude_{filtro_seleccionado}'] = df_filtrado[filtro_seleccionado] - df_filtrado['distance_modulus']

        # Verificar cuántas supernovas de cada tipo hay
        st.write("Distribución de tipos de supernovas después del filtrado:")
        st.write(df_filtrado['SN_type'].value_counts())

        # Paso 5: Ajustar el número de bins con un deslizador
        num_bins = st.slider('Selecciona el número de bins para el histograma:', min_value=5, max_value=100, value=20, step=1)

        # Paso 6: Crear el histograma con la magnitud absoluta
        fig = px.histogram(
            df_filtrado,
            x=f'absolute_magnitude_{filtro_seleccionado}',
            nbins=num_bins,
            color='SN_type',  # Usar diferentes colores según el tipo de supernova
            labels={f'absolute_magnitude_{filtro_seleccionado}': f'Magnitud Absoluta {filtro_seleccionado}', 'count': 'Número de supernovas'},
            title=f'Histograma de Magnitud Absoluta ({filtro_seleccionado}) para Supernovas'
        )

        # Invertir el eje X porque las magnitudes menores son más brillantes
        fig.update_layout(
            xaxis=dict(autorange='reversed'),
            legend_title="Supernova Type",  # Título de la leyenda
            legend=dict(itemsizing='constant')  # Ajuste para la leyenda
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)
    else:
        st.write(f"La columna seleccionada '{filtro_seleccionado}' no existe en el DataFrame.")
else:
    st.write("La columna 'Redshift' no está presente en el DataFrame.")



##---------------------------------
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
st.write(df_parametros.head())




###############
##############

# Ahora, para agregar los resultados de df_parametros a df_parameters
df_parameters = pd.merge(df_parameters, df_parametros, on='SNID', how='left')
df_parameters.head()
# Muestra los resultados
#st.write(df_parameters.head())
##############
################



###############
################



st.write("""
## **Aglomerative clustering**

This section implements a process for performing hierarchical clustering on supernova data. It allows the user to select relevant columns for clustering and then displays the supernovae grouped into clusters. The data is normalized before clustering using StandardScaler, and the user can choose the number of clusters. After clustering, the supernovae in each cluster are displayed, and the user can select a specific supernova to view its light curve. Additionally, peak magnitudes are calculated for each filter, and corrections for extinction and redshift are applied to obtain absolute magnitudes.
""")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import streamlit as st

# Suponiendo que df_parameters ya está definido y tiene las columnas necesarias
# Elimina las filas con valores NaN en df_parameters
df_supernova_clustering = df_parameters.dropna()

# Lista de columnas que deseas filtrar
columns_to_exclude = ['SNID', 'peak_magnitude_X_x', 'peak_magnitude_Y_x', 'peak_magnitude_r_x', 'peak_magnitude_z_x', 'peak_magnitude_g_x', 'peak_magnitude_i_x', 'distance_modulus']  # Especifica las columnas a excluir

# Filtrar las columnas para excluir las no deseadas
filtered_columns = [col for col in df_supernova_clustering.columns if col not in columns_to_exclude]

# Permitir que el usuario seleccione columnas específicas para el clustering
selected_columns = st.multiselect(
    'Select columns for clustering:',
    options=filtered_columns,  # Columnas filtradas
    default=filtered_columns     # Por defecto, selecciona todas las columnas filtradas
)

# Mostrar las columnas seleccionadas
st.write("Selected columns for clustering:", selected_columns)

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
    #st.write("Supernovae in each cluster:")
    #for cluster, supernovae in supernova_names_clusters.items():
    #    st.write(f"{cluster}: {supernovae}")

else:
    st.write("Please select at least one column for clustering.")



##############

# Show a dropdown menu for the user to choose the cluster
selected_cluster = st.selectbox(
    "Select the cluster to view a list of supernovae:",
    list(supernova_names_clusters.keys())  # Show available clusters
)

# Get the supernovae in the selected cluster
supernovae_in_cluster = supernova_names_clusters[selected_cluster]

if len(supernovae_in_cluster) > 0:
    # Show the number of supernovae in the cluster
    st.write(f"Found {len(supernovae_in_cluster)} supernovae in {selected_cluster}.")
    
    # Horizontal slider to select a specific supernova in the cluster
    selected_index = st.slider('Select one supernova to view its light curve:',
                               min_value=0, max_value=len(supernovae_in_cluster)-1, step=1)
    
    # Get the SNID of the selected supernova
    selected_snid = supernovae_in_cluster[selected_index]
    
    # Filter the light curve DataFrame (df_light_curves) by the selected SNID
    df_selected_supernova = df_light_curves[df_light_curves['snid'] == selected_snid]
    
    # Verify that the supernova has data to plot
    if not df_selected_supernova.empty:
        # Plot the light curve of the selected supernova
        st.plotly_chart(plot_light_curve(df_selected_supernova), key="light_curve_plot")
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
#st.write(df_supernova_clustering)
########
#st.write(df_supernova_clustering.columns)

#st.write("Supernovae in each cluster:")
#for cluster, supernovae in supernova_names_clusters.items():
#    with st.expander(f"{cluster} (Total: {len(supernovae)} supernovae)"):
#        st.write(supernovae)

# Crear un solo expander para todos los clusters
with st.expander("Supernovae in each cluster"):
    for cluster, supernovae in supernova_names_clusters.items():
        st.write(f"{cluster} (Total: {len(supernovae)} supernovae):")
        st.write(supernovae)

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st  # Importar Streamlit

# Supongamos que los dataframes df_parameters, df_light_curves y df_supernova_clustering ya están cargados
# Columnas en df_parameters: 'Redshift', 'SNID', 'peak_magnitude_r_y', 'peak_magnitude_z_y', 'peak_magnitude_X_y', 'peak_magnitude_Y_y', 'peak_magnitude_g_y'
# Columnas en df_light_curves: 'snid', 'parsnip_pred', 'mwebv' (valor de extinción)
# Columnas en df_supernova_clustering: 'SNID', 'cluster'

# Constantes
c = 3e5  # Velocidad de la luz en km/s
H0 = 70  # Constante de Hubble en km/s/Mpc

# Función para corregir la magnitud por extinción
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

# Paso 1: Calcular el módulo de la distancia para cada supernova
def calcular_modulo_distancia(redshift):
    # Calcular la distancia de luminosidad en Mpc
    DL_mpc = (c * redshift / H0) * (1 + redshift)
    # Convertir a parsecs
    DL_parsecs = DL_mpc * 1e6
    # Calcular el módulo de la distancia
    return 5 * np.log10(DL_parsecs) - 5

# Verificar que el dataframe tiene la columna 'Redshift'
if 'Redshift' in df_parameters.columns:
    # Aplicar la función para calcular el módulo de la distancia
    df_parameters['distance_modulus'] = df_parameters['Redshift'].apply(calcular_modulo_distancia)

    # Paso 2: Crear un diccionario que relacione 'snid' con 'parsnip_pred' en df_light_curves
    snid_to_type = dict(zip(df_light_curves['snid'], df_light_curves['parsnip_pred']))

    # Paso 3: Crear la columna SN_type en df_parameters usando el diccionario
    df_parameters['SN_type'] = df_parameters['SNID'].map(snid_to_type)

    # Paso 4: Crear un diccionario que relacione 'SNID' con 'cluster' en df_supernova_clustering
    snid_to_cluster = dict(zip(df_supernova_clustering['SNID'], df_supernova_clustering['cluster']))

    # Paso 5: Crear la columna 'cluster' en df_parameters usando el diccionario de clústeres
    df_parameters['cluster'] = df_parameters['SNID'].map(snid_to_cluster)

    # Paso 6: Extraer el valor de mwebv de df_light_curves y mapearlo a df_parameters usando 'SNID'
    snid_to_mwebv = dict(zip(df_light_curves['snid'], df_light_curves['mwebv']))
    df_parameters['mwebv'] = df_parameters['SNID'].map(snid_to_mwebv)

    # Verificar si hay valores nulos en la nueva columna SN_type, cluster o mwebv
    if df_parameters['SN_type'].isnull().sum() > 0:
        st.write("Existen valores en df_parameters que no tienen un tipo de supernova asociado.")
        st.write(df_parameters[df_parameters['SN_type'].isnull()][['SNID']].head())  # Muestra algunas filas problemáticas
    if df_parameters['cluster'].isnull().sum() > 0:
        st.write("Existen valores en df_parameters que no tienen un clúster asociado.")
        st.write(df_parameters[df_parameters['cluster'].isnull()][['SNID']].head())  # Muestra algunas filas problemáticas
    if df_parameters['mwebv'].isnull().sum() > 0:
        st.write("Existen valores en df_parameters que no tienen un valor de extinción mwebv asociado.")
        st.write(df_parameters[df_parameters['mwebv'].isnull()][['SNID']].head())  # Muestra algunas filas problemáticas

    # Menú desplegable para seleccionar el filtro de magnitud (modificado para reflejar los nombres con '_y')
    filtro_seleccionado = st.selectbox(
        'Filtro de magnitud para graficar:',
        ('peak_magnitude_r_y', 'peak_magnitude_z_y', 'peak_magnitude_X_y', 'peak_magnitude_Y_y', 'peak_magnitude_g_y')
    )

    # Verificar si la columna seleccionada y otras necesarias existen en el DataFrame
    columnas_necesarias = [filtro_seleccionado, 'SN_type', 'cluster', 'mwebv']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df_parameters.columns]

    if len(columnas_faltantes) == 0:
        # Filtrar las filas donde no haya nulos en las columnas necesarias
        df_filtrado = df_parameters.dropna(subset=columnas_necesarias)

        # Paso 7: Corregir la magnitud por extinción antes de calcular la magnitud absoluta
        df_filtrado[f'mag_corregida_{filtro_seleccionado}'] = df_filtrado.apply(
            lambda row: corregir_magnitud_extincion(row[filtro_seleccionado], row['mwebv'], filtro=filtro_seleccionado.split('_')[2]),
            axis=1
        )

        # Calcular la magnitud absoluta con la magnitud corregida
        df_filtrado[f'absolute_magnitude_{filtro_seleccionado}'] = df_filtrado[f'mag_corregida_{filtro_seleccionado}']

        # Verificar cuántas supernovas de cada tipo hay
        st.write("Distribución de tipos de supernovas después del filtrado:")
        st.write(df_filtrado['SN_type'].value_counts())

        # Crear el gráfico de magnitud absoluta vs. módulo de distancia (coloreado por clúster)
        fig_magnitude_vs_modulus = px.scatter(
            df_filtrado,
            x='distance_modulus',
            y=f'absolute_magnitude_{filtro_seleccionado}',
            color=df_filtrado['cluster'].astype(str),  # Convertir cluster a string para colores discretos
            hover_data=['SNID', 'Redshift', 'SN_type', 'cluster'],
            labels={'distance_modulus': 'Distance Modulus', f'absolute_magnitude_{filtro_seleccionado}': f'Absolute Magnitude ({filtro_seleccionado})'},
            title=f'Absolute Magnitude ({filtro_seleccionado}) vs Distance Modulus for Supernovae (by Cluster)'
        )


        # Invertir el eje Y porque las magnitudes menores son más brillantes
        fig_magnitude_vs_modulus.update_layout(
            yaxis=dict(autorange='reversed'),
            legend_title="Cluster",  # Título de la leyenda
            showlegend=True  # Mostrar leyenda
        )

        # Hacer clic en los labels para mostrar u ocultar
        fig_magnitude_vs_modulus.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
        fig_magnitude_vs_modulus.update_layout(
            legend=dict(
                itemclick="toggle",  # Permitir mostrar/ocultar al hacer clic
                itemdoubleclick="toggleothers"  # Doble clic para mostrar solo un grupo
            )
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig_magnitude_vs_modulus)

        # Paso 8: Ajustar el número de bins con un deslizador
        num_bins = st.slider('Bins para el histograma:', min_value=5, max_value=100, value=20, step=1)

        # Paso 9: Crear el histograma con la magnitud absoluta corregida, coloreando por clúster
        fig_histogram = px.histogram(
            df_filtrado,
            x=f'absolute_magnitude_{filtro_seleccionado}',
            nbins=num_bins,
            color='cluster',  # Usar diferentes colores según el clúster
            labels={f'absolute_magnitude_{filtro_seleccionado}': f'Magnitud Absoluta Corregida {filtro_seleccionado}', 'count': 'Número de supernovas'},
            title=f'Histograma de Magnitud Absoluta Corregida ({filtro_seleccionado}) para Supernovas (por Clúster)'
        )

        # Invertir el eje X porque las magnitudes menores son más brillantes
        fig_histogram.update_layout(
            xaxis=dict(autorange='reversed'),
            legend_title="Cluster"
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig_histogram)

    else:
        st.write(f"Faltan las siguientes columnas necesarias en el DataFrame: {', '.join(columnas_faltantes)}")
else:
    st.write("La columna 'Redshift' no está presente en el DataFrame.")


st.write("""

## **Blox plots for each cluster**

Generates a set of comparative boxplots between supernova clusters, allowing the user to select which numerical columns from the DataFrame to include in the plots. Each plot shows a boxplot comparing the values of that column between different clusters. When hovering over the points in the boxplots, additional information is displayed (the value of the 'SNID' column).
""")


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

st.write("""

## **t-SNE Plot**

This section creates a t-SNE visualization for dimensionality reduction and plots the supernovae data in two dimensions. A t-SNE instance is created with specific hyperparameters (perplexity, early exaggeration, and learning rate) to transform the PCA-reduced data into a 2D space. The resulting 2D points are grouped by their clusters, and each cluster is represented with a different marker on a scatter plot. The hover tool displays additional information for each supernova, including its 'SNID', 'RA', 'Dec', and 'Redshift'. The layout is customized with titles and a legend showing the cluster IDs.
""")


# Create a t-SNE instance with the desired hyperparameters
#tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=10, learning_rate=5)

# Fit t-SNE to the PCA data (assuming pca_data has already been calculated)
#tsne_data = tsne.fit_transform(pca_data)



# Assuming you have your PCA data
num_samples = pca_data.shape[0]  # Number of samples after PCA
default_perplexity = 30  # Set a default value for perplexity

# Dynamically adjust the perplexity
if num_samples < default_perplexity:
    perplexity = max(1, num_samples - 1)  # Ensure it's at least 1 and less than n_samples
else:
    perplexity = default_perplexity

# Now apply t-SNE with the dynamically determined perplexity
tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=10, learning_rate=5)
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

st.write("""

## **Regression Tree clasifier**

Here we implement a decision tree classifier using the `scikit-learn` library within a Streamlit application. 
It allows users to select numerical features from the `df_supernova_clustering` DataFrame to train the model, while 
excluding non-numerical columns like Right Ascension ('RA') and Declination ('Dec').


""")

with st.expander("**Key features**"):
    st.write("""
    - **Feature Selection:** Users can choose which numerical variables to include in the model through a multi-select dropdown menu.
    - **Data Splitting:** The selected features are split into training and test datasets to evaluate the model's performance.
    - **Model Training:** A decision tree classifier is trained on the selected features, with adjustable hyperparameters to control tree depth and minimum samples for splits and leaves.
    - **Visualization:** The learned decision rules from the classifier are displayed, providing insight into how the model makes predictions.
    - **Model Evaluation:** The accuracy of the model on the test set is calculated and displayed, giving users feedback on its predictive performance.""")


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import streamlit as st

# Select numerical columns, excluding 'RA' and 'Dec'
numerical_columns = df_supernova_clustering.select_dtypes(include=['number']).drop(columns=['RA', 'Dec', 'cluster', 'peak_magnitude_X_x', 'peak_magnitude_Y_x', 'peak_magnitude_z_x', 'peak_magnitude_g_x', 'peak_magnitude_r_x', 'peak_magnitude_i_x'])

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

#from sklearn.tree import DecisionTreeRegressor
#import numpy as np
#import pandas as pd
#import streamlit as st
#import plotly.graph_objects as go

# Paso 1: Filtrar las supernovas del clúster seleccionado
#selected_cluster = st.selectbox("Select the cluster to analyze:", df_supernova_clustering['cluster'].unique())
#df_clustered_supernovae = df_supernova_clustering[df_supernova_clustering['cluster'] == selected_cluster]

# Paso 2: Preparar los datos
#if not df_clustered_supernovae.empty:
#    # Unir todos los datos de supernovas en el clúster
#    supernova_ids = df_clustered_supernovae['SNID'].unique()
#    df_light_curves_cluster = df_light_curves[df_light_curves['snid'].isin(supernova_ids)]

    # Calcular días relativos al pico de luminosidad
#    df_light_curves_cluster = calculate_days_relative_to_peak(df_light_curves_cluster)

    # Paso 2.1: Filtrar para usar solo los datos desde el pico hasta el final
#    df_light_curves_cluster = df_light_curves_cluster[df_light_curves_cluster['days_relative'] >= 0]

    # Normalizar los días relativos al pico
#    df_light_curves_cluster['days_relative_normalized'] = df_light_curves_cluster.groupby('snid')['days_relative'].transform(
#        lambda x: (x - x.min()) / (x.max() - x.min())  # Normalización entre 0 y 1
#    )

    # Crear el conjunto de entrenamiento
#    X = df_light_curves_cluster[['days_relative_normalized']]  # Días relativos normalizados
#    y = df_light_curves_cluster['mag']  # Magnitudes observadas

    # Paso 3: Entrenar el modelo de árbol de regresión
#    model = DecisionTreeRegressor(random_state=42)
#    model.fit(X, y)

    # Paso 4: Predecir las magnitudes para un rango de días relativos normalizados
#    days_range = np.linspace(X['days_relative_normalized'].min(), X['days_relative_normalized'].max(), 100).reshape(-1, 1)
#    predicted_magnitudes = model.predict(days_range)

    # Paso 5: Graficar la curva de luz ajustada
#    fig = go.Figure()
    
    # Gráfica de los datos originales
#    for snid in supernova_ids:
#        df_supernova_data = df_light_curves_cluster[df_light_curves_cluster['snid'] == snid]
#        fig.add_trace(go.Scatter(
#            x=df_supernova_data['days_relative_normalized'],
#            y=df_supernova_data['mag'],
#            mode='markers',
#            name=f'SNID: {snid}',
#            hoverinfo='text',
#            text=df_supernova_data['snid']  # Información al pasar el mouse
#        ))

#    # Gráfica de la curva de ajuste
#    fig.add_trace(go.Scatter(
#        x=days_range.flatten(),
#        y=predicted_magnitudes,
#        mode='lines',
#        name='Curva de Ajuste',
#        line=dict(color='red')
#    ))

#    # Actualizar el layout
#    fig.update_layout(
#        title=f'Curva de Luz Ajustada para el Clúster {selected_cluster}',
#        xaxis_title='Días Relativos Normalizados al Pico',
#        yaxis_title='Magnitud',
#        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
#        showlegend=True
#    )

#    st.plotly_chart(fig)
#else:
#    st.write("No hay supernovas en este clúster.")



#=#

###---###---###

st.write("""
Here we implement a regression analysis of supernova light curves using a Decision Tree Regressor from the `scikit-learn` library within a Streamlit application. It allows users to analyze the light curves of supernovae within a selected cluster and visualize the adjusted light curve based on corrected magnitudes.
""")


from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeRegressor

##
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter  # Para suavizar la curva

# Paso 1: Filtrar las supernovas del clúster seleccionado
selected_cluster = st.selectbox("Select a cluster to analyze:", df_supernova_clustering['cluster'].unique())
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

    # Paso 3: Crear el deslizador para seleccionar una supernova
    selected_snid_index = st.slider('Select one supernova to show its light curve:', 
                                    min_value=0, max_value=len(supernova_ids)-1, step=1, key="supernova_slider")
    selected_snid = supernova_ids[selected_snid_index]

    # Paso 4: Graficar la curva de luz original para la supernova seleccionada
    df_supernova_data = df_light_curves_cluster[df_light_curves_cluster['snid'] == selected_snid]

    # Comprobar si hay suficientes puntos de datos
    if df_supernova_data.shape[0] < 2 or df_supernova_data['mag_corregida'].isnull().all():
        st.write(f"Not enough data points to fit a model for supernova {selected_snid}.")
    else:
        # Crear subplots lado a lado
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Original Light Curve for SNID {selected_snid}", 
                                                            f"Fitted Curve for SNID {selected_snid}"))

        # Graficar la curva de luz original a la izquierda
        fig.add_trace(go.Scatter(
            x=df_supernova_data['days_relative_normalized'],
            y=df_supernova_data['mag_corregida'],
            mode='markers',
            name='Original Data',
            hoverinfo='text',
            text=df_supernova_data['snid'],  # Información al pasar el mouse
            marker=dict(size=5)
        ), row=1, col=1)

        # Entrenar el modelo de árbol de regresión para la supernova seleccionada
        X = df_supernova_data[['days_relative_normalized']]
        y = df_supernova_data['mag_corregida']
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X, y)

        # Predecir las magnitudes para un rango de días relativos normalizados
        days_range = np.linspace(X['days_relative_normalized'].min(), X['days_relative_normalized'].max(), 100).reshape(-1, 1)
        predicted_magnitudes = model.predict(days_range)

        # Verificar si hay suficientes puntos para aplicar el filtro de Savitzky-Golay
        window_length = 11  # Longitud de la ventana por defecto
        if len(predicted_magnitudes) < window_length:
            window_length = len(predicted_magnitudes) - 1  # Ajustar longitud de ventana si hay menos puntos

        # Aplicar el filtro de Savitzky-Golay para suavizar la curva ajustada
        smoothed_magnitudes = savgol_filter(predicted_magnitudes, window_length=window_length, polyorder=3)

        # Graficar la curva de ajuste a la derecha
        fig.add_trace(go.Scatter(
            x=days_range.flatten(),
            y=predicted_magnitudes,
            mode='lines',
            name='Fitted Curve',
            line=dict(width=2, color='red')
        ), row=1, col=2)

        # Graficar la curva suavizada
        fig.add_trace(go.Scatter(
            x=days_range.flatten(),
            y=smoothed_magnitudes,
            mode='lines',
            name='Smoothed Curve',
            line=dict(width=2, color='blue')
        ), row=1, col=2)

        # Actualizar el layout
        fig.update_layout(
            title=f'Light Curve and Fitted Curve for Supernova {selected_snid}',
            xaxis_title='Normalized Days Relative to Peak',
            yaxis_title='Corrected Magnitude',
            yaxis=dict(autorange='reversed'),  # Invertir el eje Y en la gráfica de la izquierda
            yaxis2=dict(autorange='reversed'),  # Invertir el eje Y en la gráfica de la derecha
            showlegend=False
        )

        # Mostrar las gráficas lado a lado
        st.plotly_chart(fig, use_container_width=True)

else:
    st.write("No supernovas found in this cluster.")

##

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter  # Para suavizar la curva

# Paso 1: Filtrar las supernovas del clúster seleccionado
#selected_cluster = st.selectbox("Please, select the cluster to analyze:", df_supernova_clustering['cluster'].unique())
df_clustered_supernovae = df_supernova_clustering[df_supernova_clustering['cluster'] == selected_cluster]

# Inicializar la lista de supernovas con datos insuficientes
insufficient_data_supernovas = []

# Inicializar la figura para graficar las curvas suavizadas de todas las supernovas
fig_all_smoothed = go.Figure()

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
    # Input boxes para los parámetros del filtro Savitzky-Golay
    window_length = st.number_input("Tamaño de la ventana (debe ser impar)", value=11, min_value=3, step=2)
    polyorder = st.number_input("Orden del polinomio", value=3, min_value=1)

    # Paso 3: Recorrer todas las supernovas dentro del clúster y calcular las curvas suavizadas
    for snid in supernova_ids:
        df_supernova_data = df_light_curves_cluster[df_light_curves_cluster['snid'] == snid]

        # Verificar si hay suficientes puntos de datos para entrenar el modelo
        if df_supernova_data.shape[0] < 2:
            insufficient_data_supernovas.append(snid)  # Agregar a la lista de supernovas con datos insuficientes
        else:
            # Entrenar el modelo de árbol de regresión para la supernova
            X = df_supernova_data[['days_relative_normalized']]
            y = df_supernova_data['mag_corregida']
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X, y)

            # Predecir las magnitudes para un rango de días relativos normalizados
            days_range = np.linspace(X['days_relative_normalized'].min(), X['days_relative_normalized'].max(), 100).reshape(-1, 1)
            predicted_magnitudes = model.predict(days_range)

            # Aplicar el filtro de Savitzky-Golay para suavizar la curva ajustada
            #smoothed_magnitudes = savgol_filter(predicted_magnitudes, window_length=11, polyorder=3)
            smoothed_magnitudes = savgol_filter(predicted_magnitudes, window_length=window_length, polyorder=polyorder)

            # Añadir la curva suavizada al gráfico que contendrá todas las curvas
            fig_all_smoothed.add_trace(go.Scatter(
                x=days_range.flatten(),
                y=smoothed_magnitudes,
                mode='lines',
                name=f'SNID: {snid}',
                line=dict(width=2),
                hoverinfo='text',
                text=[f'SNID: {snid}']*len(days_range)  # Mostrar el SNID en el hover
            ))

    # Mostrar la lista de supernovas con datos insuficientes (si las hay)
    if insufficient_data_supernovas:
        st.write("Supernovas with insufficient data points:")
        st.write(insufficient_data_supernovas)

    # Mostrar todas las curvas suavizadas juntas
    fig_all_smoothed.update_layout(
        title='Smoothed Light Curves for Supernovae in Cluster',
        xaxis_title='Normalized Days Relative to Peak',
        yaxis_title='Corrected Magnitude',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y para que las magnitudes más brillantes estén arriba
        showlegend=True
    )

    st.plotly_chart(fig_all_smoothed, use_container_width=True)

else:
    st.write("No supernovas found in this cluster.")

###########

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter  # Para suavizar la curva

# Paso 1: Filtrar las supernovas del clúster seleccionado
df_clustered_supernovae = df_supernova_clustering[df_supernova_clustering['cluster'] == selected_cluster]

# Inicializar la lista de supernovas con datos insuficientes
insufficient_data_supernovas = []

# Inicializar la figura para graficar las curvas suavizadas de todas las supernovas
fig_all_smoothed = go.Figure()

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
    
    # Input boxes para los parámetros del filtro Savitzky-Golay
    window_length = st.number_input("Ventana (debe ser impar)", value=11, min_value=3, step=2)
    polyorder = st.number_input("Orden", value=3, min_value=1)

    # Lista para almacenar las características de cada supernova
    features = []
    all_smoothed_data = []  # Para almacenar las curvas suavizadas

    # Paso 3: Recorrer todas las supernovas dentro del clúster y calcular las curvas suavizadas
    for snid in supernova_ids:
        df_supernova_data = df_light_curves_cluster[df_light_curves_cluster['snid'] == snid]

        # Verificar si hay suficientes puntos de datos para entrenar el modelo
        if df_supernova_data.shape[0] < 2:
            insufficient_data_supernovas.append(snid)  # Agregar a la lista de supernovas con datos insuficientes
        else:
            # Entrenar el modelo de árbol de regresión para la supernova
            X = df_supernova_data[['days_relative_normalized']]
            y = df_supernova_data['mag_corregida']
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X, y)

            # Predecir las magnitudes para un rango de días relativos normalizados
            days_range = np.linspace(X['days_relative_normalized'].min(), X['days_relative_normalized'].max(), 100).reshape(-1, 1)
            predicted_magnitudes = model.predict(days_range)

            # Aplicar el filtro de Savitzky-Golay para suavizar la curva ajustada
            smoothed_magnitudes = savgol_filter(predicted_magnitudes, window_length=window_length, polyorder=polyorder)

            # Verificar si hay valores NaN en la curva suavizada
            if np.any(np.isnan(smoothed_magnitudes)):
                st.warning(f"La curva para SNID: {snid} contiene valores NaN y no será graficada.")
                continue  # Saltar esta curva si contiene NaN
            
            # Añadir la curva suavizada al gráfico que contendrá todas las curvas
            fig_all_smoothed.add_trace(go.Scatter(
                x=days_range.flatten(),
                y=smoothed_magnitudes,
                mode='lines',
                name=f'SNID: {snid}',
                line=dict(width=2),
                hoverinfo='text',
                text=[f'SNID: {snid}']*len(days_range)  # Mostrar el SNID en el hover
            ))
            
            # Calcular características
            peak_magnitude = np.max(smoothed_magnitudes)  # Pico de luminosidad
            half_peak_magnitude = peak_magnitude * 0.5  # Calcular el 50% del pico

            # Encontrar el tiempo hasta que la magnitud alcanza el 50% del pico
            crossing_indices = np.where(smoothed_magnitudes <= half_peak_magnitude)[0]
            time_to_half_peak = days_range[crossing_indices[0]][0] if crossing_indices.size > 0 else None
            
            area_under_curve = np.trapz(smoothed_magnitudes, days_range.flatten())  # Área bajo la curva

            # Almacenar las características en la lista
            features.append([peak_magnitude, time_to_half_peak, area_under_curve])
            all_smoothed_data.append(smoothed_magnitudes)  # Almacenar las curvas suavizadas

    # Mostrar la lista de supernovas con datos insuficientes (si las hay)
    if insufficient_data_supernovas:
        st.write("Supernovas con datos insuficientes:")
        st.write(insufficient_data_supernovas)

    # Mostrar todas las curvas suavizadas juntas
    fig_all_smoothed.update_layout(
        title='Smoothed Light Curves for Supernovae in Cluster',
        xaxis_title='Normalized Days Relative to Peak',
        yaxis_title='Corrected Magnitude',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y para que las magnitudes más brillantes estén arriba
        showlegend=True
    )

    st.plotly_chart(fig_all_smoothed, use_container_width=True, key="grafica")

    # Clasificación de perfiles comunes mediante clustering
    if len(features) > 1:  # Asegurarse de que hay suficientes supernovas para el clustering
        # Convertir la lista de características a un DataFrame
        df_features = pd.DataFrame(features, columns=['peak_magnitude', 'time_to_half_peak', 'area_under_curve'])

        # Escalar las características
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_features)

        # Aplicar K-Means para clasificar las supernovas en perfiles comunes
        k = st.number_input("Núm de perfiles (clusters)", value=3, min_value=1, step=1)
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            df_features['cluster'] = kmeans.fit_predict(scaled_features)

            # Mostrar los resultados de clasificación
            st.write("Clasificación de supernovas en perfiles comunes:")
            for cluster in range(k):
                st.write(f"Cluster {cluster}:")
        
                # Encontrar los índices de las supernovas en el cluster
                cluster_indices = df_features.index[df_features['cluster'] == cluster].tolist()
        
                # Obtener los SNID de las supernovas en el cluster
                cluster_ids = df_light_curves_cluster['snid'].unique()[cluster_indices]

                # Comprobar si hay suficientes curvas para calcular la curva promedio
                if len(cluster_indices) > 1:
                    # Filtrar solo curvas válidas (sin NaN) para calcular el promedio
                    valid_curves = [all_smoothed_data[i] for i in cluster_indices if np.all(np.isfinite(all_smoothed_data[i]))]
                    if valid_curves:
                        avg_smoothed_curve = np.mean(valid_curves, axis=0)
                        # Reemplazar valores NaN con el promedio de los puntos adyacentes
                        avg_smoothed_curve = np.where(
                            np.isnan(avg_smoothed_curve),
                            np.where(
                                np.arange(len(avg_smoothed_curve)) > 0, 
                                (np.roll(avg_smoothed_curve, 1) + avg_smoothed_curve) / 2, 
                                avg_smoothed_curve
                            ),
                            avg_smoothed_curve
                        )
                    else:
                        st.warning(f"No hay curvas válidas para el cluster {cluster}.")
                        continue  # Si no hay curvas válidas, saltar este cluster
                else:
                    # Si hay solo una curva, utilizarla como promedio
                    avg_smoothed_curve = all_smoothed_data[cluster_indices[0]]

                # Crear subgráfica para las curvas individuales y la curva promedio
                fig_cluster = make_subplots(rows=1, cols=2, subplot_titles=(f'Cluster {cluster} - Average Curve', 'Individual Curves'))
                
                # Graficar la curva promedio
                fig_cluster.add_trace(go.Scatter(
                    x=days_range.flatten(),
                    y=avg_smoothed_curve,
                    mode='lines',
                    name='Average Curve',
                    line=dict(width=2)
                ), row=1, col=1)

                # Graficar las curvas individuales en la segunda subgráfica
                for i in cluster_indices:
                    individual_curve = all_smoothed_data[i]
                    # Asegurarse de que la curva no tenga valores NaN
                    if np.any(np.isnan(individual_curve)):
                        st.warning(f"La curva para SNID: {df_light_curves_cluster['snid'].unique()[i]} contiene valores NaN y no será graficada.")
                        continue

                    fig_cluster.add_trace(go.Scatter(
                        x=days_range.flatten(),
                        y=individual_curve,
                        mode='lines',
                        name=f'SNID: {df_light_curves_cluster["snid"].unique()[i]}',
                        line=dict(width=1, dash='dot')  # Línea punteada para curvas individuales
                    ), row=1, col=2)

                # Invertir el eje Y para la gráfica de curvas individuales
                fig_cluster.update_yaxes(autorange='reversed', row=1, col=2)

                # Configurar la visualización de la subgráfica
                fig_cluster.update_layout(
                    title=f'Supernovae Cluster {cluster}',
                    yaxis_title='Corrected Magnitude',
                    yaxis=dict(autorange='reversed'),  # Invertir el eje Y
                    showlegend=True
                )

                # Mostrar la subgráfica en Streamlit
                st.plotly_chart(fig_cluster, use_container_width=True, key=f'cluster_plot_{cluster}')
        except ValueError as e:
            st.error(f"Error al aplicar K-Means: {e}")

    else:
        st.write("No se encontraron supernovas suficientes para clasificar.")

################################################################################################################################################################

# Function to read the downloaded supernova file and extract relevant data, including parsnip_pred
def read_supernova_file_content(content):
    # Variables and lists to store data
    mjd, flx, flxerr, mag, magerr, filters = [], [], [], [], [], []
    snid, parsnip_pred, superraenn_pred, ra, decl, redshift, mwebv = None, None, None, None, None, None, None

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
        elif line.startswith("PARSNIP_PRED:"):
            parsnip_pred = ' '.join(line.split()[1:])  # Extract parsnip_pred value
        elif line.startswith("OBS:"):  # Extract observations
            data = line.split()
            mjd.append(convert_to_float(data[1]))  # MJD (Modified Julian Date)
            filters.append(data[2])     # Filter (g, r, i, z, etc.)
            flx.append(convert_to_float(data[4]))  # Flux (FLUXCAL)
            flxerr.append(convert_to_float(data[5]))  # Flux error (FLUXCALERR)
            mag.append(convert_to_float(data[6]))  # Magnitude (MAG)
            magerr.append(convert_to_float(data[7]))  # Magnitude error (MAGERR)

    return mjd, flx, flxerr, mag, magerr, filters, snid, ra, decl, redshift, mwebv, parsnip_pred

# Function to store light curves as a DataFrame, including parsnip_pred
def save_flux_curves_as_dataframe(vector_list, file_name, mjd, flx, flxerr, mag, magerr, filters, snid, ra, decl, redshift, mwebv, parsnip_pred):
    for i in range(len(mjd)):
        curve_vector = {
            'file_name': file_name,
            'snid': snid,
            'mjd': mjd[i],
            'filter': filters[i],
            'flx': flx[i],
            'flxerr': flxerr[i],
            'mag': mag[i],
            'magerr': magerr[i],
            'ra': ra,
            'decl': decl,
            'redshift': redshift,
            'mwebv': mwebv,
            'parsnip_pred': parsnip_pred  # Include the parsnip_pred variable
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
            mjd, flx, flxerr, mag, magerr, filters, snid, ra, decl, redshift, mwebv, parsnip_pred = read_supernova_file_content(content)
            save_flux_curves_as_dataframe(vector_list, file_name, mjd, flx, flxerr, mag, magerr, filters, snid, ra, decl, redshift, mwebv, parsnip_pred)

    return pd.DataFrame(vector_list)

# Load supernova data from GitHub
st.write("Downloading and processing supernova files...")
repo_url = "https://github.com/SArcD/supernovaIA"
df_flux = download_and_process_supernovas(repo_url)

# Display resulting DataFrame
st.write(df_flux)

# Display resulting DataFrame
#st.write(df_flux)

# Save data to a CSV file
#df_flux.to_csv('flux_curves_with_magnitudes.csv', index=False)
#st.write("Data saved in 'flux_curves_with_magnitudes.csv'.")

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import streamlit as st

# Configuración de un modelo cosmológico
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Paso 1: Calcular la distancia de luminosidad (en parsecs) a partir del redshift
df_flux['D_L_mpc'] = cosmo.luminosity_distance(df_flux['redshift']).value  # en Mpc
df_flux['D_L_pc'] = df_flux['D_L_mpc'] * 10**6  # Convertir Mpc a parsecs

# Paso 2: Calcular el módulo de la distancia
df_flux['distance_modulus'] = 5 * np.log10(df_flux['D_L_pc']) - 5  # Módulo de la distancia

# Paso 3: Aplicar la corrección por redshift y extinción
def corregir_magnitud_redshift(m_corregida, z):
    K_correction = 0.1 * z  # K-correction simplificada
    return m_corregida - K_correction

df_flux['mag_corr_ext'] = df_flux.apply(
    lambda row: corregir_magnitud_extincion(row['mag'], row['mwebv'], row['filter']),
    axis=1
)

df_flux['mag_corr'] = df_flux.apply(
    lambda row: corregir_magnitud_redshift(row['mag_corr_ext'], row['redshift']),
    axis=1
)

# Paso 4: Calcular la magnitud absoluta y aplicar la corrección bolométrica
df_flux['mag_abs'] = df_flux['mag_corr'] - df_flux['distance_modulus']

def apply_bolometric_correction(row):
    sn_type = row.get('parsnip_pred', 'Unknown')  # Default to 'Unknown' if missing
    if sn_type == 'SN Ia':
        return -0.1
    elif sn_type == 'SN II':
        return -0.5
    elif sn_type == 'SN Ibc':
        return -0.3
    else:
        return 0

df_flux['bolometric_correction'] = df_flux.apply(apply_bolometric_correction, axis=1)
df_flux['mag_bol'] = df_flux['mag_abs'] + df_flux['bolometric_correction']

# Paso 5: Calcular la luminosidad bolométrica
M_solar_bol = 4.74
L_solar = 3.828e33

df_flux['L_bol'] = df_flux['mag_bol'].apply(lambda mag_bol: L_solar * 10**((M_solar_bol - mag_bol) / 2.5))

# Paso 6: Calcular la energía total radiada usando la regla trapezoidal
def calculate_total_radiated_energy(df):
    total_energies = []
    grouped = df.groupby('snid')
    
    for snid, group in grouped:
        group = group.sort_values('mjd')
        total_energy = 0
        for i in range(len(group) - 1):
            L_i = group.iloc[i]['L_bol']
            L_i1 = group.iloc[i + 1]['L_bol']
            delta_t = (group.iloc[i + 1]['mjd'] - group.iloc[i]['mjd']) * 86400
            total_energy += 0.5 * (L_i + L_i1) * delta_t
        total_energies.append({'snid': snid, 'total_radiated_energy': total_energy})
    
    return pd.DataFrame(total_energies)

df_total_energy = calculate_total_radiated_energy(df_flux)


st.write("=)")
st.write(df_total_energy)

# Step 1: Merge 'parsnip_pred' into df_total_energy
df_total_energy = df_total_energy.merge(
    df_flux[['snid', 'parsnip_pred']].drop_duplicates(),
    on='snid',
    how='left'
)

# Step 2: Define the function to calculate the energy in neutrinos
def calculate_neutrino_energy(df):
    neutrino_energies = []
    
    for index, row in df.iterrows():
        energy_total = row['total_radiated_energy']  # Energía total radiada
        sn_type = row.get('parsnip_pred', 'Unknown')  # Columna que define el tipo de supernova
        
        # Determinar la energía en neutrinos según el tipo de supernova
        if sn_type == 'SN Ia':
            # Entre 1% y 2% de la energía total se libera en neutrinos
            E_nu = 0.01 * energy_total  # Ajuste: 0.01 a 0.02
        elif sn_type == 'SN II':
            # Aproximadamente 99% de la energía total se libera en neutrinos
            E_nu = 0.99 * energy_total
        elif sn_type == 'SN Ibc':
            # Entre 90% y 99% de la energía total se libera en neutrinos
            E_nu = 0.95 * energy_total  # Ajuste: 0.90 a 0.99
        else:
            # Si no se conoce el tipo, se asume que la energía en neutrinos es 0
            E_nu = 0
        
        neutrino_energies.append(E_nu)
    
    df['neutrino_energy'] = neutrino_energies  # Añadir la columna con la energía de neutrinos
    return df

# Step 3: Apply the function to calculate neutrino energy
df_total_energy = calculate_neutrino_energy(df_total_energy)

# Verify the result
st.write(df_total_energy[['snid', 'total_radiated_energy', 'parsnip_pred', 'neutrino_energy']].head())



st.write("dff")
# Energy of a single neutrino in erg (10 MeV = 1.6e-5 erg)
E_neutrino_individual = 1.6e-5  # erg/neutrino

# Step 1: Define a function to calculate the number of neutrinos produced
def calculate_neutrino_count(df):
    neutrino_counts = []
    
    for index, row in df.iterrows():
        neutrino_energy = row['neutrino_energy']  # Energy in neutrinos for this supernova
        
        # Calculate the number of neutrinos (total neutrino energy / energy per neutrino)
        N_neutrinos = neutrino_energy / E_neutrino_individual
        neutrino_counts.append(N_neutrinos)
    
    df['neutrino_count'] = neutrino_counts  # Add the column to the DataFrame
    return df

# Step 2: Apply the function to calculate the number of neutrinos
df_total_energy = calculate_neutrino_count(df_total_energy)

# Step 3: Show the updated DataFrame with the calculated neutrino count
st.write(df_total_energy[['snid', 'total_radiated_energy', 'neutrino_energy', 'neutrino_count']].head())


# Radio de la Tierra en cm
R_Tierra = 6.371e8  # en cm

# Área efectiva de la Tierra (sección transversal)
A_Tierra = np.pi * R_Tierra**2  # en cm^2

# Merge para obtener 'D_L_mpc' en `df_total_energy`
df_total_energy = df_total_energy.merge(
    df_flux[['snid', 'D_L_mpc']].drop_duplicates(),
    on='snid',
    how='left'
)

# Verificar si D_L_mpc existe antes de calcular
if 'D_L_mpc' not in df_total_energy.columns:
    st.write("Error: No se pudo calcular 'D_L_mpc'. Revisa el cálculo de la distancia de luminosidad.")

# Step 1: Update the function to calculate how many neutrinos reach the Earth
def calculate_neutrinos_reaching_earth(row):
    # Number of total neutrinos emitted
    N_nu = row['neutrino_count']
    
    # Distance of luminosity in cm
    D_L_cm = row['D_L_mpc'] * 3.086e24  # 1 Mpc = 3.086e24 cm
    
    # Area of the sphere at distance D_L
    A_esfera = 4 * np.pi * D_L_cm**2  # in cm^2
    
    # Calculate how many neutrinos reach Earth
    N_nu_earth = N_nu * (A_Tierra / A_esfera)
    
    # Format the result in scientific notation
    return f"{N_nu_earth:.2e}"

# Step 2: Apply the function to calculate and format neutrinos reaching Earth
df_total_energy['neutrino_reach_earth'] = df_total_energy.apply(calculate_neutrinos_reaching_earth, axis=1)

# Step 3: Display the updated DataFrame with neutrinos reaching Earth in scientific notation
#st.write(df_total_energy[['snid', 'total_radiated_energy', 'neutrino_energy', 'neutrino_count', 'neutrino_reach_earth']].head())

# Mostrar el DataFrame final
st.write(df_total_energy)

# Guardar el DataFrame actualizado en un archivo CSV
df_total_energy.to_csv('neutrinos_reaching_earth.csv', index=False)
st.write("Data saved in 'neutrinos_reaching_earth.csv'.")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Step 1: Locate the MJD of the peak (minimum magnitude) for each supernova in df_flux
if 'snid' in df_flux.columns and 'mag' in df_flux.columns and 'mjd' in df_flux.columns:
    peak_mjd_df = df_flux.loc[df_flux.groupby('snid')['mag'].idxmin(), ['snid', 'mjd']]
    peak_mjd_df.rename(columns={'mjd': 'mjd_peak'}, inplace=True)  # Rename the column for clarity
else:
    st.write("Error: Missing required columns in df_flux ('snid', 'mag', or 'mjd')")

# Step 2: Merge the peak MJD into df_total_energy
if 'snid' in df_total_energy.columns:
    df_total_energy = df_total_energy.merge(peak_mjd_df, on='snid', how='left')
else:
    st.write("Error: Missing 'snid' column in df_total_energy")

# Step 3: Check if the 'mjd_peak' column was successfully added to df_total_energy
if 'mjd_peak' in df_total_energy.columns:
    # First graph: Count of supernovas by MJD (Peak)
    mjd_peak_counts = df_total_energy['mjd_peak'].value_counts().sort_index()

    # Create line plot for the count of supernovas by MJD (Peak)
    fig_lines_peak = go.Figure()

    # Add trace to the plot
    fig_lines_peak.add_trace(go.Scatter(
        x=mjd_peak_counts.index,
        y=mjd_peak_counts.values,
        mode='lines',
        name='Count of Supernovas (Peak)',
        line=dict(color='blue')
    ))

    # Update layout of the figure
    fig_lines_peak.update_layout(
        title='Count of Supernovas by MJD (Peak)',
        xaxis_title='MJD (Peak)',
        yaxis_title='Count of Supernovas',
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig_lines_peak, use_container_width=True)

    # Second graph: Neutrinos reaching Earth by peak MJD
    neutrino_counts_by_mjd_peak = df_total_energy.groupby('mjd_peak')['neutrino_reach_earth'].sum().sort_index()

    # Create line plot for neutrinos reaching Earth by MJD (Peak)
    fig_lines_neutrino_peak = go.Figure()

    # Add trace to the plot
    fig_lines_neutrino_peak.add_trace(go.Scatter(
        x=neutrino_counts_by_mjd_peak.index,
        y=neutrino_counts_by_mjd_peak.values,
        mode='lines',
        name='Neutrinos Reaching Earth (Peak)',
        line=dict(color='green')
    ))

    # Update layout of the figure
    fig_lines_neutrino_peak.update_layout(
        title='Neutrinos Reaching Earth by MJD (Peak)',
        xaxis_title='MJD (Peak)',
        yaxis_title='Neutrinos Reaching Earth',
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig_lines_neutrino_peak, use_container_width=True, key="2")
else:
    st.write("Error: 'mjd_peak' column not found in df_total_energy")



