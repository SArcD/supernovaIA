import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# Función para encontrar el MJD del pico (mínima magnitud) y calcular días relativos al pico
def calcular_dias_relativos(df_supernova):
    # Identificar el MJD del pico de luminosidad (mínima magnitud)
    if df_supernova['mag'].isnull().all():
        return df_supernova['mjd']  # Si todos los valores son nulos, devolver el MJD original

    mjd_pico = df_supernova.loc[df_supernova['mag'].idxmin(), 'mjd']  # MJD del pico de luminosidad
    df_supernova['dias_relativos'] = df_supernova['mjd'] - mjd_pico  # Calcular días antes y después del pico
    
    return df_supernova

# Función para graficar la curva de luz de una supernova con el eje X en días relativos al pico
def graficar_curva_de_luz(df_supernova):
    # Calcular días relativos al pico de luminosidad
    df_supernova = calcular_dias_relativos(df_supernova)
    
    fig = go.Figure()

    # Graficar para cada filtro
    for filtro in df_supernova['filtro'].unique():
        df_filtro = df_supernova[df_supernova['filtro'] == filtro]
        fig.add_trace(go.Scatter(
            x=df_filtro['dias_relativos'],  # Usar días relativos al pico como eje X
            y=df_filtro['mag'],
            mode='lines+markers',
            name=filtro
        ))

    # Invertir el eje Y porque las magnitudes menores son más brillantes
    fig.update_layout(
        title=f'Curva de luz de {df_supernova["snid"].iloc[0]} (días relativos al pico de luminosidad)',
        xaxis_title='Días relativos al pico de luminosidad',
        yaxis_title='Magnitud',
        yaxis=dict(autorange='reversed'),  # Invertir el eje Y
        showlegend=True
    )

    return fig

# Seleccionar supernova
snid_seleccionado = st.selectbox("Selecciona una supernova para ver su curva de luz:", df_curvas_luz['snid'].unique())

# Filtrar los datos de la supernova seleccionada
df_supernova_seleccionada = df_curvas_luz[df_curvas_luz['snid'] == snid_seleccionado]

# Mostrar la gráfica de curva de luz
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
