import pandas as pd
import streamlit as st

autos_electricos = pd.read_csv("df_electric_cars.csv")
renombrar_columnas = {"Brand":"Marca", "ROI":"ROI (%)","price_dollars":"Costo Unitario", "Combined (wh/km)":"Eficiencia (Wh/Km)","range_km":"Distancia antes de cargar (Km)", "battery_life_km":"Vida útil batería (Km)", "Model":"Modelo", "model_year":"Año salida", "vehicle_class":"Clase vehiculo", "charge_time_minutes":"Tiempo de carga (min)", "engine_size_L":"Tamaño motor (Litros)", "engine_cylinder":"Cilindros motor", "transmission_type":"Tipo de trasmisión"}
autos_electricos=autos_electricos.rename(columns=renombrar_columnas)

marcas_ordenadas = sorted(autos_electricos['Marca'].unique())

st.set_page_config(page_title="Recomendación de Taxis Eléctricos para la ciudad de New York", page_icon="-", layout="wide")

with st.container():
    st.title("Recomendación de Taxis Eléctricos")
    st.write("Este es un prototipo de recomendación de taxis eléctricos en la Ciudad de Nueva York.")

    # Cuadro de texto para seleccionar la cantidad de taxis (opcional)
    cantidad_taxis = st.number_input('Cantidad de Taxis a Consultar', min_value=1, max_value=2000, value=1)

    # Seleccionar una marca
    seleccionada = st.selectbox('Selecciona una Marca', options=marcas_ordenadas)

    # Botón para validar y realizar la consulta
    if st.button('Consultar'):
        if seleccionada and  cantidad_taxis >1:
            
            
            # Filtrar  dataframe según la marca seleccionada
            df_filtrado = autos_electricos[autos_electricos['Marca'] == seleccionada]
            df_filtrado["Inversión"] = df_filtrado["Costo Unitario"] * cantidad_taxis

            #Calculo de ROI
            promedio_viajes_futuros_por_taxi = 7
            promedio_ingreso_por_viaje = 24 # Dolares
            dias_laborables = 365
            ingresos_totales_anuales = promedio_viajes_futuros_por_taxi * cantidad_taxis * promedio_ingreso_por_viaje * dias_laborables
            distancia_promedio_por_viaje_en_millas = 4.61
            roi = round((ingresos_totales_anuales - df_filtrado["Costo Unitario"]) / df_filtrado["Costo Unitario"] * 100, 2)
            df_filtrado["ROI (%)"] = roi

            # Formatos de Columnas
            df_filtrado['Inversión'] = df_filtrado['Inversión'].apply(lambda x: f"${x:,.0f}")
            df_filtrado['Costo Unitario'] = df_filtrado['Costo Unitario'].apply(lambda x: f"${x:,.0f}")
            df_filtrado['ROI (%)'] = df_filtrado['ROI (%)'].apply(lambda x: f"{x:.2%}")
            df_mostrar = df_filtrado.drop(columns=['Marca', 'vehicle_type'])

            #
            columnas = df_mostrar.columns.tolist()
            columnas_reordenadas = [columnas[0]] + columnas[-3:] + columnas[1:-3]
            df_reordenado = df_mostrar[columnas_reordenadas]
            
            # Mostrar la tabla con los modelos y columnas relacionadas
            st.write(f"Modelos de {seleccionada}:")
            st.table(df_reordenado)
        else:
            st.write('Por favor, selecciona una marca y/o cantidad.')
