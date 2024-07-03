import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Función para cargar los datos
@st.cache_data
def load_data():
    taxis_mayo_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-05.parquet")
    taxis_junio_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-06.parquet")
    taxis_julio_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-07.parquet")
    taxis_agosto_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-08.parquet")
    taxis_septiembre_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-09.parquet")
    taxis_octubre_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-10.parquet")
    taxis_noviembre_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-11.parquet")
    taxis_diciembre_2023 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-12.parquet")
    taxis_enero_2024 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet")
    taxis_febrero_2024 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet")
    taxis_marzo_2024 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet")
    taxis_abril_2024 = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-04.parquet")

    return [taxis_mayo_2023, taxis_junio_2023, taxis_julio_2023, taxis_agosto_2023, 
            taxis_septiembre_2023, taxis_octubre_2023, taxis_noviembre_2023, 
            taxis_diciembre_2023, taxis_enero_2024, taxis_febrero_2024, 
            taxis_marzo_2024, taxis_abril_2024]

# Cargar los datos
datasets = load_data()

# Concatenar los datos
registros_taxis = pd.concat(datasets)

# Limpiar los datos
registros_taxis = registros_taxis.drop(columns=['VendorID', 'tpep_dropoff_datetime',
       'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
       'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'congestion_surcharge', 'Airport_fee'])

registros_taxis = registros_taxis[(registros_taxis["tpep_pickup_datetime"] >= "2023-05-01") & (registros_taxis["tpep_pickup_datetime"] < "2024-05-01")]

# Convertir la columna a datetime
registros_taxis['tpep_pickup_datetime'] = pd.to_datetime(registros_taxis['tpep_pickup_datetime'])

# Agrupar los datos por día
conteo_dias = registros_taxis.groupby(registros_taxis['tpep_pickup_datetime'].dt.date).size()

data = {
    'fecha': conteo_dias.index,
    'valor': conteo_dias.values
}
df = pd.DataFrame(data)

# Establecer la fecha como índice
df.set_index('fecha', inplace=True)

# Dividir los datos en entrenamiento y prueba
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Ajustar el modelo ARIMA
model = ARIMA(train, order=(7, 0, 15))  # Ajustar los parámetros según los datos
model_fit = model.fit()

# Predecir los próximos 180 días
future_predictions = model_fit.forecast(steps=180)

# Obtener los meses inicial y final
start_month = df.index[-1].strftime('%B %Y')
end_month = pd.date_range(start=df.index[-1], periods=180, freq='D')[-1].strftime('%B %Y')

# Visualizar las predicciones futuras
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df, label='Histórico')
ax.plot(pd.date_range(start=df.index[-1], periods=180, freq='D'), future_predictions, label='Predicciones Futuras', color='green')
ax.set_xlabel('Fecha')
ax.set_ylabel('Cantidad de Viajes')
ax.set_title(f'Predicciones Futuras de Cantidad de Viajes desde {start_month} hasta {end_month}')
ax.legend()

# Mostrar la gráfica en Streamlit
st.pyplot(fig)
