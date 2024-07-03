import requests
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    bucket_name = "modelosllm"
    object_name = "car_model_electric.pkl"
    url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Esto lanzará una excepción para códigos de estado no exitosos
        
        # Crear un objeto de tipo archivo en memoria
        file_object = pickle.loads(response.content)
        
        st.success("Modelo cargado exitosamente.")
        return file_object
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo: {e}")
        return None
    except pickle.UnpicklingError as e:
        st.error(f"Error al deserializar el modelo: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado al cargar el modelo: {e}")
        return None

# Cargar el modelo
model, tokenizer = load_model()


def generate_response(prompt, model, tokenizer, max_length=100):
    # Tokenizar la entrada
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generar respuesta
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Para respuestas variadas
        top_k=15,        # Para limitar el espacio de búsqueda a las 50 mejores opciones
        top_p=0.95       # Para el método de muestreo nucleus
    )
    
    # Decodificar la salida
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Interfaz de Streamlit
st.title("Chatbot de autos eléctricos")

prompt = st.text_input("Haz una pregunta:")

if st.button("Enviar"):
    if prompt:
        response = generate_response(prompt, model, tokenizer)
        st.write(response)
    else:
        st.write("Por favor ingresa una pregunta.")