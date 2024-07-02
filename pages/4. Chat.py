import streamlit as st
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import requests
import io



# # Cargar el modelo y el tokenizer
# @st.cache_resource
# def load_model():
#     with open('car_model.pkl', 'rb') as f:
#         model, tokenizer = pickle.load(f)
#     return model, tokenizer

# model, tokenizer = load_model()

@st.cache_resource
def load_model():
    bucket_name = "modelosllm"
    object_name = "car_model.pkl"
    url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
    
    st.write(f"Intentando cargar el modelo desde: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Esto lanzará una excepción para códigos de estado no exitosos
        
        # Crear un objeto de tipo archivo en memoria
        file_object = io.BytesIO(response.content)
        
        # Cargar el modelo y el tokenizer directamente desde el objeto en memoria
        model, tokenizer = pickle.load(file_object)
        
        st.success("Modelo cargado exitosamente.")
        return model, tokenizer
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo: {e}")
        return None, None
    except pickle.UnpicklingError as e:
        st.error(f"Error al deserializar el modelo: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error inesperado al cargar el modelo: {e}")
        return None, None

# Cargar el modelo
model, tokenizer = load_model()

# Verificar si el modelo y el tokenizer se cargaron correctamente
if model is None or tokenizer is None:
    st.error("No se pudo cargar el modelo o el tokenizer.")
else:
    st.success("Modelo y tokenizer cargados correctamente.")


# Definir las funciones necesarias
def generate_response(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_info(text):
    info = {}
    patterns = {
        'Brand': r'Brand:\s*(\w+)',
        'Model': r'Model:\s*([\w\s]+)',
        'Year': r'Year:\s*(\d+)',
        'Class': r'Class:\s*(\w+)',
        'Type': r'Type:\s*(\w+)',
        'Charge Time': r'Charge Time:\s*(\d+)',
        'Battery Life': r'Battery Life:\s*(\d+)',
        'Range': r'Range:\s*(\d+)',
        'Engine Size': r'Engine Size:\s*([\d.]+)',
        'Cylinders': r'Cylinders:\s*(\d+)',
        'Transmission': r'Transmission:\s*(\w+)',
        'Efficiency': r'Efficiency:\s*([\d.]+)',
        'Price': r'Price:\s*\$(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            info[key] = match.group(1)
    
    return info

def elaborate_response(info):
    responses = []
    
    if 'Brand' in info and 'Model' in info:
        responses.append(f"The {info['Brand']} {info['Model']} is a cutting-edge electric vehicle that exemplifies modern automotive engineering.")
    
    if 'Class' in info:
        responses.append(f"As a {info['Class'].lower()}, it offers a perfect balance of comfort and performance for discerning drivers.")
    
    if 'Range' in info:
        responses.append(f"With an impressive range of {info['Range']} km on a single charge, this vehicle is designed for long-distance travel without compromise.")
    
    if 'Charge Time' in info:
        responses.append(f"The advanced charging system allows for a full charge in just {info['Charge Time']} minutes, minimizing downtime and maximizing your mobility.")
    
    if 'Battery Life' in info:
        responses.append(f"The high-capacity battery is engineered to last up to {info['Battery Life']} km, ensuring long-term reliability and reducing environmental impact.")
    
    if 'Efficiency' in info:
        responses.append(f"With an energy efficiency of {info['Efficiency']} Wh/km, this vehicle sets a new standard in electric powertrain optimization.")
    
    if 'Price' in info:
        responses.append(f"Priced at ${info['Price']}, it represents a compelling value proposition in the electric vehicle market.")
    
    return ' '.join(responses)

def generate_detailed_response(prompt, model, tokenizer):
    initial_response = generate_response(prompt, model, tokenizer)
    info = extract_info(initial_response)
    detailed_response = elaborate_response(info)
    return detailed_response

# Interfaz de Streamlit
st.title("Car Information Generator")

prompt = st.text_input("Enter a car brand and model:")

if st.button("Generate Information"):
    if prompt:
        response = generate_detailed_response(prompt, model, tokenizer)
        st.write(response)
    else:
        st.write("Please enter a car brand and model.")