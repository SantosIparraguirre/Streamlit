import streamlit as st
import pickle

@st.cache_resource
def load_model():
    with open('./car_qa_model.pkl', 'rb') as f:
        model, tokenizer = pickle.load(f)
    return model, tokenizer

def generate_answer(question, model, tokenizer, max_length=513):
    input_text = f"Pregunta: {question}\nRespuesta:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512)
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.5,
    )
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split("Respuesta:")[-1].strip()

st.title("Asistente de Autos Eléctricos")

model, tokenizer = load_model()

st.write("Este asistente puede responder preguntas sobre autos eléctricos. ¡Prueba haciendo una pregunta!")

question = st.text_input("Tu pregunta:")

if st.button("Obtener respuesta"):
    if question:
        with st.spinner('Generando respuesta...'):
            answer = generate_answer(question, model, tokenizer)
        st.write("Respuesta:", answer)
    else:
        st.write("Por favor, ingresa una pregunta.")

st.write("Ejemplos de preguntas que puedes hacer:")
st.write("- ¿Cuál es la autonomía del Tesla Model 3?")
st.write("- ¿Cuánto tarda en cargarse el Hyundai IONIQ?")
st.write("- ¿Cuál es el precio del Ford Mustang Mach-E?")
st.write("- ¿Por qué los autos eléctricos son más eficientes?")