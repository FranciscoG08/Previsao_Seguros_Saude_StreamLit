import streamlit as st
import pandas as pd
import joblib
#Embelezado  por gemini
# Configuração da página (deve ser a primeira coisa no script)
st.set_page_config(page_title="HealthPredict", page_icon="🏥", layout="centered")

# Carregar modelo
@st.cache_resource # Isto evita carregar o modelo do disco em cada clique
def load_model():
    return joblib.load("models/seguros_saude.pkl")

modelo = load_model()

# Estilo CSS personalizado para melhorar o aspeto
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Cabeçalho com ícone
st.title("🏥 Previsão de Custos de Saúde")
st.markdown("---")

# Organizar inputs em colunas
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Perfil Pessoal")
    genero = st.selectbox("Género", ["masculino", "feminino"])
    estado_civil = st.selectbox("Estado Civil", ["solteiro", "casado", "divorciado", "viuvo"])
    class_etaria = st.selectbox("Classe Etária", ["18-30", "31-45", "46-60", "60+"])

with col2:
    st.markdown("### 📋 Dados Médicos")
    imc = st.slider("Índice de Massa Corporal (IMC)", 10.0, 60.0, 25.0)
    fumador = st.radio("Fumador?", ["sim", "nao"], horizontal=True)
    zona_residencia = st.selectbox("Zona de Residência", ["urbana", "suburbana", "rural"])

st.markdown("---")

# Mapeamento e tratamento de dados
# (Assegura que os nomes das colunas são EXATAMENTE iguais aos do treino)
dados_input = pd.DataFrame({
    "genero": [1 if genero == "masculino" else 0],
    "estado_civil": [estado_civil],
    "zona_residencia": [zona_residencia],
    "imc": [imc],
    "fumador": [1 if fumador == "sim" else 0],
    "class_etaria": [class_etaria]
})

# Botão de previsão centralizado
if st.button("🚀 Calcular Estimativa de Seguro"):
    with st.spinner('O algoritmo está a analisar os dados...'):
        # Fazer a previsão
        previsao = modelo.predict(dados_input)[0]
        
        # Exibição bonita do resultado
        st.markdown(f"""
            <div class="result-box">
                <h2 style='color: #333;'>Resultado da Análise</h2>
                <p style='font-size: 1.2em;'>Com base no perfil indicado, o custo anual estimado é:</p>
                <h1 style='color: #28a745;'>{previsao:,.2f} €</h1>
                <p style='color: #666; font-size: 0.8em;'>*Este valor é uma estimativa baseada em modelos estatísticos.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Adicionar um aviso visual se for fumador (explicabilidade)
        if fumador == "sim":
            st.warning("⚠️ Nota: O hábito de fumar é o fator que mais eleva o custo da sua apólice.")