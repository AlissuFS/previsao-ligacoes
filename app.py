import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

st.set_page_config(page_title="SERCOM Digitais - Projeção", layout="wide", initial_sidebar_state="expanded")

# === Estilo visual personalizado ===
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #4b0081;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div,
    [data-testid="stSidebar"] .stDateInput > div > div,
    [data-testid="stSidebar"] .stTextInput > div > input,
    [data-testid="stSidebar"] .stNumberInput > div {
        background: #4b0081 !important;
        color: white !important;
        border: 1px solid white !important;
        border-radius: 10px !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        padding: 6px;
    }
    [data-testid="stSidebar"] .stFileUploader > div:first-child {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stFileUploadDropzone"] > div {
        display: none !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background-color: #9032bb !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 20px !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        font-weight: 600;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background-color: #a84be0 !important;
    }
    .stButton button {
        background-color: #9032bb;
        color: white;
        border: none;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .stButton button:hover {
        background-color: #a84be0;
    }
    </style>
""", unsafe_allow_html=True)

# === Logo e menu lateral ===
st.sidebar.image(
    "https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif",
    use_container_width=True
)
st.sidebar.markdown("### 🔍 Configurações")

# Uploads bases
uploaded_file_diario = st.sidebar.file_uploader("📂 Base Diária - 'Data', 'Quantidade de Ligações', 'TMA'", type=[".xlsx", ".xls", ".csv"])
uploaded_file_intrahora = st.sidebar.file_uploader("📂 Base Intrahora - 'Intervalo', 'Dia', 'Volume', 'TMA'", type=[".xlsx", ".xls", ".csv"])

dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']

# Checkbox para ativar/desativar funcionamento dos dias da semana
dias_funcionamento = {}
for dia in dias_semana_port:
    dias_funcionamento[dia] = st.sidebar.checkbox(f"{dia}", value=True)

# Funções e preparações (mantidas iguais) ...

# --- AQUI: Captura datas e converte para datetime64 pd.Timestamp ---
mes_referencia = st.sidebar.date_input(
    "📅 Mês referência (base diária)",
    value=datetime.today().replace(day=1)
)
mes_referencia = pd.to_datetime(mes_referencia)

mes_proj1 = st.sidebar.date_input(
    "📅 Mês proj. 1",
    value=(datetime.today() + pd.DateOffset(months=1)).replace(day=1)
)
mes_proj1 = pd.to_datetime(mes_proj1)

mes_proj2 = st.sidebar.date_input(
    "📅 Mês proj. 2",
    value=(datetime.today() + pd.DateOffset(months=2)).replace(day=1)
)
mes_proj2 = pd.to_datetime(mes_proj2)

mes_proj3 = st.sidebar.date_input(
    "📅 Mês proj. 3",
    value=(datetime.today() + pd.DateOffset(months=3)).replace(day=1)
)
mes_proj3 = pd.to_datetime(mes_proj3)

# Depois que fizer a leitura dos arquivos e preparar os DataFrames
if uploaded_file_diario and uploaded_file_intrahora:
    # Leitura e preparação base diária
    df_diario = pd.read_excel(uploaded_file_diario) if uploaded_file_diario.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file_diario)
    df_diario.columns = df_diario.columns.str.strip()
    required_cols_diario = {'Data', 'Quantidade de Ligações', 'TMA'}
    if not required_cols_diario.issubset(df_diario.columns):
        st.error(f"Base diária deve conter as colunas: {required_cols_diario}")
        st.stop()
    df_diario = preparar_df_diario(df_diario)
    
    # *** Filtragem do df_diario pelo mes_referencia convertida ***
    df_diario = df_diario[df_diario['ds'] >= mes_referencia]
    
    # Continue com os cálculos e projeções normalmente...

    # (O restante do código continua igual, incluindo preparo do intrahora, modelos Prophet, projeções e gráficos)

else:
    st.info("Por favor, envie as duas bases para gerar as projeções.")
