import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime
from prophet import Prophet
import altair as alt

st.set_page_config(page_title="SERCOM Digitais - Projeção", layout="wide", initial_sidebar_state="expanded")

# CSS com ajustes finais aplicados
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
    [data-testid="stSidebar"] .stDateInput > div > div {
        background: #4b0081 !important;
        color: white !important;
        border: 1px solid white !important;
        border-radius: 10px !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        padding: 6px;
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

# Sidebar
st.sidebar.image(
    "https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif",
    use_container_width=True
)
st.sidebar.markdown("### 🔍 Configurações")

uploaded_file = st.sidebar.file_uploader("📂 Selecionar arquivo com colunas 'Data', 'Quantidade de Ligações' e 'TMA'", type=[".xlsx", ".xls", ".csv"])
dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.sidebar.multiselect("📍 Dias da semana considerados", dias_semana_port, default=dias_semana_port)

def ocorrencia_semana(data):
    dia_semana = data.weekday()
    dias_mes = pd.date_range(start=data.replace(day=1), end=data)
    return sum(d.weekday() == dia_semana for d in dias_mes)

def remover_outliers_detalhado(df, valor_col):
    df = df.copy()
    df['ordem'] = df['ds'].apply(ocorrencia_semana)
    grupos = df.groupby(['dia_semana_pt', 'ordem'])
    df_filtrado = []

    for (dia, ordem), grupo in grupos:
        if dia == 'Domingo':
            media_valor = grupo[valor_col].mean()
            data_referencia = grupo['ds'].iloc[0]
            df_filtrado.append(pd.DataFrame({
                'ds': [data_referencia],
                valor_col: [media_valor],
                'dia_semana_pt': [dia],
                'ordem': [ordem]
            }))
            continue

        if len(grupo) < 3:
            df_filtrado.append(grupo)
            continue

        Q1 = grupo[valor_col].quantile(0.25)
        Q3 = grupo[valor_col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        grupo_filtrado = grupo[(grupo[valor_col] >= lim_inf) & (grupo[valor_col] <= lim_sup)]
        df_filtrado.append(grupo_filtrado)

    return pd.concat(df_filtrado).sort_values('ds')

def format_num_brl(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'Data' not in df.columns or 'Quantidade de Ligações' not in df.columns or 'TMA' not in df.columns:
        st.error("A planilha deve conter as colunas: 'Data', 'Quantidade de Ligações' e 'TMA'")
        st.stop()

    df['ds'] = pd.to_datetime(df['Data'])
    df['y'] = pd.to_numeric(df['Quantidade de Ligações'], errors='coerce').fillna(0).clip(lower=0)
    df['tma'] = pd.to_numeric(df['TMA'], errors='coerce').fillna(0).clip(lower=0)
    df['ano_mes'] = df['ds'].dt.to_period('M')
    df['dia_semana'] = df['ds'].dt.day_name()
    mapa_dias = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['dia_semana_pt'] = df['dia_semana'].map(mapa_dias)
    df['ordem'] = df['ds'].apply(ocorrencia_semana)

    df_limpo_volume = remover_outliers_detalhado(df, 'y')
    df_limpo_volume = df_limpo_volume[['ds', 'y']].drop_duplicates(subset=['ds']).reset_index(drop=True)

    df_limpo_tma = remover_outliers_detalhado(df, 'tma')
    df_tma = df_limpo_tma.groupby('ds')['tma'].mean().reset_index().rename(columns={'tma': 'y'})

    modelo_volume = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_volume.fit(df_limpo_volume)
    modelo_tma = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma.fit(df_tma)

    mes_proj = st.sidebar.date_input("🗓 Escolha o primeiro dia do mês para projeção futura", value=datetime.today().replace(day=1))
    mes_proj = pd.to_datetime(mes_proj)
    dias_futuros = pd.date_range(start=mes_proj, end=(mes_proj + pd.offsets.MonthEnd(0)))
    df_futuro = pd.DataFrame({'ds': dias_futuros})

    previsao_volume = modelo_volume.predict(df_futuro)[['ds', 'yhat']].rename(columns={'yhat': 'y'})
    previsao_volume['y'] = previsao_volume['y'].apply(lambda x: 1 if x <= 0 else x)

    previsao_tma = modelo_tma.predict(df_futuro)[['ds', 'yhat']].rename(columns={'yhat': 'tma'})

    df_prev = pd.merge(previsao_volume, previsao_tma, on='ds')
    df_prev['percentual_volume'] = df_prev['y'] / df_prev['y'].sum() * 100
    media_tma = df_prev['tma'].mean()
    df_prev['percentual_tma'] = df_prev['tma'] / media_tma * 100

    df_prev['percentual_volume_str'] = df_prev['percentual_volume'].apply(lambda x: format_num_brl(x) + '%')
    df_prev['percentual_tma_str'] = df_prev['percentual_tma'].apply(lambda x: format_num_brl(x) + '%')

    df_prev_formatado = df_prev[['ds', 'y', 'percentual_volume', 'tma', 'percentual_tma']].copy()
    df_prev_formatado.columns = ['Data', 'Volume projetado', '% curva volume', 'TMA projetado (s)', '% curva TMA']
    df_prev_formatado['Data'] = pd.to_datetime(df_prev_formatado['Data']).dt.strftime('%d/%m/%Y')
    df_prev_formatado['% curva volume'] = df_prev_formatado['% curva volume'].apply(lambda x: format_num_brl(x) + '%')
    df_prev_formatado['% curva TMA'] = df_prev_formatado['% curva TMA'].apply(lambda x: format_num_brl(x) + '%')

    st.success("Previsões geradas com sucesso!")
    st.dataframe(df_prev_formatado, use_container_width=True)

    st.markdown("### 📊 Gráficos de Comparação")

    df_chart = df_prev.copy()
    df_chart['percentual_volume'] = df_chart['percentual_volume'].round(2)
    df_chart['percentual_tma'] = df_chart['percentual_tma'].round(2)
    df_chart['percentual_volume_str'] = df_chart['percentual_volume'].apply(lambda x: format_num_brl(x) + '%')
    df_chart['percentual_tma_str'] = df_chart['percentual_tma'].apply(lambda x: format_num_brl(x) + '%')

    linha_volume = alt.Chart(df_chart).mark_line(color='#4b0081').encode(
        x=alt.X('ds:T', title='Data'),
        y=alt.Y('percentual_volume:Q', title='% Volume')
    )
    pontos_volume = alt.Chart(df_chart).mark_point(color='#9032bb', filled=True).encode(
        x='ds:T',
        y='percentual_volume:Q',
        tooltip=[
            alt.Tooltip('ds:T', title='Data'),
            alt.Tooltip('y:Q', title='Volume', format='.0f'),
            alt.Tooltip('percentual_volume_str:N', title='% Volume')
        ]
    )
    chart_volume = (linha_volume + pontos_volume).properties(
        title='Curva de Volume Projetado', width=800, height=300
    )

    linha_tma = alt.Chart(df_chart).mark_line(color='#4b0081').encode(
        x=alt.X('ds:T', title='Data'),
        y=alt.Y('percentual_tma:Q', title='% TMA')
    )
    pontos_tma = alt.Chart(df_chart).mark_point(color='#9032bb', filled=True).encode(
        x='ds:T',
        y='percentual_tma:Q',
        tooltip=[
            alt.Tooltip('ds:T', title='Data'),
            alt.Tooltip('tma:Q', title='TMA (s)', format='.0f'),
            alt.Tooltip('percentual_tma_str:N', title='% TMA')
        ]
    )
    chart_tma = (linha_tma + pontos_tma).properties(
        title='Curva de TMA Projetado', width=800, height=300
    )

    st.altair_chart(chart_volume, use_container_width=True)
    st.altair_chart(chart_tma, use_container_width=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_prev_formatado.to_excel(writer, index=False, sheet_name='Projecao')

    st.download_button(
        label="📥 Baixar Excel com projeções de Volume e TMA",
        data=buffer.getvalue(),
        file_name="projecao_volume_tma.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
