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

# **Aqui:** Checkbox para ativar/desativar funcionamento dos dias da semana
dias_funcionamento = {}
for dia in dias_semana_port:
    dias_funcionamento[dia] = st.sidebar.checkbox(f"{dia}", value=True)

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

def preparar_df_diario(df):
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
    return df

def preparar_df_intrahora(df):
    df['ds'] = pd.to_datetime(df['Intervalo'])
    df['dia_semana'] = pd.to_datetime(df['Dia']).dt.day_name()
    mapa_dias = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['dia_semana_pt'] = df['dia_semana'].map(mapa_dias)
    df['ordem'] = df['ds'].apply(ocorrencia_semana)
    df['y'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).clip(lower=0)
    df['tma'] = pd.to_numeric(df['TMA'], errors='coerce').fillna(0).clip(lower=0)
    return df

if uploaded_file_diario and uploaded_file_intrahora:
    # Leitura e preparação base diária
    df_diario = pd.read_excel(uploaded_file_diario) if uploaded_file_diario.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file_diario)
    df_diario.columns = df_diario.columns.str.strip()
    required_cols_diario = {'Data', 'Quantidade de Ligações', 'TMA'}
    if not required_cols_diario.issubset(df_diario.columns):
        st.error(f"Base diária deve conter as colunas: {required_cols_diario}")
        st.stop()
    df_diario = preparar_df_diario(df_diario)
    df_limpo_vol_diario = remover_outliers_detalhado(df_diario, 'y')[['ds','y']].drop_duplicates().reset_index(drop=True)
    df_limpo_tma_diario = remover_outliers_detalhado(df_diario, 'tma')
    df_tma_diario = df_limpo_tma_diario.groupby('ds')['tma'].mean().reset_index().rename(columns={'tma':'y'})

    # Leitura e preparação base intrahora
    df_intrahora = pd.read_excel(uploaded_file_intrahora) if uploaded_file_intrahora.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file_intrahora)
    df_intrahora.columns = df_intrahora.columns.str.strip()
    required_cols_intrahora = {'Intervalo', 'Dia', 'Volume', 'TMA'}
    if not required_cols_intrahora.issubset(df_intrahora.columns):
        st.error(f"Base intrahora deve conter as colunas: {required_cols_intrahora}")
        st.stop()
    df_intrahora = preparar_df_intrahora(df_intrahora)
    df_limpo_vol_intrahora = remover_outliers_detalhado(df_intrahora, 'y')[['ds','y']].drop_duplicates().reset_index(drop=True)
    df_limpo_tma_intrahora = remover_outliers_detalhado(df_intrahora, 'tma')
    df_tma_intrahora = df_limpo_tma_intrahora.groupby('ds')['tma'].mean().reset_index().rename(columns={'tma':'y'})

    # Modelos Prophet
    modelo_vol_diario = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_vol_diario.fit(df_limpo_vol_diario)
    modelo_tma_diario = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma_diario.fit(df_tma_diario)

    modelo_vol_intrahora = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_vol_intrahora.fit(df_limpo_vol_intrahora)
    modelo_tma_intrahora = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma_intrahora.fit(df_tma_intrahora)

    # Seleção mês projeção
    mes_proj = st.sidebar.date_input("🗓 Escolha o primeiro dia do mês para projeção futura", value=datetime.today().replace(day=1))
    mes_proj = pd.to_datetime(mes_proj)

    def gerar_datas_futuras_intrahora(mes):
        inicio = mes
        fim = mes + pd.offsets.MonthEnd(0)
        rng = pd.date_range(start=inicio, end=fim + timedelta(days=1), freq='30min')[:-1]
        return rng

    # Datas futuras
    dias_futuros_diario = pd.date_range(start=mes_proj, end=(mes_proj + pd.offsets.MonthEnd(0)))
    df_futuro_diario = pd.DataFrame({'ds': dias_futuros_diario})

    datas_futuras_intrahora = gerar_datas_futuras_intrahora(mes_proj)
    df_futuro_intrahora = pd.DataFrame({'ds': datas_futuras_intrahora})

    # Previsões
    previsao_vol_diario = modelo_vol_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'y'})
    previsao_vol_diario['y'] = previsao_vol_diario['y'].apply(lambda x: max(1, x))
    previsao_tma_diario = modelo_tma_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'tma'})

    previsao_vol_intrahora = modelo_vol_intrahora.predict(df_futuro_intrahora)[['ds','yhat']].rename(columns={'yhat':'y'})
    previsao_vol_intrahora['y'] = previsao_vol_intrahora['y'].apply(lambda x: max(0.1, x))
    previsao_tma_intrahora = modelo_tma_intrahora.predict(df_futuro_intrahora)[['ds','yhat']].rename(columns={'yhat':'tma'})

    # --- Aqui: Adiciona dias da semana em português nas previsões ---
    mapa_dias = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }

    previsao_vol_intrahora['dia_semana_pt'] = previsao_vol_intrahora['ds'].dt.day_name().map(mapa_dias)
    previsao_tma_intrahora['dia_semana_pt'] = previsao_tma_intrahora['ds'].dt.day_name().map(mapa_dias)
    previsao_vol_diario['dia_semana_pt'] = previsao_vol_diario['ds'].dt.day_name().map(mapa_dias)
    previsao_tma_diario['dia_semana_pt'] = previsao_tma_diario['ds'].dt.day_name().map(mapa_dias)

    # --- Zerar valores dos dias desativados ---
    dias_desativados = [dia for dia, ativo in dias_funcionamento.items() if not ativo]

    previsao_vol_intrahora.loc[previsao_vol_intrahora['dia_semana_pt'].isin(dias_desativados), 'y'] = 0
    previsao_tma_intrahora.loc[previsao_tma_intrahora['dia_semana_pt'].isin(dias_desativados), 'tma'] = 0
    previsao_vol_diario.loc[previsao_vol_diario['dia_semana_pt'].isin(dias_desativados), 'y'] = 0
    previsao_tma_diario.loc[previsao_tma_diario['dia_semana_pt'].isin(dias_desativados), 'tma'] = 0

    # --- Recalcular percentuais após zerar dias desativados ---
    # Intrahora (percentual dentro de cada dia)
    previsao_vol_intrahora['data'] = previsao_vol_intrahora['ds'].dt.date
    previsao_tma_intrahora['data'] = previsao_tma_intrahora['ds'].dt.date

    previsao_vol_intrahora['percentual_volume'] = previsao_vol_intrahora.groupby('data')['y'].transform(
        lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0
    )
    previsao_tma_intrahora['percentual_tma'] = previsao_tma_intrahora.groupby('data')['tma'].transform(
        lambda x: (x / x.mean() * 100) if x.mean() > 0 else 0
    )

    # Diária
    if previsao_vol_diario['y'].sum() > 0:
        previsao_vol_diario['percentual_volume'] = previsao_vol_diario['y'] / previsao_vol_diario['y'].sum() * 100
    else:
        previsao_vol_diario['percentual_volume'] = 0

    media_tma_diario = previsao_tma_diario['tma'].mean() if previsao_tma_diario['tma'].mean() > 0 else 1
    previsao_tma_diario['percentual_tma'] = previsao_tma_diario['tma'] / media_tma_diario * 100

    # Função formatar tooltip percentual
    def fmt_perc(x):
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + '%'

    previsao_vol_intrahora['percentual_volume_str'] = previsao_vol_intrahora['percentual_volume'].apply(fmt_perc)
    previsao_tma_intrahora['percentual_tma_str'] = previsao_tma_intrahora['percentual_tma'].apply(fmt_perc)
    previsao_vol_diario['percentual_volume_str'] = previsao_vol_diario['percentual_volume'].apply(fmt_perc)
    previsao_tma_diario['percentual_tma_str'] = previsao_tma_diario['percentual_tma'].apply(fmt_perc)

    # --- Exibir tabelas ---
    st.markdown("## Curva Diária - Projeção")
    df_formatado_diario = pd.DataFrame({
        'Data': previsao_vol_diario['ds'].dt.strftime('%d/%m/%Y'),
        'Volume projetado': previsao_vol_diario['y'].round(0).astype(int),
        '% curva volume': previsao_vol_diario['percentual_volume_str'],
        'TMA projetado (s)': previsao_tma_diario['tma'].round(0).astype(int),
        '% curva TMA': previsao_tma_diario['percentual_tma_str']
    })
    st.dataframe(df_formatado_diario, use_container_width=True)

    st.markdown("## Curva Intrahora - Projeção (intervalo 30min)")
    df_formatado_intrahora = pd.DataFrame({
        'Data-Hora': previsao_vol_intrahora['ds'].dt.strftime('%d/%m/%Y %H:%M'),
        'Volume projetado': previsao_vol_intrahora['y'].round(0).astype(int),
        '% curva volume': previsao_vol_intrahora['percentual_volume_str'],
        'TMA projetado (s)': previsao_tma_intrahora['tma'].round(0).astype(int),
        '% curva TMA': previsao_tma_intrahora['percentual_tma_str']
    })
    st.dataframe(df_formatado_intrahora, use_container_width=True)

    # --- Gráficos comparativos diária ---
    st.markdown("### 📊 Gráficos Comparativos - Curva Diária")

    chart_vol_diario = (alt.Chart(previsao_vol_diario)
        .mark_line(color='#4b0081')
        .encode(
            x=alt.X('ds:T', title='Data'),
            y=alt.Y('percentual_volume:Q', title='% Volume')
        ))

    points_vol_diario = (alt.Chart(previsao_vol_diario)
        .mark_circle(color='#9032bb', size=50)
        .encode(
            x='ds:T',
            y='percentual_volume:Q',
            tooltip=[
                alt.Tooltip('ds:T', title='Data', format='%d/%m/%Y'),
                alt.Tooltip('y:Q', title='Volume'),
                alt.Tooltip('percentual_volume:Q', title='% Volume', format='.2f')
            ]
        ))

    st.altair_chart(chart_vol_diario + points_vol_diario, use_container_width=True)

    chart_tma_diario = (alt.Chart(previsao_tma_diario)
        .mark_line(color='#4b0081')
        .encode(
            x=alt.X('ds:T', title='Data'),
            y=alt.Y('percentual_tma:Q', title='% TMA')
        ))

    points_tma_diario = (alt.Chart(previsao_tma_diario)
        .mark_circle(color='#9032bb', size=50)
        .encode(
            x='ds:T',
            y='percentual_tma:Q',
            tooltip=[
                alt.Tooltip('ds:T', title='Data', format='%d/%m/%Y'),
                alt.Tooltip('tma:Q', title='TMA (s)'),
                alt.Tooltip('percentual_tma:Q', title='% TMA', format='.2f')
            ]
        ))

    st.altair_chart(chart_tma_diario + points_tma_diario, use_container_width=True)

    # --- Gráficos comparativos intrahora ---
    st.markdown("### 📊 Gráficos Comparativos - Curva Intrahora")

    chart_vol_intrahora = (alt.Chart(previsao_vol_intrahora)
        .mark_line(color='#4b0081')
        .encode(
            x=alt.X('ds:T', title='Data-Hora'),
            y=alt.Y('percentual_volume:Q', title='% Volume')
        ))

    points_vol_intrahora = (alt.Chart(previsao_vol_intrahora)
        .mark_circle(color='#9032bb', size=30)
        .encode(
            x='ds:T',
            y='percentual_volume:Q',
            tooltip=[
                alt.Tooltip('ds:T', title='Data-Hora', format='%d/%m/%Y %H:%M'),
                alt.Tooltip('y:Q', title='Volume'),
                alt.Tooltip('percentual_volume:Q', title='% Volume', format='.2f')
            ]
        ))

    st.altair_chart(chart_vol_intrahora + points_vol_intrahora, use_container_width=True)

    chart_tma_intrahora = (alt.Chart(previsao_tma_intrahora)
        .mark_line(color='#4b0081')
        .encode(
            x=alt.X('ds:T', title='Data-Hora'),
            y=alt.Y('percentual_tma:Q', title='% TMA')
        ))

    points_tma_intrahora = (alt.Chart(previsao_tma_intrahora)
        .mark_circle(color='#9032bb', size=30)
        .encode(
            x='ds:T',
            y='percentual_tma:Q',
            tooltip=[
                alt.Tooltip('ds:T', title='Data-Hora', format='%d/%m/%Y %H:%M'),
                alt.Tooltip('tma:Q', title='TMA (s)'),
                alt.Tooltip('percentual_tma:Q', title='% TMA', format='.2f')
            ]
        ))

    st.altair_chart(chart_tma_intrahora + points_tma_intrahora, use_container_width=True)

else:
    st.info("Por favor, envie as duas bases para gerar as projeções.")

