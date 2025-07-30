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

# Upload bases
uploaded_file_diario = st.sidebar.file_uploader("📂 Base Diária - 'Data', 'Quantidade de Ligações', 'TMA'", type=[".xlsx", ".xls", ".csv"])
uploaded_file_intrahora = st.sidebar.file_uploader("📂 Base Intrahora - 'Intervalo', 'Dia', 'Volume', 'TMA'", type=[".xlsx", ".xls", ".csv"])

dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']

# Checkbox para ativar/desativar dias da semana
dias_funcionamento = {}
for dia in dias_semana_port:
    dias_funcionamento[dia] = st.sidebar.checkbox(f"{dia}", value=True)

# Funções auxiliares
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

def gerar_datas_futuras_intrahora(mes):
    inicio = mes
    fim = mes + pd.offsets.MonthEnd(0)
    rng = pd.date_range(start=inicio, end=fim + timedelta(days=1), freq='30min')[:-1]
    return rng

def format_num_brl(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_perc(x):
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + '%'

if uploaded_file_diario and uploaded_file_intrahora:

    # === Preparar dados diários ===
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

    # === Preparar dados intrahora ===
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

    # === Modelos Prophet ===
    modelo_vol_diario = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_vol_diario.fit(df_limpo_vol_diario)
    modelo_tma_diario = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma_diario.fit(df_tma_diario)

    modelo_vol_intrahora = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_vol_intrahora.fit(df_limpo_vol_intrahora)
    modelo_tma_intrahora = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma_intrahora.fit(df_tma_intrahora)

    # === Seleção mês para projeção ===
    mes_proj = st.sidebar.date_input("🗓 Escolha o primeiro dia do mês para projeção futura", value=datetime.today().replace(day=1))
    mes_proj = pd.to_datetime(mes_proj)

    # === Datas futuras para diária e intrahora ===
    dias_futuros_diario = pd.date_range(start=mes_proj, end=(mes_proj + pd.offsets.MonthEnd(0)))
    df_futuro_diario = pd.DataFrame({'ds': dias_futuros_diario})

    datas_futuras_intrahora = gerar_datas_futuras_intrahora(mes_proj)
    df_futuro_intrahora = pd.DataFrame({'ds': datas_futuras_intrahora})

    # === Previsões diárias ===
    previsao_vol_diario = modelo_vol_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'y'})
    previsao_vol_diario['y'] = previsao_vol_diario['y'].apply(lambda x: max(1, x))
    previsao_tma_diario = modelo_tma_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'tma'})

    # === Mapear dias da semana em português nas previsões diárias ===
    mapa_dias = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }

    previsao_vol_diario['dia_semana_pt'] = previsao_vol_diario['ds'].dt.day_name().map(mapa_dias)
    previsao_tma_diario['dia_semana_pt'] = previsao_tma_diario['ds'].dt.day_name().map(mapa_dias)

    # === Ajustar dias desativados para diária ===
    dias_desativados = [dia for dia, ativo in dias_funcionamento.items() if not ativo]

    previsao_vol_diario.loc[previsao_vol_diario['dia_semana_pt'].isin(dias_desativados), 'y'] = 0
    previsao_tma_diario.loc[previsao_tma_diario['dia_semana_pt'].isin(dias_desativados), 'tma'] = 0

    # === Calcular percentuais diários ===
    if previsao_vol_diario['y'].sum() > 0:
        previsao_vol_diario['percentual_volume'] = previsao_vol_diario['y'] / previsao_vol_diario['y'].sum() * 100
    else:
        previsao_vol_diario['percentual_volume'] = 0

    media_tma_diario = previsao_tma_diario['tma'].mean() if previsao_tma_diario['tma'].mean() > 0 else 1
    previsao_tma_diario['percentual_tma'] = previsao_tma_diario['tma'] / media_tma_diario * 100

    previsao_vol_diario['percentual_volume_str'] = previsao_vol_diario['percentual_volume'].apply(fmt_perc)
    previsao_tma_diario['percentual_tma_str'] = previsao_tma_diario['percentual_tma'].apply(fmt_perc)

    # === Cálculo do perfil intrahora histórico (percentual dentro do dia) ===
    df_intrahora = df_intrahora.copy()
    df_intrahora['data'] = df_intrahora['ds'].dt.date

    vol_diario_historico = df_intrahora.groupby('data')['y'].sum().rename('volume_diario')
    df_intrahora = df_intrahora.merge(vol_diario_historico, left_on='data', right_index=True)
    df_intrahora['percentual_intrahora'] = df_intrahora['y'] / df_intrahora['volume_diario']

    tma_diario_historico = df_intrahora.groupby('data')['tma'].mean().rename('tma_diario')
    df_intrahora = df_intrahora.merge(tma_diario_historico, left_on='data', right_index=True)
    df_intrahora['percentual_intrahora_tma'] = df_intrahora['tma'] / df_intrahora['tma_diario']

    df_intrahora['hora'] = df_intrahora['ds'].dt.time

    perfil_intrahora = df_intrahora.groupby(['dia_semana_pt', 'hora'])[['percentual_intrahora', 'percentual_intrahora_tma']].mean().reset_index()

    # === Preparar df futuro intrahora com datas e horas ===
    df_futuro_intrahora['data'] = df_futuro_intrahora['ds'].dt.date
    df_futuro_intrahora['hora'] = df_futuro_intrahora['ds'].dt.time
    df_futuro_intrahora['dia_semana_pt'] = df_futuro_intrahora['ds'].dt.day_name().map(mapa_dias)

    # Merge perfil intrahora histórico (percentuais) com df futuro intrahora
    df_futuro_intrahora = df_futuro_intrahora.merge(perfil_intrahora, on=['dia_semana_pt', 'hora'], how='left')
    df_futuro_intrahora['percentual_intrahora'] = df_futuro_intrahora['percentual_intrahora'].fillna(0)
    df_futuro_intrahora['percentual_intrahora_tma'] = df_futuro_intrahora['percentual_intrahora_tma'].fillna(0)

    # Merge previsão diária volume e TMA para cada dia na curva intrahora
    previsao_vol_diario['data'] = previsao_vol_diario['ds'].dt.date
    previsao_tma_diario['data'] = previsao_tma_diario['ds'].dt.date

    df_futuro_intrahora = df_futuro_intrahora.merge(previsao_vol_diario[['data', 'y']], on='data', how='left')
    df_futuro_intrahora = df_futuro_intrahora.merge(previsao_tma_diario[['data', 'tma']], on='data', how='left')

    # === Calcular projeção intrahora ponderada pelo perfil histórico ===
    df_futuro_intrahora['volume_projetado'] = df_futuro_intrahora['percentual_intrahora'] * df_futuro_intrahora['y']
    df_futuro_intrahora['tma_projetado'] = df_futuro_intrahora['percentual_intrahora_tma'] * df_futuro_intrahora['tma']

    # Zerar volume e TMA em dias desativados
    df_futuro_intrahora.loc[df_futuro_intrahora['dia_semana_pt'].isin(dias_desativados), ['volume_projetado', 'tma_projetado']] = 0

    # Calcular percentuais intrahora para exibição (dentro de cada dia)
    def calc_perc_intrahora(df):
        return df.groupby('data')['volume_projetado'].transform(lambda x: x / x.sum() * 100 if x.sum() > 0 else 0)

    df_futuro_intrahora['percentual_volume_intrahora'] = calc_perc_intrahora(df_futuro_intrahora)
    df_futuro_intrahora['percentual_volume_intrahora_str'] = df_futuro_intrahora['percentual_volume_intrahora'].apply(fmt_perc)

    # === Visualização ===

    st.title("📞 Projeção de Volume e TMA de Ligações")

    st.subheader("📅 Previsão diária - Volume")
    st.dataframe(previsao_vol_diario[['ds', 'y', 'percentual_volume_str']].rename(columns={'ds': 'Data', 'y': 'Volume'}))

    st.subheader("📅 Previsão diária - TMA")
    st.dataframe(previsao_tma_diario[['ds', 'tma', 'percentual_tma_str']].rename(columns={'ds': 'Data', 'tma': 'TMA'}))

    # Gráfico Volume Diário
    grafico_vol_diario = alt.Chart(previsao_vol_diario).mark_line(point=True).encode(
        x=alt.X('ds:T', title='Data'),
        y=alt.Y('y:Q', title='Volume'),
        tooltip=[
            alt.Tooltip('ds:T', title='Data'),
            alt.Tooltip('y:Q', title='Volume'),
            alt.Tooltip('percentual_volume:Q', title='Percentual (%)', format=".2f"),
            alt.Tooltip('dia_semana_pt:N', title='Dia da Semana'),
        ],
        color=alt.value('#9032bb')
    ).properties(width=800, height=300)
    st.altair_chart(grafico_vol_diario)

    # Gráfico TMA Diário
    grafico_tma_diario = alt.Chart(previsao_tma_diario).mark_line(point=True).encode(
        x=alt.X('ds:T', title='Data'),
        y=alt.Y('tma:Q', title='TMA'),
        tooltip=[
            alt.Tooltip('ds:T', title='Data'),
            alt.Tooltip('tma:Q', title='TMA'),
            alt.Tooltip('percentual_tma:Q', title='Percentual (%)', format=".2f"),
            alt.Tooltip('dia_semana_pt:N', title='Dia da Semana'),
        ],
        color=alt.value('#4b0081')
    ).properties(width=800, height=300)
    st.altair_chart(grafico_tma_diario)

    # Visualizar tabela e gráfico Intrahora Volume
    st.subheader("⏱️ Previsão intrahora - Volume")
    tabela_intrahora_vol = df_futuro_intrahora[['ds', 'volume_projetado', 'percentual_volume_intrahora_str', 'dia_semana_pt']].copy()
    tabela_intrahora_vol.columns = ['DataHora', 'Volume Projetado', '% Intrahora', 'Dia da Semana']
    st.dataframe(tabela_intrahora_vol)

    grafico_intrahora_vol = alt.Chart(df_futuro_intrahora).mark_area(opacity=0.6, line=True).encode(
        x='ds:T',
        y=alt.Y('volume_projetado:Q', stack=None, title='Volume'),
        color=alt.Color('dia_semana_pt:N', legend=alt.Legend(title='Dia da Semana')),
        tooltip=[
            alt.Tooltip('ds:T', title='DataHora'),
            alt.Tooltip('volume_projetado:Q', title='Volume Projetado'),
            alt.Tooltip('percentual_volume_intrahora:Q', title='% Intrahora', format=".2f"),
            alt.Tooltip('dia_semana_pt:N', title='Dia da Semana'),
        ]
    ).properties(width=800, height=300)
    st.altair_chart(grafico_intrahora_vol)

    # Visualizar tabela e gráfico Intrahora TMA
    st.subheader("⏱️ Previsão intrahora - TMA")
    tabela_intrahora_tma = df_futuro_intrahora[['ds', 'tma_projetado', 'dia_semana_pt']].copy()
    tabela_intrahora_tma.columns = ['DataHora', 'TMA Projetado', 'Dia da Semana']
    st.dataframe(tabela_intrahora_tma)

    grafico_intrahora_tma = alt.Chart(df_futuro_intrahora).mark_line(point=True, strokeDash=[5,5]).encode(
        x='ds:T',
        y=alt.Y('tma_projetado:Q', title='TMA Projetado'),
        color=alt.Color('dia_semana_pt:N', legend=alt.Legend(title='Dia da Semana')),
        tooltip=[
            alt.Tooltip('ds:T', title='DataHora'),
            alt.Tooltip('tma_projetado:Q', title='TMA Projetado'),
            alt.Tooltip('dia_semana_pt:N', title='Dia da Semana'),
        ]
    ).properties(width=800, height=300)
    st.altair_chart(grafico_intrahora_tma)

else:
    st.warning("Por favor, faça upload dos arquivos base diária e base intrahora para iniciar a projeção.")
