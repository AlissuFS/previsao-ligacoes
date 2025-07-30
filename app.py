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

if uploaded_file_diario and uploaded_file_intrahora:
    # Leitura e preparação base diária
    df_diario = pd.read_excel(uploaded_file_diario) if uploaded_file_diario.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file_diario)
    df_diario.columns = df_diario.columns.str.strip()
    required_cols_diario = {'Data', 'Quantidade de Ligações', 'TMA'}
    if not required_cols_diario.issubset(df_diario.columns):
        st.error(f"Base diária deve conter as colunas: {required_cols_diario}")
        st.stop()
    df_diario = preparar_df_diario(df_diario)

    # Remove outliers para Volume e TMA diários
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

    # Remove outliers intrahora
    df_limpo_vol_intrahora = remover_outliers_detalhado(df_intrahora, 'y')[['ds','y']].drop_duplicates().reset_index(drop=True)
    df_limpo_tma_intrahora = remover_outliers_detalhado(df_intrahora, 'tma')
    df_tma_intrahora = df_limpo_tma_intrahora.groupby('ds')['tma'].mean().reset_index().rename(columns={'tma':'y'})

    # Modelos Prophet diários e intrahora
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

    # Datas futuras diárias e intrahora
    dias_futuros_diario = pd.date_range(start=mes_proj, end=(mes_proj + pd.offsets.MonthEnd(0)))
    df_futuro_diario = pd.DataFrame({'ds': dias_futuros_diario})

    datas_futuras_intrahora = gerar_datas_futuras_intrahora(mes_proj)
    df_futuro_intrahora = pd.DataFrame({'ds': datas_futuras_intrahora})

    # Previsões diárias
    previsao_vol_diario = modelo_vol_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'y'})
    previsao_vol_diario['y'] = previsao_vol_diario['y'].apply(lambda x: max(1, x))
    previsao_tma_diario = modelo_tma_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'tma'})

    # Previsões intrahora
    previsao_vol_intrahora = modelo_vol_intrahora.predict(df_futuro_intrahora)[['ds','yhat']].rename(columns={'yhat':'y'})
    previsao_vol_intrahora['y'] = previsao_vol_intrahora['y'].apply(lambda x: max(0.1, x))
    previsao_tma_intrahora = modelo_tma_intrahora.predict(df_futuro_intrahora)[['ds','yhat']].rename(columns={'yhat':'tma'})

    # Mapear dias da semana em português nas previsões
    mapa_dias = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }

    previsao_vol_intrahora['dia_semana_pt'] = previsao_vol_intrahora['ds'].dt.day_name().map(mapa_dias)
    previsao_tma_intrahora['dia_semana_pt'] = previsao_tma_intrahora['ds'].dt.day_name().map(mapa_dias)
    previsao_vol_diario['dia_semana_pt'] = previsao_vol_diario['ds'].dt.day_name().map(mapa_dias)
    previsao_tma_diario['dia_semana_pt'] = previsao_tma_diario['ds'].dt.day_name().map(mapa_dias)

    # Zerar valores dos dias desativados
    dias_desativados = [dia for dia, ativo in dias_funcionamento.items() if not ativo]

    previsao_vol_intrahora.loc[previsao_vol_intrahora['dia_semana_pt'].isin(dias_desativados), 'y'] = 0
    previsao_tma_intrahora.loc[previsao_tma_intrahora['dia_semana_pt'].isin(dias_desativados), 'tma'] = 0
    previsao_vol_diario.loc[previsao_vol_diario['dia_semana_pt'].isin(dias_desativados), 'y'] = 0
    previsao_tma_diario.loc[previsao_tma_diario['dia_semana_pt'].isin(dias_desativados), 'tma'] = 0

    # --- Ajuste para curva intrahora seguir proporcionalmente a curva diária ---

    # 1) Calcular total diário previsto (Volume e TMA)
    totais_volume_diario = previsao_vol_diario.set_index('ds')['y']
    totais_tma_diario = previsao_tma_diario.set_index('ds')['tma']

    # 2) Agregar previsão intrahora por dia para Volume e TMA
    soma_intrahora_vol = previsao_vol_intrahora.groupby(previsao_vol_intrahora['ds'].dt.date)['y'].transform('sum')
    media_intrahora_tma = previsao_tma_intrahora.groupby(previsao_tma_intrahora['ds'].dt.date)['tma'].transform('mean')

    # 3) Criar coluna data para acesso ao total diário
    previsao_vol_intrahora['data'] = previsao_vol_intrahora['ds'].dt.date
    previsao_tma_intrahora['data'] = previsao_tma_intrahora['ds'].dt.date

    # 4) Calcular fator ajuste para volume intrahora: proporção da soma intrahora vs total previsto diário
    # Se soma_intrahora_vol = 0, evitar divisão por zero, fator ajuste = 0
    fator_ajuste_vol = []
    for i, row in previsao_vol_intrahora.iterrows():
        data = row['data']
        soma_intrahora = soma_intrahora_vol[i]
        total_diario = totais_volume_diario.get(pd.to_datetime(data), 0)
        if soma_intrahora > 0:
            fator = total_diario / soma_intrahora
        else:
            fator = 0
        fator_ajuste_vol.append(fator)
    previsao_vol_intrahora['fator_ajuste'] = fator_ajuste_vol

    # Ajustar volume intrahora multiplicando pelo fator
    previsao_vol_intrahora['y'] = previsao_vol_intrahora['y'] * previsao_vol_intrahora['fator_ajuste']

    # 5) Ajustar TMA intrahora proporcional ao TMA diário
    fator_ajuste_tma = []
    for i, row in previsao_tma_intrahora.iterrows():
        data = row['data']
        media_intrahora = media_intrahora_tma[i]
        total_diario = totais_tma_diario.get(pd.to_datetime(data), 0)
        if media_intrahora > 0:
            fator = total_diario / media_intrahora
        else:
            fator = 0
        fator_ajuste_tma.append(fator)
    previsao_tma_intrahora['fator_ajuste'] = fator_ajuste_tma

    # Ajustar TMA intrahora multiplicando pelo fator
    previsao_tma_intrahora['tma'] = previsao_tma_intrahora['tma'] * previsao_tma_intrahora['fator_ajuste']

    # 6) Agora recalcular percentuais intrahora após ajuste para volume e TMA
    previsao_vol_intrahora['percentual_volume'] = previsao_vol_intrahora.groupby('data')['y'].transform(
        lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0
    )
    previsao_tma_intrahora['percentual_tma'] = previsao_tma_intrahora.groupby('data')['tma'].transform(
        lambda x: (x / x.mean() * 100) if x.mean() > 0 else 0
    )

    # Percentual diário já calculado antes, só formatar para string
    if previsao_vol_diario['y'].sum() > 0:
        previsao_vol_diario['percentual_volume'] = previsao_vol_diario['y'] / previsao_vol_diario['y'].sum() * 100
    else:
        previsao_vol_diario['percentual_volume'] = 0

    media_tma_diario = previsao_tma_diario['tma'].mean() if previsao_tma_diario['tma'].mean() > 0 else 1
    previsao_tma_diario['percentual_tma'] = previsao_tma_diario['tma'] / media_tma_diario * 100

    # === Interface para mostrar gráficos ===
    st.title("📊 Projeção de Volume e TMA - Curvas Diária e Intrahora")

    aba = st.selectbox("📈 Escolha a curva para visualizar:", options=['Volume', 'TMA'])

    if aba == 'Volume':
        # Curva diária
        grafico_diario = alt.Chart(previsao_vol_diario).mark_line(point=True).encode(
            x='ds:T',
            y=alt.Y('y', title='Volume Diário'),
            tooltip=['ds:T', alt.Tooltip('y', format=',.0f'), alt.Tooltip('percentual_volume', format='.2f')]
        ).properties(title='Previsão Volume Diária')

        # Curva intrahora
        grafico_intrahora = alt.Chart(previsao_vol_intrahora).mark_line(point=False, strokeDash=[5,5], color='#9032bb').encode(
            x='ds:T',
            y=alt.Y('y', title='Volume Intrahora'),
            tooltip=['ds:T', alt.Tooltip('y', format=',.2f'), alt.Tooltip('percentual_volume', format='.2f')]
        ).properties(title='Previsão Volume Intrahora (Ajustada)')

        st.altair_chart(grafico_diario + grafico_intrahora, use_container_width=True)

    else:
        # Curva diária TMA
        grafico_diario_tma = alt.Chart(previsao_tma_diario).mark_line(point=True).encode(
            x='ds:T',
            y=alt.Y('tma', title='TMA Diário (segundos)'),
            tooltip=['ds:T', alt.Tooltip('tma', format=',.2f'), alt.Tooltip('percentual_tma', format='.2f')]
        ).properties(title='Previsão TMA Diária')

        # Curva intrahora TMA
        grafico_intrahora_tma = alt.Chart(previsao_tma_intrahora).mark_line(point=False, strokeDash=[5,5], color='#9032bb').encode(
            x='ds:T',
            y=alt.Y('tma', title='TMA Intrahora (segundos)'),
            tooltip=['ds:T', alt.Tooltip('tma', format=',.2f'), alt.Tooltip('percentual_tma', format='.2f')]
        ).properties(title='Previsão TMA Intrahora (Ajustada)')

        st.altair_chart(grafico_diario_tma + grafico_intrahora_tma, use_container_width=True)

else:
    st.warning("Por favor, faça o upload das bases diária e intrahora para prosseguir.")

