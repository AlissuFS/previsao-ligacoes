import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

st.set_page_config(page_title="SERCOM Digitais - Projeção", layout="wide", initial_sidebar_state="expanded")

st.sidebar.image(
    "https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif",
    use_container_width=True
)
st.sidebar.markdown("### 🔍 Configurações")

uploaded_file = st.sidebar.file_uploader("📁 Upload: Arquivo com colunas 'Data', 'Quantidade de Ligações' e 'TMA'", type=[".xlsx", ".xls", ".csv"])

dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.sidebar.multiselect("📍 Dias da semana considerados", dias_semana_port, default=dias_semana_port)

# Função para detectar ocorrência do dia da semana no mês
def ocorrencia_semana(data):
    dia_semana = data.weekday()
    dias_mes = pd.date_range(start=data.replace(day=1), end=data)
    return sum(d.weekday() == dia_semana for d in dias_mes)

# Função para remover outliers por dia da semana e ordem de ocorrência
def remover_outliers_detalhado(df, valor_col):
    df = df.copy()
    df['ordem'] = df['ds'].apply(ocorrencia_semana)
    grupos = df.groupby(['dia_semana_pt', 'ordem'])
    df_filtrado = []

    for (dia, ordem), grupo in grupos:
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
    df_tma = df_limpo_tma.rename(columns={'tma': 'y'})[['ds', 'y']].drop_duplicates(subset=['ds']).reset_index(drop=True)

    modelo_volume = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_volume.fit(df_limpo_volume)
    modelo_tma = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma.fit(df_tma)

    mes_proj = st.sidebar.date_input("🗓 Escolha o primeiro dia do mês para projeção futura", value=datetime.today().replace(day=1))
    mes_proj = pd.to_datetime(mes_proj)
    dias_futuros = pd.date_range(start=mes_proj, end=(mes_proj + pd.offsets.MonthEnd(0)))
    df_futuro = pd.DataFrame({'ds': dias_futuros})

    previsao_volume = modelo_volume.predict(df_futuro)[['ds', 'yhat']].rename(columns={'yhat': 'y'})
    previsao_tma = modelo_tma.predict(df_futuro)[['ds', 'yhat']].rename(columns={'yhat': 'tma'})

    df_prev = pd.merge(previsao_volume, previsao_tma, on='ds')
    df_prev['percentual_volume'] = df_prev['y'] / df_prev['y'].sum() * 100
    media_tma = df_prev['tma'].mean()
    df_prev['percentual_tma'] = df_prev['tma'] / media_tma * 100

    df_prev_formatado = df_prev[['ds', 'y', 'percentual_volume', 'tma', 'percentual_tma']].copy()
    df_prev_formatado.columns = ['Data', 'Volume projetado', '% curva volume', 'TMA projetado (s)', '% curva TMA']
    df_prev_formatado['Data'] = pd.to_datetime(df_prev_formatado['Data'])

    st.success("Previsões geradas com sucesso!")
    st.dataframe(df_prev_formatado.style.format({
        'Volume projetado': '{:,.0f}',
        '% curva volume': '{:.2f}%'.format,
        'TMA projetado (s)': '{:,.0f}',
        '% curva TMA': '{:.2f}%'.format
    }), use_container_width=True)

    chart = alt.Chart(df_prev_formatado).transform_fold(
        ['% curva volume', '% curva TMA'],
        as_=['Métrica', 'Percentual']
    ).mark_line(point=True).encode(
        x=alt.X('Data:T', title='Data'),
        y=alt.Y('Percentual:Q', title='Percentual (%)'),
        color='Métrica:N',
        tooltip=[
            alt.Tooltip('Data:T', title='Data'),
            alt.Tooltip('Volume projetado:Q', title='Volume', format='.0f'),
            alt.Tooltip('% curva volume:N', title='% Volume'),
            alt.Tooltip('TMA projetado (s):Q', title='TMA (s)', format='.0f'),
            alt.Tooltip('% curva TMA:N', title='% TMA')
        ]
    ).properties(width=900, height=400).interactive()

    st.altair_chart(chart, use_container_width=True)

    df_prev_formatado['Data'] = df_prev_formatado['Data'].dt.strftime('%d/%m/%Y')

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_prev_formatado.to_excel(writer, index=False, sheet_name='Projecao')

    st.download_button(
        label="📥 Baixar projeção completa",
        data=buffer.getvalue(),
        file_name="projecao_completa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
