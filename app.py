import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import streamlit as st
import io
import holidays

st.set_page_config(page_title="Previsão de Ligações por Dia da Semana", layout="wide") st.title("📞 Previsão de Ligações com IA")

uploaded_file = st.file_uploader("Envie sua planilha Excel com colunas 'Data' e 'Quantidade de Ligações'", type=[".xlsx", ".xls", ".csv"]) feriados_custom = st.text_area("Feriados personalizados (1 por linha, formato: AAAA-MM-DD)", height=100) pred_por_mes = st.checkbox("Exibir previsão mensal agregada") pred_por_hora = st.checkbox("Exibir previsção por hora (necessário ter coluna de hora na planilha)")

if uploaded_file: try: df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls") else pd.read_csv(uploaded_file) df.columns = df.columns.str.strip()

# Verifica se coluna de hora existe
    if pred_por_hora and 'Hora' in df.columns:
        df['ds'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'])
    else:
        df = df.rename(columns={"Data": "ds", "Quantidade de Ligações": "y"})
        df['ds'] = pd.to_datetime(df['ds'])

    df = df.sort_values('ds')

    # Remover outliers (método do IQR)
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    filtro = (df['y'] >= Q1 - 1.5 * IQR) & (df['y'] <= Q3 + 1.5 * IQR)
    df_filtrado = df[filtro].copy()

    st.subheader("Visualização dos Dados")
    st.write("Amostra dos dados filtrados:", df_filtrado.tail())

    # Adicionar feriados personalizados ao modelo
    feriados_lista = [x.strip() for x in feriados_custom.splitlines() if x.strip() != '']
    df_feriados = pd.DataFrame({'holiday': 'feriado_pessoal', 'ds': pd.to_datetime(feriados_lista), 'lower_window': 0, 'upper_window': 1}) if feriados_lista else None

    modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    if df_feriados is not None:
        modelo = modelo.add_country_holidays(country_name='BR')
        modelo = modelo.add_seasonality(name='feriados_custom', period=365.25, fourier_order=10)
        modelo = modelo.add_regressor('feriado_pessoal', mode='additive')

    modelo.fit(df_filtrado)
    futuro = modelo.make_future_dataframe(periods=90, freq='D')

    previsao = modelo.predict(futuro)
    previsao['dia_semana'] = previsao['ds'].dt.day_name(locale='pt_BR')
    dias_futuros = previsao[previsao['ds'] > df['ds'].max()]

    percentual = dias_futuros.groupby('dia_semana')['yhat'].sum()
    percentual = percentual / percentual.sum() * 100
    percentual = percentual.sort_index()

    st.subheader("📊 Percentual Projetado por Dia da Semana")
    st.bar_chart(percentual)

    if pred_por_mes:
        st.subheader("📅 Previsão Mensal Agregada")
        previsao['mes'] = previsao['ds'].dt.to_period('M')
        mensal = previsao.groupby('mes')['yhat'].sum()
        st.line_chart(mensal)

    if pred_por_hora and 'Hora' in df.columns:
        st.subheader("⏰ Previsão por Hora")
        previsao['hora'] = previsao['ds'].dt.hour
        por_hora = previsao.groupby('hora')['yhat'].mean()
        st.bar_chart(por_hora)

    st.subheader("📥 Baixar Resultado")
    resultado_df = percentual.reset_index()
    resultado_df.columns = ['Dia da Semana', 'Percentual Projetado']

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        resultado_df.to_excel(writer, index=False, sheet_name="Projecao")
        if pred_por_mes:
            mensal.reset_index().to_excel(writer, index=False, sheet_name="Mensal")
        if pred_por_hora and 'Hora' in df.columns:
            por_hora.reset_index().to_excel(writer, index=False, sheet_name="PorHora")

    st.download_button("Baixar Excel", data=buffer.getvalue(), file_name="projecao_ligacoes_completa.xlsx")

except Exception as e:
    st.error(f"Erro ao processar a planilha: {str(e)}")
