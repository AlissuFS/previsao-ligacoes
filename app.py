import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
import io
import holidays

# Configuração da página
st.set_page_config(page_title="Previsão de Ligações por Dia da Semana", layout="wide")
st.title("📞 Previsão de Ligações com IA")

# Interface do usuário
uploaded_file = st.file_uploader("Envie sua planilha Excel com colunas 'Data' e 'Quantidade de Ligações'", type=[".xlsx", ".xls", ".csv"])
feriados_custom = st.text_area("Feriados personalizados (1 por linha, formato: AAAA-MM-DD)", height=100)
pred_por_mes = st.checkbox("Exibir previsão mensal agregada")
pred_por_hora = st.checkbox("Exibir previsão por hora (necessário ter coluna de hora na planilha)")

# Se arquivo foi enviado
if uploaded_file:
    try:
        # Carregar dados
        if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        # Validação de colunas
        if 'Data' not in df.columns or 'Quantidade de Ligações' not in df.columns:
            st.error("A planilha deve conter as colunas 'Data' e 'Quantidade de Ligações'.")
            st.stop()

        # Processamento das datas e horas
        if pred_por_hora and 'Hora' in df.columns:
            df['ds'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'].astype(str))
            df = df.rename(columns={"Quantidade de Ligações": "y"})
        else:
            df = df.rename(columns={"Data": "ds", "Quantidade de Ligações": "y"})
            df['ds'] = pd.to_datetime(df['ds'])

        df = df.sort_values('ds')

        # Remoção de outliers
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        filtro = (df['y'] >= Q1 - 1.5 * IQR) & (df['y'] <= Q3 + 1.5 * IQR)
        df_filtrado = df[filtro].copy()

        st.subheader("Visualização dos Dados")
        st.write("Amostra dos dados filtrados:", df_filtrado.tail())

        # Feriados personalizados
        feriados_lista = [x.strip() for x in feriados_custom.splitlines() if x.strip() != '']
        df_feriados = pd.DataFrame({
            'holiday': 'feriado_pessoal',
            'ds': pd.to_datetime(feriados_lista),
            'lower_window': 0,
            'upper_window': 1
        }) if feriados_lista else None

        # Modelo Prophet
        modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        if df_feriados is not None:
            modelo = modelo.add_country_holidays(country_name='BR')
            modelo.add_seasonality(name='feriados_custom', period=365.25, fourier_order=10)

        modelo.fit(df_filtrado)

        # Previsão
        futuro = modelo.make_future_dataframe(periods=90, freq='D')
        previsao = modelo.predict(futuro)

        # Mapear dias da semana para português
        dias_em_portugues = {
            'Monday': 'Segunda-feira',
            'Tuesday': 'Terça-feira',
            'Wednesday': 'Quarta-feira',
            'Thursday': 'Quinta-feira',
            'Friday': 'Sexta-feira',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        }
        previsao['dia_semana'] = previsao['ds'].dt.day_name().map(dias_em_portugues)
        dias_futuros = previsao[previsao['ds'] > df['ds'].max()]

        # Percentual por dia da semana
        percentual = dias_futuros.groupby('dia_semana')['yhat'].sum()
        percentual = percentual / percentual.sum() * 100
        percentual = percentual.sort_index()

        st.subheader("📊 Percentual Projetado por Dia da Semana")
        st.bar_chart(percentual)

        # Previsão mensal
        if pred_por_mes:
            st.subheader("📅 Previsão Mensal Agregada")
            previsao['mes'] = previsao['ds'].dt.to_period('M')
            mensal = previsao.groupby('mes')['yhat'].sum()
            st.line_chart(mensal)

        # Previsão por hora
        if pred_por_hora and 'Hora' in df.columns:
            st.subheader("⏰ Previsão por Hora")
            previsao['hora'] = previsao['ds'].dt.hour
            por_hora = previsao.groupby('hora')['yhat'].mean()
            st.bar_chart(por_hora)

        # Exportar resultado
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
