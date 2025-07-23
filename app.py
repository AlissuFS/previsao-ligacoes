import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet

# Configurar a página
st.set_page_config(page_title="Comparativo e Projeção de Ligações", layout="wide")
st.title("📞 Comparativo e Projeção de Ligações por Dia da Semana")

# Upload
uploaded_file = st.file_uploader("Envie a planilha com 'Data' e 'Quantidade de Ligações'", type=[".xlsx", ".xls", ".csv"])

# Dias da semana selecionáveis
dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.multiselect("Selecionar dias da semana para análise", dias_semana_port, default=dias_semana_port)

if uploaded_file:
    try:
        # Leitura
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        df.columns = df.columns.str.strip()
        if 'Data' not in df.columns or 'Quantidade de Ligações' not in df.columns:
            st.error("A planilha precisa conter 'Data' e 'Quantidade de Ligações'.")
            st.stop()

        df['ds'] = pd.to_datetime(df['Data'])
        df['y'] = df['Quantidade de Ligações'].clip(lower=0)
        df['ano_mes'] = df['ds'].dt.to_period('M')
        df['dia_semana'] = df['ds'].dt.day_name()
        df['dia_semana_pt'] = df['dia_semana'].map({
            'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
            'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        })

        meses_disponiveis = sorted(df['ano_mes'].unique(), reverse=True)
        mes_map = {str(m): m for m in meses_disponiveis}

        col1, col2 = st.columns(2)
        with col1:
            mes_base_str = st.selectbox("🗓️ Mês base (Histórico)", list(mes_map.keys()), index=0)
        with col2:
            mes_proj_str = st.text_input("🔮 Mês projetado (AAAA-MM)", value=str((datetime.now() + timedelta(days=30)).strftime('%Y-%m')))

        mes_base = mes_map[mes_base_str]
        mes_proj = pd.Period(mes_proj_str, freq='M')

        def calcular_curva(df_mes, dias_filtrados, sufixo=""):
            df_mes = df_mes[df_mes['dia_semana_pt'].isin(dias_filtrados)].copy()
            if df_mes.empty:
                return pd.Series(dtype=float)

            def ocorrencia_semana(data):
                dia = data.day
                dia_semana = data.weekday()
                return sum((datetime(data.year, data.month, d).weekday() == dia_semana)
                           for d in range(1, dia + 1))

            df_mes['ordem'] = df_mes['ds'].apply(ocorrencia_semana)
            ordinais = {1: '1ª', 2: '2ª', 3: '3ª', 4: '4ª', 5: '5ª'}
            df_mes['rotulo'] = df_mes.apply(lambda row: f"{ordinais.get(row['ordem'], str(row['ordem']) + 'ª')} {row['dia_semana_pt']}", axis=1)
            grupo = df_mes.groupby('rotulo')['y'].sum()
            grupo_total = grupo.sum()
            percentual = grupo / grupo_total * 100
            percentual.name = f"Percentual{sufixo}"
            return percentual

        curva_base = calcular_curva(df[df['ano_mes'] == mes_base], dias_selecionados, sufixo=" (Histórico)")

        # Projeção via Prophet se mês futuro não estiver na base
        df_proj = df[df['ano_mes'] == mes_proj]

        if df_proj.empty:
            st.info("🔮 Gerando previsão com IA para o mês selecionado...")

            # Limpeza de outliers
            Q1 = df['y'].quantile(0.25)
            Q3 = df['y'].quantile(0.75)
            IQR = Q3 - Q1
            filtro = (df['y'] >= Q1 - 1.5 * IQR) & (df['y'] <= Q3 + 1.5 * IQR)
            df_limpo = df[filtro][['ds', 'y']].copy().sort_values('ds')

            modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            modelo.fit(df_limpo)

            dias_proj = (mes_proj.to_timestamp(), (mes_proj + 1).to_timestamp() - timedelta(days=1))
            futuro = pd.date_range(start=dias_proj[0], end=dias_proj[1], freq='D')
            df_futuro = pd.DataFrame({'ds': futuro})
            previsao = modelo.predict(df_futuro)

            df_prev = previsao[['ds', 'yhat']].rename(columns={'yhat': 'y'})
            df_prev['dia_semana'] = df_prev['ds'].dt.day_name()
            df_prev['dia_semana_pt'] = df_prev['dia_semana'].map({
                'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
                'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            })
            curva_proj = calcular_curva(df_prev, dias_selecionados, sufixo=" (Projetado)")
        else:
            curva_proj = calcular_curva(df_proj, dias_selecionados, sufixo=" (Projetado)")

        curva_comparativa = pd.concat([curva_base, curva_proj], axis=1).fillna(0)
        curva_fmt = curva_comparativa.copy()
        curva_fmt['Histórico (%)'] = curva_fmt.iloc[:, 0].apply(lambda x: f"{x:.2f}%" if x > 0 else "0%")
        curva_fmt['Projetado (%)'] = curva_fmt.iloc[:, 1].apply(lambda x: f"{x:.2f}%" if x > 0 else "0%")
        curva_fmt = curva_fmt[['Histórico (%)', 'Projetado (%)']]

        st.subheader(f"📊 Comparativo: {mes_base.strftime('%m/%Y')} vs {mes_proj.strftime('%m/%Y')}")
        st.dataframe(curva_fmt, use_container_width=True)

        st.subheader("📈 Gráfico de Linha - Comparativo")
        st.line_chart(curva_comparativa)

        st.subheader("📄 Baixar Excel")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            curva_comparativa.reset_index().to_excel(writer, index=False, sheet_name="Comparativo")
        st.download_button("Download Excel", data=buffer.getvalue(), file_name="comparativo_projecao_ligacoes.xlsx")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
