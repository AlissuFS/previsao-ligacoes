import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Representatividade por Dia da Semana", layout="wide")
st.title("📞 Representatividade de Ligações por Dia da Semana")

# Upload do arquivo
uploaded_file = st.file_uploader("Envie sua planilha com as colunas 'Data' e 'Quantidade de Ligações'", type=[".xlsx", ".xls", ".csv"])

# Seleção de dias da semana no topo
dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.multiselect("Selecione os dias da semana para considerar no cálculo", dias_semana_port, default=dias_semana_port)

# Processamento
if uploaded_file:
    try:
        # Leitura do arquivo
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        df.columns = df.columns.str.strip()
        if 'Data' not in df.columns or 'Quantidade de Ligações' not in df.columns:
            st.error("A planilha deve conter as colunas 'Data' e 'Quantidade de Ligações'.")
            st.stop()

        # Tratamento de datas
        df['ds'] = pd.to_datetime(df['Data'])
        df['y'] = df['Quantidade de Ligações'].clip(lower=0)
        df['ano_mes'] = df['ds'].dt.to_period('M')
        df['dia_semana'] = df['ds'].dt.day_name()
        df['dia_semana_pt'] = df['dia_semana'].map({
            'Monday': 'Segunda-feira',
            'Tuesday': 'Terça-feira',
            'Wednesday': 'Quarta-feira',
            'Thursday': 'Quinta-feira',
            'Friday': 'Sexta-feira',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        })

        # Último mês completo
        meses_disponiveis = df['ano_mes'].sort_values().unique()
        ultimo_mes = pd.Period(datetime.now(), freq='M') - 1
        if ultimo_mes not in meses_disponiveis:
            ultimo_mes = df['ano_mes'].max()

        df_mes = df[df['ano_mes'] == ultimo_mes].copy()

        if df_mes.empty:
            st.warning("Não há dados suficientes para o último mês completo.")
            st.stop()

        # Cálculo de ordem de ocorrência no mês
        def ocorrencia_semana(data):
            dia = data.day
            dia_semana = data.weekday()
            return sum((datetime(data.year, data.month, d).weekday() == dia_semana)
                       for d in range(1, dia + 1))

        df_mes['ordem'] = df_mes['ds'].apply(ocorrencia_semana)
        ordinais = {1: '1ª', 2: '2ª', 3: '3ª', 4: '4ª', 5: '5ª'}
        df_mes['rotulo'] = df_mes.apply(lambda row: f"{ordinais.get(row['ordem'], str(row['ordem']) + 'ª')} {row['dia_semana_pt']}", axis=1)

        # Agrupamento
        grupo = df_mes.groupby('rotulo')['y'].sum()
        grupo_total = grupo.sum()

        # Criar todos os rótulos esperados (para mostrar 0% se não tiver)
        todos_rotulos = []
        for dia in dias_selecionados:
            for i in range(1, 6):
                todos_rotulos.append(f"{ordinais.get(i, str(i)+'ª')} {dia}")
        grupo = grupo.reindex(todos_rotulos, fill_value=0)

        percentual = grupo / grupo_total * 100
        percentual_formatado = percentual.apply(lambda x: f"{x:.2f}%" if x > 0 else "0%")

        st.subheader(f"📅 Representatividade - Mês {ultimo_mes.strftime('%m/%Y')}")
        st.dataframe(percentual_formatado.rename("Percentual"), use_container_width=True)

        # Gráfico de linha
        st.subheader("📈 Gráfico de Linha - Representatividade por Dia e Ocorrência")
        st.line_chart(percentual)

        # Exportação
        st.subheader("📥 Baixar Resultado")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            pd.DataFrame({
                'Dia Ocorrência': percentual.index,
                'Percentual': percentual.values
            }).to_excel(writer, index=False, sheet_name="Representatividade")
        st.download_button("Baixar Excel", data=buffer.getvalue(), file_name="representatividade_dia_semana.xlsx")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
