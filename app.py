import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta, time
from prophet import Prophet
import altair as alt

st.set_page_config(page_title="SERCOM Digitais - Projeção e Dimensionamento", layout="wide")

st.title("🔮 SERCOM Digitais - Projeção e Dimensionamento de Operação")

st.sidebar.header("⚙️ Configurações")

# Upload do arquivo
dados = st.sidebar.file_uploader("📂 Importar planilha com colunas 'Data', 'Quantidade de Ligações' e 'TMA'", type=["csv", "xlsx"])

# Horários da operação por dia da semana
st.sidebar.subheader("🕒 Horário de Funcionamento")
hr_semana_ini = st.sidebar.time_input("Início Seg a Sex", value=time(8, 0))
hr_semana_fim = st.sidebar.time_input("Fim Seg a Sex", value=time(20, 0))
hr_sabado_ini = st.sidebar.time_input("Início Sábado", value=time(9, 0))
hr_sabado_fim = st.sidebar.time_input("Fim Sábado", value=time(15, 0))
hr_domingo_ini = st.sidebar.time_input("Início Domingo", value=time(10, 0))
hr_domingo_fim = st.sidebar.time_input("Fim Domingo", value=time(14, 0))

# Taxa da operação
total_hcs = st.sidebar.number_input("👥 Total de HCs da célula", min_value=1, value=8)
pico_hcs_logados = st.sidebar.number_input("📈 Pico de HCs logados simultaneamente", min_value=1, value=4)
taxa_operacao = total_hcs / pico_hcs_logados if pico_hcs_logados else 0
st.sidebar.markdown(f"**Taxa da Operação:** {taxa_operacao:.2f}")

# NS e SLA
st.sidebar.subheader("🎯 NS e SLA")
ns_percentual = st.sidebar.slider("Nível de Serviço (%)", 50, 100, 80)
sla_segundos = st.sidebar.number_input("SLA (segundos)", min_value=10, max_value=300, value=60)

# Jornada do operador
st.sidebar.subheader("⏱️ Jornada do Operador")
jornada_opcao = st.sidebar.selectbox("Selecione Jornada", ["06:20:00", "08:10:00"])
pausas_jornada = {
    "06:20:00": [("09:20", 10), ("11:30", 20), ("14:00", 10)],
    "08:10:00": [("11:10", 10), ("13:00", 60), ("16:10", 10)]
}[jornada_opcao]

# Função para aplicar pausas a uma curva horária
def aplicar_pausas(curva, pausas):
    curva = curva.copy()
    for hora_str, duracao in pausas:
        h = int(hora_str.split(":")[0])
        idx = curva[curva['hora'] == h].index
        if not idx.empty:
            curva.loc[idx, 'disponiveis'] *= max(0, 1 - duracao / 60)  # Reduz disponibilidade proporcionalmente
    return curva

# Distribui volume do dia por hora com curva horária padrão
def curva_horaria(volume_total):
    distribuicao = np.array([0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.11, 0.12, 0.11, 0.10, 0.08, 0.06, 0.04, 0.03])
    distribuicao /= distribuicao.sum()
    horas = list(range(8, 22))  # 14 horas das 8h às 21h inclusive
    volume_horario = volume_total * distribuicao
    return pd.DataFrame({"hora": horas, "volume": volume_horario})

if dados:
    ext = dados.name.split(".")[-1]
    df = pd.read_excel(dados) if ext in ["xls", "xlsx"] else pd.read_csv(dados)
    df.columns = df.columns.str.strip()
    if not set(['Data', 'Quantidade de Ligações', 'TMA']).issubset(df.columns):
        st.error("A planilha deve conter as colunas: 'Data', 'Quantidade de Ligações' e 'TMA'")
        st.stop()

    df['ds'] = pd.to_datetime(df['Data'])
    df['y'] = pd.to_numeric(df['Quantidade de Ligações'], errors='coerce').fillna(0).clip(lower=0)
    df['tma'] = pd.to_numeric(df['TMA'], errors='coerce').fillna(0).clip(lower=0)

    df_volume = df[['ds', 'y']].dropna()
    df_tma = df[['ds', 'tma']].dropna().groupby('ds').mean().reset_index()
    
    modelo_v = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_v.fit(df_volume)
    modelo_t = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_t.fit(df_tma.rename(columns={'tma': 'y'}))

    st.sidebar.subheader("📅 Mês de Projeção")
    data_inicio_proj = st.sidebar.date_input("Data inicial", value=datetime.today().replace(day=1))
    datas_proj = pd.date_range(start=data_inicio_proj, end=data_inicio_proj + pd.offsets.MonthEnd(0))
    futuro = pd.DataFrame({"ds": datas_proj})

    prev_v = modelo_v.predict(futuro)[['ds', 'yhat']].rename(columns={'yhat': 'volume'})
    prev_t = modelo_t.predict(futuro)[['ds', 'yhat']].rename(columns={'yhat': 'tma'})
    previsao = pd.merge(prev_v, prev_t, on='ds')

    st.success("Previsão gerada com sucesso!")
    st.dataframe(previsao)

    st.markdown("### 📊 Curva Intrahora e Dimensionamento")

    resultados = []
    for _, linha in previsao.iterrows():
        vol_total = max(linha['volume'], 1)  # Garantir valor positivo mínimo
        tma_minutos = linha['tma'] / 60  # Convertendo segundos para minutos
        curva = curva_horaria(vol_total)
        # Calcular tráfego em Erlang por hora: volume por hora * TMA (minutos) / 60
        curva['trafego_erlang'] = curva['volume'] * tma_minutos / 60
        curva['disponiveis'] = 1.0  # 100% disponibilidade inicial
        curva = aplicar_pausas(curva, pausas_jornada)
        # Ajuste pela taxa da operação
        curva['PA_necessarios'] = curva['trafego_erlang'] * taxa_operacao
        curva['Data'] = linha['ds'].date()
        resultados.append(curva)

    df_dimens = pd.concat(resultados)

    st.dataframe(df_dimens)

    chart = alt.Chart(df_dimens).mark_line().encode(
        x=alt.X('hora:O', title='Hora do dia'),
        y=alt.Y('PA_necessarios:Q', title='PAs Necessários'),
        color=alt.Color('Data:N')
    ).properties(title="Dimensionamento intradiário", width=800)

    st.altair_chart(chart, use_container_width=True)
