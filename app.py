import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

st.set_page_config(page_title="SERCOM Digitais - Projeção", layout="wide", initial_sidebar_state="expanded")

# === Estilo Visual ===
st.markdown("""
<style>
[data-testid="stSidebar"] {background-color: #4b0081;}
[data-testid="stSidebar"] * {color: white !important;}
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
[data-testid="stFileUploadDropzone"] > div {display: none !important;}
[data-testid="stFileUploadDropzone"] button {
  background-color: #9032bb !important;
  color: white !important;
  border-radius: 10px !important;
  border: none !important;
  padding: 10px 20px !important;
  box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
  font-weight: 600;
}
[data-testid="stFileUploadDropzone"] button:hover {background-color: #a84be0 !important;}
.stButton button {
  background-color: #9032bb;
  color: white;
  border: none;
  border-radius: 10px;
  box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}
.stButton button:hover {background-color: #a84be0;}
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.image(
    "https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif",
    use_container_width=True
)
st.sidebar.markdown("### 🔍 Configurações")

uploaded_file_diario = st.sidebar.file_uploader("📂 Base Diária - 'Data', 'Quantidade de Ligações', 'TMA'", type=[".xlsx", ".xls", ".csv"])
uploaded_file_intrahora = st.sidebar.file_uploader("📂 Base Intrahora - 'Intervalo', 'Dia', 'Volume', 'TMA'", type=[".xlsx", ".xls", ".csv"])

dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']

st.sidebar.markdown("### ⚙️ Dias da Semana Funcionando")
dias_funcionamento = {}
for dia in dias_semana_port:
    dias_funcionamento[dia] = st.sidebar.checkbox(f"{dia}", value=True)

mes_referencia = st.sidebar.date_input("📅 Mês referência (base diária)", value=datetime.today().replace(day=1))
mes_proj1 = st.sidebar.date_input("📅 Mês proj. 1", value=(datetime.today() + pd.DateOffset(months=1)).replace(day=1))
mes_proj2 = st.sidebar.date_input("📅 Mês proj. 2", value=(datetime.today() + pd.DateOffset(months=2)).replace(day=1))
mes_proj3 = st.sidebar.date_input("📅 Mês proj. 3", value=(datetime.today() + pd.DateOffset(months=3)).replace(day=1))

meses_projecoes = [mes_proj1, mes_proj2, mes_proj3]

# Funções auxiliares (como anteriormente)...

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
            df_filtrado.append(pd.DataFrame({'ds':[data_referencia], valor_col:[media_valor], 'dia_semana_pt':[dia], 'ordem':[ordem]}))
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
    mapa_dias = {'Monday':'Segunda-feira','Tuesday':'Terça-feira','Wednesday':'Quarta-feira',
                 'Thursday':'Quinta-feira','Friday':'Sexta-feira','Saturday':'Sábado','Sunday':'Domingo'}
    df['dia_semana_pt'] = df['dia_semana'].map(mapa_dias)
    df['ordem'] = df['ds'].apply(ocorrencia_semana)
    return df

def preparar_df_intrahora(df):
    df['ds'] = pd.to_datetime(df['Intervalo'])
    df['dia_semana'] = pd.to_datetime(df['Dia']).dt.day_name()
    mapa_dias = {'Monday':'Segunda-feira','Tuesday':'Terça-feira','Wednesday':'Quarta-feira',
                 'Thursday':'Quinta-feira','Friday':'Sexta-feira','Saturday':'Sábado','Sunday':'Domingo'}
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

def rodar_previsoes(df_diario, df_intrahora, mes_ref, meses_proj, dias_func):
    # Preparar e limpar base diária
    df_diario = preparar_df_diario(df_diario)
    df_diario = df_diario[df_diario['ds'] >= mes_ref]
    df_diario = df_diario[df_diario['ds'] < mes_ref + pd.offsets.MonthEnd(0) + timedelta(days=1)]

    df_limpo_vol_diario = remover_outliers_detalhado(df_diario, 'y')[['ds','y']].drop_duplicates().reset_index(drop=True)
    df_limpo_tma_diario = remover_outliers_detalhado(df_diario, 'tma')
    df_tma_diario = df_limpo_tma_diario.groupby('ds')['tma'].mean().reset_index().rename(columns={'tma':'y'})

    # Preparar e limpar base intrahora
    df_intrahora = preparar_df_intrahora(df_intrahora)
    df_intrahora = df_intrahora[(df_intrahora['ds'] >= mes_ref) & (df_intrahora['ds'] < mes_ref + pd.offsets.MonthEnd(0) + timedelta(days=1))]

    df_limpo_vol_intrahora = remover_outliers_detalhado(df_intrahora, 'y')[['ds','y']].drop_duplicates().reset_index(drop=True)
    df_limpo_tma_intrahora = remover_outliers_detalhado(df_intrahora, 'tma')
    df_tma_intrahora = df_limpo_tma_intrahora.groupby('ds')['tma'].mean().reset_index().rename(columns={'tma':'y'})

    # Fit dos modelos Prophet
    modelo_vol_diario = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_vol_diario.fit(df_limpo_vol_diario)
    modelo_tma_diario = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma_diario.fit(df_tma_diario)

    modelo_vol_intrahora = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_vol_intrahora.fit(df_limpo_vol_intrahora)
    modelo_tma_intrahora = Prophet(daily_seasonality=True, weekly_seasonality=True)
    modelo_tma_intrahora.fit(df_tma_intrahora)

    resultados = {}

    # Função para processar mês (base ou projeção)
    def processar_mes(mes):
        # Datas futuras para diário e intrahora
        dias_futuros_diario = pd.date_range(start=mes, end=mes + pd.offsets.MonthEnd(0))
        df_futuro_diario = pd.DataFrame({'ds': dias_futuros_diario})

        datas_futuras_intrahora = gerar_datas_futuras_intrahora(mes)
        df_futuro_intrahora = pd.DataFrame({'ds': datas_futuras_intrahora})

        # Previsões diárias
        pred_vol_diario = modelo_vol_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'y'})
        pred_vol_diario['y'] = pred_vol_diario['y'].apply(lambda x: max(1, x))
        pred_tma_diario = modelo_tma_diario.predict(df_futuro_diario)[['ds','yhat']].rename(columns={'yhat':'tma'})

        # Previsões intrahora
        pred_vol_intrahora = modelo_vol_intrahora.predict(df_futuro_intrahora)[['ds','yhat']].rename(columns={'yhat':'y'})
        pred_vol_intrahora['y'] = pred_vol_intrahora['y'].apply(lambda x: max(0.1, x))
        pred_tma_intrahora = modelo_tma_intrahora.predict(df_futuro_intrahora)[['ds','yhat']].rename(columns={'yhat':'tma'})

        # Mapear dias da semana
        mapa_dias = {'Monday':'Segunda-feira','Tuesday':'Terça-feira','Wednesday':'Quarta-feira',
                     'Thursday':'Quinta-feira','Friday':'Sexta-feira','Saturday':'Sábado','Sunday':'Domingo'}
        for df_p in [pred_vol_diario, pred_tma_diario]:
            df_p['dia_semana_pt'] = df_p['ds'].dt.day_name().map(mapa_dias)
        for df_p in [pred_vol_intrahora, pred_tma_intrahora]:
            df_p['dia_semana_pt'] = df_p['ds'].dt.day_name().map(mapa_dias)
            df_p['data'] = df_p['ds'].dt.date

        # Zerar dias desativados
        dias_desativados = [dia for dia, ativo in dias_func.items() if not ativo]
        pred_vol_diario.loc[pred_vol_diario['dia_semana_pt'].isin(dias_desativados), 'y'] = 0
        pred_tma_diario.loc[pred_tma_diario['dia_semana_pt'].isin(dias_desativados), 'tma'] = 0
        pred_vol_intrahora.loc[pred_vol_intrahora['dia_semana_pt'].isin(dias_desativados), 'y'] = 0
        pred_tma_intrahora.loc[pred_tma_intrahora['dia_semana_pt'].isin(dias_desativados), 'tma'] = 0

        # Ajuste volume intrahora proporcional
        totais_vol_diario = pred_vol_diario.set_index('ds')['y']
        soma_vol_intrahora = pred_vol_intrahora.groupby('data')['y'].transform('sum')
        fator_ajuste_vol = []
        for i, row in pred_vol_intrahora.iterrows():
            data = row['data']
            soma_intrahora = soma_vol_intrahora[i]
            total_diario = totais_vol_diario.get(pd.to_datetime(data), 0)
            fator = total_diario / soma_intrahora if soma_intrahora > 0 else 0
            fator_ajuste_vol.append(fator)
        pred_vol_intrahora['y'] *= fator_ajuste_vol

        # Ajuste TMA intrahora proporcional
        totais_tma_diario = pred_tma_diario.set_index('ds')['tma']
        media_tma_intrahora = pred_tma_intrahora.groupby('data')['tma'].transform('mean')
        fator_ajuste_tma = []
        for i, row in pred_tma_intrahora.iterrows():
            data = row['data']
            media_intrahora = media_tma_intrahora[i]
            total_diario = totais_tma_diario.get(pd.to_datetime(data), 0)
            fator = total_diario / media_intrahora if media_intrahora > 0 else 0
            fator_ajuste_tma.append(fator)
        pred_tma_intrahora['tma'] *= fator_ajuste_tma

        # Percentuais
        pred_vol_diario['percentual_volume'] = pred_vol_diario['y'] / pred_vol_diario['y'].sum() * 100 if pred_vol_diario['y'].sum() > 0 else 0
        media_tma_diario = pred_tma_diario['tma'].mean() if pred_tma_diario['tma'].mean() > 0 else 1
        pred_tma_diario['percentual_tma'] = pred_tma_diario['tma'] / media_tma_diario * 100

        pred_vol_intrahora['percentual_volume'] = pred_vol_intrahora.groupby('data')['y'].transform(lambda x: (x/x.sum()*100) if x.sum()>0 else 0)
        pred_tma_intrahora['percentual_tma'] = pred_tma_intrahora.groupby('data')['tma'].transform(lambda x: (x/x.mean()*100) if x.mean()>0 else 0)

        return {
            'vol_diario': pred_vol_diario,
            'tma_diario': pred_tma_diario,
            'vol_intrahora': pred_vol_intrahora,
            'tma_intrahora': pred_tma_intrahora
        }

    # Processar mês referência e projeções
    resultados[mes_ref.strftime('%Y-%m')] = rodar_previsoes(df_diario, df_intrahora, mes_ref, meses_projecoes, dias_funcionamento)
    for mes in meses_proj:
        if mes > mes_ref:
            resultados[mes.strftime('%Y-%m')] = rodar_previsoes(df_diario, df_intrahora, mes, meses_projecoes, dias_funcionamento)

    return resultados

if uploaded_file_diario and uploaded_file_intrahora:
    # Leitura arquivos
    df_diario = pd.read_excel(uploaded_file_diario) if uploaded_file_diario.name.endswith(('xlsx','xls')) else pd.read_csv(uploaded_file_diario)
    df_intrahora = pd.read_excel(uploaded_file_intrahora) if uploaded_file_intrahora.name.endswith(('xlsx','xls')) else pd.read_csv(uploaded_file_intrahora)

    resultados = rodar_previsoes(df_diario, df_intrahora, mes_referencia, meses_projecoes, dias_funcionamento)

    # === Interface para resultados ===
    st.title("📊 Projeções e Comparativos")

    aba_metricas = st.selectbox("Selecione métrica:", options=['Volume', 'TMA'])
    aba_curvas = st.radio("Visualizar curvas:", options=['Diária', 'Intrahora', 'Comparativo Diário + Intrahora'])

    # Seleção do mês para exibir
    mes_selecionado = st.selectbox("Mês para visualizar:", options=list(resultados.keys()))

    dados = resultados[mes_selecionado]

    if aba_metricas == 'Volume':
        if aba_curvas == 'Diária':
            st.subheader(f"Curva Diária de Volume - {mes_selecionado}")
            st.dataframe(dados['vol_diario'][['ds','y','percentual_volume']].rename(columns={
                'ds':'Data', 'y':'Volume', 'percentual_volume':'% Volume'
            }))
            grafico = alt.Chart(dados['vol_diario']).mark_line(point=True).encode(
                x='ds:T',
                y=alt.Y('y', title='Volume'),
                tooltip=['ds:T', alt.Tooltip('y', format=',.0f'), alt.Tooltip('percentual_volume', format='.2f')]
            ).properties(title=f"Volume Diário - {mes_selecionado}")
            st.altair_chart(grafico, use_container_width=True)

        elif aba_curvas == 'Intrahora':
            st.subheader(f"Curva Intrahora de Volume - {mes_selecionado}")
            st.dataframe(dados['vol_intrahora'][['ds','y','percentual_volume']].rename(columns={
                'ds':'Intervalo', 'y':'Volume Intrahora', 'percentual_volume':'% Volume Intrahora'
            }))
            grafico = alt.Chart(dados['vol_intrahora']).mark_line(point=False, strokeDash=[5,5], color='#9032bb').encode(
                x='ds:T',
                y=alt.Y('y', title='Volume Intrahora'),
                tooltip=['ds:T', alt.Tooltip('y', format=',.0f'), alt.Tooltip('percentual_volume', format='.2f')]
            ).properties(title=f"Volume Intrahora - {mes_selecionado}")
            st.altair_chart(grafico, use_container_width=True)

        else:  # Comparativo diário + intrahora
            st.subheader(f"Comparativo Diário + Intrahora - Volume - {mes_selecionado}")
            grafico_diario = alt.Chart(dados['vol_diario']).mark_line(point=True, color='#4b0081').encode(
                x='ds:T',
                y=alt.Y('y', title='Volume Diário'),
                tooltip=['ds:T', alt.Tooltip('y', format=',.0f'), alt.Tooltip('percentual_volume', format='.2f')]
            )
            grafico_intrahora = alt.Chart(dados['vol_intrahora']).mark_line(point=False, strokeDash=[5,5], color='#9032bb').encode(
                x='ds:T',
                y=alt.Y('y', title='Volume Intrahora'),
                tooltip=['ds:T', alt.Tooltip('y', format=',.0f'), alt.Tooltip('percentual_volume', format='.2f')]
            )
            st.altair_chart(grafico_diario + grafico_intrahora, use_container_width=True)

    else:  # TMA
        if aba_curvas == 'Diária':
            st.subheader(f"Curva Diária de TMA - {mes_selecionado}")
            st.dataframe(dados['tma_diario'][['ds','tma','percentual_tma']].rename(columns={
                'ds':'Data', 'tma':'TMA (s)', 'percentual_tma':'% TMA'
            }))
            grafico = alt.Chart(dados['tma_diario']).mark_line(point=True, color='#4b0081').encode(
                x='ds:T',
                y=alt.Y('tma', title='TMA (segundos)'),
                tooltip=['ds:T', alt.Tooltip('tma', format=',.2f'), alt.Tooltip('percentual_tma', format='.2f')]
            ).properties(title=f"TMA Diário - {mes_selecionado}")
            st.altair_chart(grafico, use_container_width=True)

        elif aba_curvas == 'Intrahora':
            st.subheader(f"Curva Intrahora de TMA - {mes_selecionado}")
            st.dataframe(dados['tma_intrahora'][['ds','tma','percentual_tma']].rename(columns={
                'ds':'Intervalo', 'tma':'TMA Intrahora (s)', 'percentual_tma':'% TMA Intrahora'
            }))
            grafico = alt.Chart(dados['tma_intrahora']).mark_line(point=False, strokeDash=[5,5], color='#9032bb').encode(
                x='ds:T',
                y=alt.Y('tma', title='TMA Intrahora (segundos)'),
                tooltip=['ds:T', alt.Tooltip('tma', format=',.2f'), alt.Tooltip('percentual_tma', format='.2f')]
            ).properties(title=f"TMA Intrahora - {mes_selecionado}")
            st.altair_chart(grafico, use_container_width=True)

        else:  # Comparativo diário + intrahora
            st.subheader(f"Comparativo Diário + Intrahora - TMA - {mes_selecionado}")
            grafico_diario = alt.Chart(dados['tma_diario']).mark_line(point=True, color='#4b0081').encode(
                x='ds:T',
                y=alt.Y('tma', title='TMA Diário (s)'),
                tooltip=['ds:T', alt.Tooltip('tma', format=',.2f'), alt.Tooltip('percentual_tma', format='.2f')]
            )
            grafico_intrahora = alt.Chart(dados['tma_intrahora']).mark_line(point=False, strokeDash=[5,5], color='#9032bb').encode(
                x='ds:T',
                y=alt.Y('tma', title='TMA Intrahora (s)'),
                tooltip=['ds:T', alt.Tooltip('tma', format=',.2f'), alt.Tooltip('percentual_tma', format='.2f')]
            )
            st.altair_chart(grafico_diario + grafico_intrahora, use_container_width=True)

else:
    st.warning("Por favor, faça o upload das bases diária e intrahora para prosseguir.")
