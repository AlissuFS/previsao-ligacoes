import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

# Configurar página
st.set_page_config(page_title="SERCOM Digitais - Projeção de Ligações", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif", use_column_width=True)
st.sidebar.markdown("### 🔍 Configurações")

# Dark Mode
dark_mode = st.sidebar.checkbox("🌙 Modo Escuro", value=False)

# CSS personalizado
if dark_mode:
    css_style = """
    <style>
    .block-container { background-color: #121212; color: #e0e0e0; }
    .stApp { background-color: #121212; color: #e0e0e0; }
    label, .stMarkdown, .stTextInput>div>input, .stSelectbox label, .stMultiselect label, .stTextArea label, .stDateInput label, .stFileUploader label {
        color: #e0e0e0 !important;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb], .stMultiselect div[data-baseweb] {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
        border-color: #444 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #2196f3 !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    ::placeholder { color: #aaaaaa !important; }
    </style>
    """
else:
    css_style = """
    <style>
    .block-container { background-color: white; color: black; }
    .stApp { background-color: white; color: black; }
    label, .stMarkdown { color: black !important; }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb], .stMultiselect div[data-baseweb] {
        background-color: white !important;
        color: black !important;
        border-color: #ccc !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #002f6c !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    </style>
    """
st.markdown(css_style, unsafe_allow_html=True)

# Entrada de dados
st.sidebar.markdown("### 📁 Upload da Planilha")
uploaded_file = st.sidebar.file_uploader("Envie arquivo com colunas 'Data' e 'Quantidade de Ligações'", type=[".xlsx", ".xls", ".csv"])

# Dias da semana
dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.sidebar.multiselect("📍 Dias da semana considerados", dias_semana_port, default=dias_semana_port)

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    if 'Data' not in df.columns or 'Quantidade de Ligações' not in df.columns:
        st.error("A planilha precisa conter as colunas 'Data' e 'Quantidade de Ligações'.")
        st.stop()

    df['ds'] = pd.to_datetime(df['Data'])
    df['y'] = df['Quantidade de Ligações'].clip(lower=0)
    df['ano_mes'] = df['ds'].dt.to_period('M')
    df['dia_semana'] = df['ds'].dt.day_name()
    mapa_dias = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['dia_semana_pt'] = df['dia_semana'].map(mapa_dias).fillna(df['dia_semana'])

    # Sidebar: Mês base e meses projetados
    meses_disponiveis = sorted(df['ano_mes'].unique(), reverse=True)
    mes_map = {str(m): m for m in meses_disponiveis}
    mes_base_str = st.sidebar.selectbox("📅 Mês base (histórico)", list(mes_map.keys()), index=0)
    meses_proj_str = st.sidebar.multiselect("🌟 Meses projetados (AAAA-MM)", [])

    mes_base = mes_map[mes_base_str]
    meses_proj = [pd.Period(m, freq='M') for m in meses_proj_str if m]

    def ocorrencia_semana(data):
        dia_semana = data.weekday()
        dias_mes = pd.date_range(start=data.replace(day=1), end=data)
        return sum(d.weekday() == dia_semana for d in dias_mes)

    def calcular_curva(df_mes, dias_filtrados, sufixo=""):
        df_mes = df_mes[df_mes['dia_semana_pt'].isin(dias_filtrados)].copy()
        if df_mes.empty:
            return pd.Series(dtype=float)
        df_mes['ordem'] = df_mes['ds'].apply(ocorrencia_semana)
        ordinais = {1: '1ª', 2: '2ª', 3: '3ª', 4: '4ª', 5: '5ª'}
        df_mes['rotulo'] = df_mes.apply(lambda row: f"{ordinais.get(row['ordem'], str(row['ordem']) + 'ª')} {row['dia_semana_pt']}", axis=1)
        grupo = df_mes.groupby('rotulo')['y'].sum()
        grupo_total = grupo.sum()
        percentual = grupo / grupo_total * 100
        percentual.name = f"Percentual{sufixo}"
        return percentual

    curva_base = calcular_curva(df[df['ano_mes'] == mes_base], dias_selecionados, sufixo=" (Histórico)")

    tabs = st.tabs(["📊 Comparativos", "📅 Curvas Diárias", "📥 Exportação"])

    resultados = {}

    for mes_proj in meses_proj:
        df_proj = df[df['ano_mes'] == mes_proj]
        if df_proj.empty:
            Q1, Q3 = df['y'].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df_limpo = df[(df['y'] >= Q1 - 1.5 * IQR) & (df['y'] <= Q3 + 1.5 * IQR)][['ds', 'y']].sort_values('ds')
            modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            modelo.fit(df_limpo)
            inicio, fim = mes_proj.to_timestamp(), (mes_proj + 1).to_timestamp() - timedelta(days=1)
            futuro = pd.date_range(start=inicio, end=fim, freq='D')
            df_futuro = pd.DataFrame({'ds': futuro})
            previsao = modelo.predict(df_futuro)
            df_prev = previsao[['ds', 'yhat']].rename(columns={'yhat': 'y'})
            df_prev['y'] = df_prev['y'].clip(lower=0)
            df_prev['dia_semana'] = df_prev['ds'].dt.day_name()
            df_prev['dia_semana_pt'] = df_prev['dia_semana'].map(mapa_dias).fillna(df_prev['dia_semana'])
            dados_proj = df_prev
        else:
            dados_proj = df_proj

        curva_proj = calcular_curva(dados_proj, dias_selecionados, sufixo=f" ({mes_proj})")
        comparativo = pd.concat([curva_base, curva_proj], axis=1).fillna(0)

        resultados[str(mes_proj)] = {
            "dados": dados_proj,
            "curva": curva_proj,
            "comparativo": comparativo
        }

    with tabs[0]:
        for mes_str, dados in resultados.items():
            st.subheader(f"📊 Comparativo: {mes_base.strftime('%m/%Y')} vs {mes_str}")
            curva_fmt = dados['comparativo'].copy()
            curva_fmt['Histórico (%)'] = curva_fmt.iloc[:, 0].apply(lambda x: f"{x:.2f}%" if x > 0 else "0%")
            curva_fmt[f'{mes_str} (%)'] = curva_fmt.iloc[:, 1].apply(lambda x: f"{x:.2f}%" if x > 0 else "0%")
            st.dataframe(curva_fmt[["Histórico (%)", f"{mes_str} (%)"]], use_container_width=True)

    with tabs[1]:
        for mes_str, dados in resultados.items():
            st.subheader(f"📅 Curva Diária da Projeção: {mes_str}")
            df_mes_proj = dados['dados']
            total_mes = df_mes_proj['y'].sum()
            if total_mes > 0:
                df_dia = df_mes_proj[['ds', 'y']].copy()
                df_dia['percentual'] = df_dia['y'] / total_mes * 100
                chart_dia = alt.Chart(df_dia).mark_line(point=True).encode(
                    x=alt.X('ds:T', title='Data'),
                    y=alt.Y('percentual:Q', title='Percentual Diário (%)'),
                    tooltip=[alt.Tooltip('ds:T', title='Data'), alt.Tooltip('percentual:Q', format='.2f')]
                ).properties(width=800, height=350).interactive()
                st.altair_chart(chart_dia, use_container_width=True)

    with tabs[2]:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            for mes_str, dados in resultados.items():
                comp = dados['comparativo'].reset_index()
                comp.to_excel(writer, index=False, sheet_name=f"Comparativo_{mes_str}")
                df_export = dados['dados'][['ds', 'y']].copy()
                total_proj = df_export['y'].sum()
                df_export['Percentual (%)'] = df_export['y'] / total_proj * 100
                df_export['Percentual (%)'] = df_export['Percentual (%)'].round(2)
                df_export.columns = ['Data', 'Quantidade', 'Percentual (%)']
                df_export.to_excel(writer, index=False, sheet_name=f"Curva_{mes_str}")
        st.download_button("📄 Baixar Excel", data=buffer.getvalue(), file_name="projecoes_SERCOM.xlsx")
