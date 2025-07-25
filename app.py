import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

st.set_page_config(page_title="SERCOM Digitais - Projeção de Ligações", layout="wide", initial_sidebar_state="expanded")

st.sidebar.image(
    "https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif",
    use_container_width=True
)
st.sidebar.markdown("### 🔍 Configurações")

dark_mode = st.sidebar.checkbox("🌙 Modo Escuro", value=False)

css_style = """
<style>
.block-container {background-color: %s; color: %s;}
.stApp {background-color: %s; color: %s;}
label, .stMarkdown, .stTextInput>div>input, .stSelectbox label, .stMultiselect label, .stTextArea label, .stDateInput label, .stFileUploader label {
    color: %s !important;
}
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb], .stMultiselect div[data-baseweb] {
    background-color: %s !important;
    color: %s !important;
    border-color: %s !important;
}
.stButton>button, .stDownloadButton>button {
    background-color: #6600cc !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] {
    background-color: %s !important;
    color: %s !important;
}
[data-testid="stSidebar"] * {
    color: %s !important;
}
</style>
""" % (
    "#121212" if dark_mode else "white",
    "#e0e0e0" if dark_mode else "black",
    "#121212" if dark_mode else "white",
    "#e0e0e0" if dark_mode else "black",
    "#e0e0e0" if dark_mode else "black",
    "#1e1e1e" if dark_mode else "white",
    "#e0e0e0" if dark_mode else "black",
    "#444" if dark_mode else "#ccc",
    "#121212" if dark_mode else "white",
    "#e0e0e0" if dark_mode else "black",
    "#e0e0e0" if dark_mode else "black"
)
st.markdown(css_style, unsafe_allow_html=True)

st.sidebar.markdown("### 📁 Upload da Planilha")
uploaded_file = st.sidebar.file_uploader("Envie arquivo com colunas 'Data', 'Chamadas Recebidas' e 'TMA'", type=[".xlsx", ".xls", ".csv"])

dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.sidebar.multiselect("📍 Dias da semana considerados", dias_semana_port, default=dias_semana_port)

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    required_cols = {'Data', 'Chamadas Recebidas', 'TMA'}
    if not required_cols.issubset(set(df.columns)):
        st.error("A planilha precisa conter as colunas: 'Data', 'Chamadas Recebidas' e 'TMA'.")
        st.stop()

    df['ds'] = pd.to_datetime(df['Data'])
    df['volume'] = df['Chamadas Recebidas'].clip(lower=0)
    df['tma'] = df['TMA'].clip(lower=0)
    df['ano_mes'] = df['ds'].dt.to_period('M')
    df['dia_semana'] = df['ds'].dt.day_name()
    mapa_dias = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['dia_semana_pt'] = df['dia_semana'].map(mapa_dias).fillna(df['dia_semana'])

    tipo_curva = st.radio("Selecionar Curva:", ["Volume", "TMA"], horizontal=True)

    mes_hoje = pd.Period(datetime.now(), freq='M')
    meses_proximos = [mes_hoje + i for i in range(4)]
    meses_disponiveis = sorted(df['ano_mes'].unique())
    meses_disponiveis_str = [str(m) for m in meses_disponiveis]
    ultimo_mes_hist = meses_disponiveis[-1]

    mes_base = st.sidebar.selectbox("\U0001f4c5 Mês base (histórico)", options=meses_disponiveis_str, index=meses_disponiveis_str.index(str(ultimo_mes_hist)))
    mes_base = pd.Period(mes_base, freq='M')
    meses_para_projetar = [m for m in meses_proximos if m not in meses_disponiveis]

    meses_proj_str = st.sidebar.multiselect("\U0001f31f Meses projetados", options=[str(m) for m in meses_para_projetar], default=[str(m) for m in meses_para_projetar])
    meses_proj = [pd.Period(m, freq='M') for m in meses_proj_str]

    def calcular_curva(df_mes, dias_filtrados, tipo, sufixo=""):
        df_mes = df_mes[df_mes['dia_semana_pt'].isin(dias_filtrados)].copy()
        if df_mes.empty:
            return pd.Series(dtype=float)
        df_mes['ordem'] = df_mes['ds'].dt.day
        ordinais = {1: '1ª', 2: '2ª', 3: '3ª', 4: '4ª', 5: '5ª'}
        df_mes['rotulo'] = df_mes.apply(lambda row: f"{ordinais.get((row['ds'].day - 1) // 7 + 1, '')} {row['dia_semana_pt']}", axis=1)

        if tipo == "Volume":
            grupo = df_mes.groupby('rotulo')['volume'].sum()
            total = grupo.sum()
            percentual = grupo / total * 100 if total > 0 else grupo
            percentual.name = f"Percentual{sufixo}"
            return percentual

        elif tipo == "TMA":
            grupo = df_mes.groupby('rotulo').apply(lambda x: np.average(x['tma'], weights=x['volume']) if x['volume'].sum() > 0 else 0)
            grupo.name = f"TMA{sufixo}"
            return grupo

    curva_base = calcular_curva(df[df['ano_mes'] == mes_base], dias_selecionados, tipo_curva, sufixo=" (Histórico)")

    tabs = st.tabs(["\U0001f4ca Comparativos", "\U0001f4c5 Curvas Diárias", "\U0001f4e5 Exportação"])

    resultados = {}

    for mes_proj in meses_proj:
        df_proj = df[df['ano_mes'] == mes_proj]
        if df_proj.empty:
            continue  # Projeção futura com Prophet para volume, ainda não implementado para TMA

        curva_proj = calcular_curva(df_proj, dias_selecionados, tipo_curva, sufixo=f" ({mes_proj})")
        comparativo = pd.concat([curva_base, curva_proj], axis=1).fillna(0)
        resultados[str(mes_proj)] = {
            "dados": df_proj,
            "curva": curva_proj,
            "comparativo": comparativo
        }

    with tabs[0]:
        for mes_str, dados in resultados.items():
            st.subheader(f"\U0001f4ca Comparativo: {mes_base.strftime('%m/%Y')} vs {mes_str} - {tipo_curva}")
            curva_fmt = dados['comparativo'].copy()
            if tipo_curva == "Volume":
                curva_fmt['Histórico (%)'] = curva_fmt.iloc[:, 0].apply(lambda x: f"{x:.2f}%")
                curva_fmt[f'{mes_str} (%)'] = curva_fmt.iloc[:, 1].apply(lambda x: f"{x:.2f}%")
            else:
                curva_fmt['Histórico (min)'] = curva_fmt.iloc[:, 0].apply(lambda x: f"{x:.2f} min")
                curva_fmt[f'{mes_str} (min)'] = curva_fmt.iloc[:, 1].apply(lambda x: f"{x:.2f} min")
            st.dataframe(curva_fmt.iloc[:, -2:], use_container_width=True)

    with tabs[1]:
        for mes_str, dados in resultados.items():
            st.subheader(f"\U0001f4c5 Curva Diária: {mes_str} - {tipo_curva}")
            df_mes_proj = dados['dados']
            df_dia = df_mes_proj[['ds', 'volume', 'tma', 'dia_semana_pt']].copy()
            if tipo_curva == "Volume":
                total = df_dia['volume'].sum()
                df_dia['valor'] = df_dia['volume'] / total * 100 if total > 0 else 0
                y_title = 'Percentual Diário (%)'
                tooltip_val = alt.Tooltip('valor:Q', format='.2f', title='Percentual')
            else:
                df_dia = df_dia[df_dia['volume'] > 0]
                df_dia['valor'] = df_dia['tma']
                y_title = 'TMA (min)'
                tooltip_val = alt.Tooltip('valor:Q', format='.2f', title='TMA')

            chart = alt.Chart(df_dia).mark_line(point=True).encode(
                x=alt.X('ds:T', title='Data'),
                y=alt.Y('valor:Q', title=y_title),
                tooltip=[
                    alt.Tooltip('ds:T', title='Data'),
                    alt.Tooltip('dia_semana_pt:N', title='Dia da Semana'),
                    alt.Tooltip('volume:Q', title='Chamadas'),
                    tooltip_val
                ]
            ).properties(width=800, height=350).interactive()
            st.altair_chart(chart, use_container_width=True)

    with tabs[2]:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            for mes_str, dados in resultados.items():
                comp = dados['comparativo'].reset_index()
                comp.to_excel(writer, sheet_name=f"{tipo_curva}_{mes_str}", index=False)

        st.download_button(
            label="\U0001f4e5 Baixar resultados em Excel",
            data=buffer.getvalue(),
            file_name=f"resultados_{tipo_curva.lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
