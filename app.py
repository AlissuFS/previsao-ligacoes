import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

st.set_page_config(page_title="SERCOM Digitais - Projeção", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.image(
    "https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif",
    use_container_width=True
)
st.sidebar.markdown("### 🔍 Configurações")

# Dark Mode
dark_mode = st.sidebar.checkbox("🌙 Modo Escuro", value=False)

# Tema
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

# Upload
st.sidebar.markdown("### 📁 Upload da Planilha")
uploaded_file = st.sidebar.file_uploader("Envie arquivo com colunas 'Data', 'Quantidade de Ligações' e 'TMA'", type=[".xlsx", ".xls", ".csv"])

dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.sidebar.multiselect("📍 Dias da semana considerados", dias_semana_port, default=dias_semana_port)

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
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

    # Base e projeção
    mes_hoje = pd.Period(datetime.now(), freq='M')
    meses_proximos = [mes_hoje + i for i in range(4)]

    meses_disponiveis = sorted(df['ano_mes'].unique())
    meses_disponiveis_str = [str(m) for m in meses_disponiveis]
    mes_base = st.sidebar.selectbox("📅 Mês base (histórico)", options=meses_disponiveis_str, index=len(meses_disponiveis_str)-1)
    mes_base = pd.Period(mes_base, freq='M')

    meses_para_projetar = [m for m in meses_proximos if m not in meses_disponiveis]
    meses_proj_str = st.sidebar.multiselect("🌟 Meses projetados (futuros até +3 meses)", options=[str(m) for m in meses_para_projetar], default=[str(m) for m in meses_para_projetar])
    meses_proj = [pd.Period(m, freq='M') for m in meses_proj_str]

    # Seleção entre Volume e TMA
    tipo_curva = st.radio("📈 Tipo de Curva", ["Volume", "TMA"], horizontal=True)

    def ocorrencia_semana(data):
        dia_semana = data.weekday()
        dias_mes = pd.date_range(start=data.replace(day=1), end=data)
        return sum(d.weekday() == dia_semana for d in dias_mes)

    def calcular_curva(df_mes, dias_filtrados, coluna_valor, sufixo=""):
        df_mes = df_mes[df_mes['dia_semana_pt'].isin(dias_filtrados)].copy()
        if df_mes.empty:
            return pd.Series(dtype=float)
        df_mes['ordem'] = df_mes['ds'].apply(ocorrencia_semana)
        ordinais = {1: '1ª', 2: '2ª', 3: '3ª', 4: '4ª', 5: '5ª'}
        df_mes['rotulo'] = df_mes.apply(lambda row: f"{ordinais.get(row['ordem'], str(row['ordem']) + 'ª')} {row['dia_semana_pt']}", axis=1)

        max_ordem = df_mes['ordem'].max()
        rotulos_esperados = [f"{ordinais.get(i)} {dia}" for dia in dias_filtrados for i in range(1, max_ordem + 1)]

        grupo = df_mes.groupby('rotulo')[coluna_valor].sum()
        grupo = grupo.reindex(rotulos_esperados, fill_value=0)

        if coluna_valor == "tma":
            media_mes = grupo.mean() if grupo.mean() > 0 else 1
            percentual = grupo / media_mes * 100
        else:
            grupo_total = grupo.sum()
            percentual = grupo / grupo_total * 100 if grupo_total > 0 else grupo

        percentual.name = f"Percentual{sufixo}"
        return percentual

    coluna_analise = 'tma' if tipo_curva == "TMA" else 'y'
    curva_base = calcular_curva(df[df['ano_mes'] == mes_base], dias_selecionados, coluna_analise, sufixo=" (Histórico)")

    tabs = st.tabs(["📊 Comparativos", "📅 Curvas Diárias", "📥 Exportação"])
    resultados = {}

    for mes_proj in meses_proj:
        df_proj = df[df['ano_mes'] == mes_proj]
        if df_proj.empty:
            df_limpo = df.copy()
            Q1, Q3 = df['y'].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df_limpo = df[(df['y'] >= Q1 - 1.5 * IQR) & (df['y'] <= Q3 + 1.5 * IQR)][['ds', 'y']].sort_values('ds')
            modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            modelo.fit(df_limpo)
            futuro = pd.date_range(start=mes_proj.to_timestamp(), end=(mes_proj + 1).to_timestamp() - timedelta(days=1))
            df_futuro = pd.DataFrame({'ds': futuro})
            previsao = modelo.predict(df_futuro)
            df_prev = previsao[['ds', 'yhat']].rename(columns={'yhat': 'y'})
            df_prev['y'] = df_prev['y'].clip(lower=0)
            df_prev['tma'] = df_prev['y'] * 0.0  # Sem histórico de TMA previsto
            df_prev['dia_semana'] = df_prev['ds'].dt.day_name()
            df_prev['dia_semana_pt'] = df_prev['dia_semana'].map(mapa_dias)
            dados_proj = df_prev
        else:
            dados_proj = df_proj

        curva_proj = calcular_curva(dados_proj, dias_selecionados, coluna_analise, sufixo=f" ({mes_proj})")
        comparativo = pd.concat([curva_base, curva_proj], axis=1).fillna(0)

        resultados[str(mes_proj)] = {
            "dados": dados_proj,
            "curva": curva_proj,
            "comparativo": comparativo
        }

    with tabs[0]:
        for mes_str, dados in resultados.items():
            st.subheader(f"📊 Comparativo: {mes_base.strftime('%m/%Y')} vs {mes_str} — {tipo_curva}")
            curva_fmt = dados['comparativo'].copy()
            curva_fmt['Histórico (%)'] = curva_fmt.iloc[:, 0].apply(lambda x: f"{x:.2f}%" if x > 0 else "0%")
            curva_fmt[f'{mes_str} (%)'] = curva_fmt.iloc[:, 1].apply(lambda x: f"{x:.2f}%" if x > 0 else "0%")
            st.dataframe(curva_fmt[["Histórico (%)", f"{mes_str} (%)"]], use_container_width=True)

    with tabs[1]:
        for mes_str, dados in resultados.items():
            st.subheader(f"📅 Curva Diária da Projeção: {mes_str} — {tipo_curva}")
            df_mes_proj = dados['dados']
            total_mes = df_mes_proj[coluna_analise].sum()
            if total_mes > 0:
                df_dia = df_mes_proj[['ds', coluna_analise]].copy()
                if tipo_curva == "TMA":
                    media_tma = df_dia[coluna_analise].mean()
                    df_dia['percentual'] = df_dia[coluna_analise] / media_tma * 100
                    tooltip = [
                        alt.Tooltip('ds:T', title='Data'),
                        alt.Tooltip(f'{coluna_analise}:Q', title='TMA (s)', format='.0f'),
                        alt.Tooltip('percentual:Q', title='% sobre média', format='.2f')
                    ]
                else:
                    df_dia['percentual'] = df_dia[coluna_analise] / total_mes * 100
                    tooltip = [
                        alt.Tooltip('ds:T', title='Data'),
                        alt.Tooltip(f'{coluna_analise}:Q', title='Ligações', format='.0f'),
                        alt.Tooltip('percentual:Q', title='Percentual Diário (%)', format='.2f')
                    ]

                chart_dia = alt.Chart(df_dia).mark_line(point=True).encode(
                    x=alt.X('ds:T', title='Data'),
                    y=alt.Y('percentual:Q', title='Percentual (%)'),
                    tooltip=tooltip
                ).properties(width=800, height=350).interactive()
                st.altair_chart(chart_dia, use_container_width=True)

    with tabs[2]:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            for mes_str, dados in resultados.items():
                comp = dados['comparativo'].reset_index()
                comp.to_excel(writer, sheet_name=f"Comparativo_{mes_str}", index=False)

        st.download_button(
            label="📥 Baixar resultados em Excel",
            data=buffer.getvalue(),
            file_name="resultados_previsao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
