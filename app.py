import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

# Configurar página
st.set_page_config(page_title="SERCOM Digitais - Projeção de Ligações", layout="wide")

# Opção Dark Mode
dark_mode = st.sidebar.checkbox("🌓 Ativar Dark Mode", value=False)

# CSS atualizado para dark mode + labels e inputs
if dark_mode:
    css_style = """
    <style>
    .block-container {
        padding-top: 2rem;
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    label, .stMarkdown, .stTextInput>div>input, .stSelectbox label, .stMultiselect label,
    .stTextArea label, .stDateInput label, .stFileUploader label {
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
    ::placeholder {
        color: #aaaaaa !important;
    }
    </style>
    """
else:
    css_style = """
    <style>
    .block-container {
        padding-top: 2rem;
        background-color: white;
        color: black;
    }
    .stApp {
        background-color: white;
        color: black;
    }
    label, .stMarkdown {
        color: black !important;
    }
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

# Logo
logo_url = "https://raw.githubusercontent.com/AlissuFS/previsao-ligacoes/main/Logotipo%20Sercom%20Digital%20br%20_png_edited_p.avif"
st.markdown(f"""
    <div style="background-color:#002f6c; padding:12px 24px; display:flex; align-items:center; border-bottom: 3px solid #0059b3;">
        <img src="{logo_url}" style="height:42px; margin-right:20px;" alt="Logo SERCOM">
        <h1 style="color:#ffffff; font-size:1.6rem; margin:0;">SERCOM Digitais - Projeção de Ligações</h1>
    </div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📁 Envie a planilha com 'Data' e 'Quantidade de Ligações'", type=[".xlsx", ".xls", ".csv"])

# Dias da semana
dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.multiselect("📌 Selecione os dias da semana a considerar", dias_semana_port, default=dias_semana_port)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        if 'Data' not in df.columns or 'Quantidade de Ligações' not in df.columns:
            st.error("A planilha precisa conter as colunas 'Data' e 'Quantidade de Ligações'.")
            st.stop()

        df['ds'] = pd.to_datetime(df['Data'])
        df['y'] = df['Quantidade de Ligações'].clip(lower=0)
        df['ano_mes'] = df['ds'].dt.to_period('M')
        df['dia_semana'] = df['ds'].dt.day_name()
        df['dia_semana_pt'] = df['dia_semana'].map({
            'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
            'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        })

        # Seleção de meses
        meses_disponiveis = sorted(df['ano_mes'].unique(), reverse=True)
        mes_map = {str(m): m for m in meses_disponiveis}
        col1, col2 = st.columns(2)
        with col1:
            mes_base_str = st.selectbox("📅 Mês base (Histórico)", list(mes_map.keys()), index=0)
        with col2:
            mes_proj_str = st.text_input("🔮 Mês projetado (AAAA-MM)", value=str((datetime.now() + timedelta(days=30)).strftime('%Y-%m')))
        mes_base = mes_map[mes_base_str]
        mes_proj = pd.Period(mes_proj_str, freq='M')

        # Função de cálculo da curva
        def calcular_curva(df_mes, dias_filtrados, sufixo=""):
            df_mes = df_mes[df_mes['dia_semana_pt'].isin(dias_filtrados)].copy()
            if df_mes.empty:
                return pd.Series(dtype=float)

            # Função para calcular a ocorrência de uma semana no mês
            def ocorrencia_semana(data):
                dia = data.day
                dia_semana = data.weekday()  # 0 = segunda-feira, 6 = domingo
                return sum((datetime(data.year, data.month, d).weekday() == dia_semana) for d in range(1, dia + 1))

            # Adiciona uma coluna 'ordem' que indica a ordem do dia da semana dentro do mês
            df_mes['ordem'] = df_mes['ds'].apply(ocorrencia_semana)

            # Definindo rótulos para os dias da semana
            ordinais = {1: '1ª', 2: '2ª', 3: '3ª', 4: '4ª', 5: '5ª'}
            df_mes['rotulo'] = df_mes.apply(lambda row: f"{ordinais.get(row['ordem'], str(row['ordem']) + 'ª')} {row['dia_semana_pt']}", axis=1)

            # Agrupamento por rótulo e somatório das ligações
            grupo = df_mes.groupby('rotulo')['y'].sum()

            # Garantir que todos os dias da semana apareçam, incluindo os domingos (mesmo com zero)
            dias_completos = dias_semana_port  # A lista completa de dias da semana
            for dia in dias_completos:
                if dia not in grupo.index:
                    grupo[dia] = 0

            grupo_total = grupo.sum()
            percentual = grupo / grupo_total * 100
            percentual.name = f"Percentual{sufixo}"

            return percentual

        curva_base = calcular_curva(df[df['ano_mes'] == mes_base], dias_selecionados, sufixo=" (Histórico)")
        df_proj = df[df['ano_mes'] == mes_proj]

        if df_proj.empty:
            st.info("📈 Gerando previsão com IA para o mês projetado...")
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
            df_prev['y'] = df_prev['y'].clip(lower=0)
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

        # Gráfico comparativo
        df_temp = curva_comparativa.reset_index()
        df_temp.rename(columns={df_temp.columns[0]: 'Categoria'}, inplace=True)
        df_plot = df_temp.melt(id_vars='Categoria', var_name='Tipo', value_name='Percentual')

        cor_azul_escuro = '#90caf9' if dark_mode else '#002f6c'
        cor_azul_claro = '#bbdefb' if dark_mode else '#0059b3'
        fundo_grafico = '#121212' if dark_mode else 'white'

        chart_comp = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X('Categoria:N', title='Ordem e Dia da Semana', sort=None),
            y=alt.Y('Percentual:Q', title='Percentual (%)'),
            color=alt.Color('Tipo:N', scale=alt.Scale(domain=list(df_plot['Tipo'].unique()), range=[cor_azul_escuro, cor_azul_claro])),
            tooltip=['Categoria', 'Tipo', alt.Tooltip('Percentual', format='.2f')]
        ).properties(width=800, height=350, background=fundo_grafico).interactive()

        st.subheader("📈 Evolução em Gráfico de Linha")
        st.altair_chart(chart_comp, use_container_width=True)

        # Exportação Excel
        st.subheader("📥 Exportar Resultado")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            curva_comparativa.reset_index().to_excel(writer, index=False, sheet_name="Comparativo")
        st.download_button("📄 Baixar Excel", data=buffer.getvalue(), file_name="comparativo_projecao_ligacoes_SERCOM.xlsx")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
