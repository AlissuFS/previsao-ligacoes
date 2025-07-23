import pandas as pd
import numpy as np
import streamlit as st
import io
from datetime import datetime, timedelta
from prophet import Prophet
import altair as alt

# Configurar a página com identidade visual
st.set_page_config(page_title="SERCOM Digitais - Projeção de Ligações", layout="wide")

# CSS customizado para visual mais sofisticado e parecido com o site SERCOM
st.markdown("""
<style>
/* Corpo da página */
.main {
    background-color: #f5f6f8 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #2c3e50;
}

/* Container principal */
.block-container {
    background-color: #fff;
    border-radius: 12px;
    padding: 3rem 3rem 4rem 3rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Header escondido do Streamlit padrão */
header, footer {
    visibility: hidden;
}

/* Cabeçalho personalizado fixo */
.custom-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 72px;
    background-color: #002f6c;
    display: flex;
    align-items: center;
    padding: 0 2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    z-index: 1000;
}

.custom-header img {
    height: 48px;
    margin-right: 1rem;
}

.custom-header h1 {
    color: #fff;
    font-weight: 700;
    font-size: 1.5rem;
    margin: 0;
    user-select: none;
}

/* Botões estilo SERCOM */
.stButton>button, .stDownloadButton>button {
    background-color: #002f6c !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.4rem !important;
    transition: background-color 0.3s ease;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #0059b3 !important;
    cursor: pointer;
}

/* Multiselect box */
.stMultiSelect > div > div:first-child {
    border-radius: 6px !important;
    border: 1px solid #ccc !important;
    padding: 0.4rem 0.6rem !important;
}

/* Input text */
.stTextInput > div > input {
    border-radius: 6px !important;
    border: 1px solid #ccc !important;
    padding: 0.45rem 0.7rem !important;
    font-size: 1rem !important;
}

/* Subheaders */
h2, h3, h4 {
    color: #002f6c;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

/* Tabela com linhas melhoradas */
[data-testid="stDataFrame"] table {
    border-collapse: separate !important;
    border-spacing: 0 10px !important;
}

[data-testid="stDataFrame"] thead tr th {
    background-color: #e1eaf7 !important;
    color: #002f6c !important;
    font-weight: 700 !important;
    padding: 10px 12px !important;
}

[data-testid="stDataFrame"] tbody tr td {
    background-color: #f8faff !important;
    padding: 10px 12px !important;
    border-bottom: none !important;
}

/* Espaço entre gráficos e tabelas */
[data-testid="stLineChart"] {
    margin-top: 1rem;
    margin-bottom: 2rem;
}

/* Ajuste para evitar que conteúdo fique embaixo do header fixo */
.block-container {
    padding-top: 100px !important;
}

/* Scrollbar personalizada para as tabelas */
[data-testid="stDataFrame"] table::-webkit-scrollbar {
    height: 8px;
}

[data-testid="stDataFrame"] table::-webkit-scrollbar-thumb {
    background-color: #002f6c;
    border-radius: 4px;
}

/* Responsividade */
@media (max-width: 768px) {
    .custom-header {
        padding: 0 1rem;
    }
    .block-container {
        padding: 100px 1rem 2rem 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Cabeçalho fixo customizado
st.markdown("""
<div class="custom-header">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Logo-sercom.png/320px-Logo-sercom.png" alt="Logo SERCOM">
    <h1>SERCOM Digitais - Projeção de Ligações</h1>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Envie a planilha com 'Data' e 'Quantidade de Ligações'", type=[".xlsx", ".xls", ".csv"])

# Dias da semana selecionáveis
dias_semana_port = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
dias_selecionados = st.multiselect("📌 Selecione os dias da semana a considerar", dias_semana_port, default=dias_semana_port)

if uploaded_file:
    try:
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
            mes_base_str = st.selectbox("📅 Mês base (Histórico)", list(mes_map.keys()), index=0)
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
                return sum((datetime(data.year, data.month, d).weekday() == dia_semana) for d in range(1, dia + 1))

            df_mes['ordem'] = df_mes['ds'].apply(ocorrencia_semana)
            ordinais = {1: '1ª', 2: '2ª', 3: '3ª', 4: '4ª', 5: '5ª'}
            df_mes['rotulo'] = df_mes.apply(lambda row: f"{ordinais.get(row['ordem'], str(row['ordem']) + 'ª')} {row['dia_semana_pt']}", axis=1)
            df_mes['y'] = df_mes['y'].clip(lower=0)
            grupo = df_mes.groupby('rotulo')['y'].sum()
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

        # --- Gráfico comparativo com Altair ---
        df_plot = curva_comparativa.reset_index().melt(id_vars='index', var_name='Tipo', value_name='Percentual')
        df_plot.rename(columns={'index': 'Categoria'}, inplace=True)

        cor_azul_escuro = '#002f6c'
        cor_azul_claro = '#0059b3'

        color_scale = alt.Scale(domain=[curva_comparativa.columns[0], curva_comparativa.columns[1]],
                                range=[cor_azul_escuro, cor_azul_claro])

        chart_comp = alt.Chart(df_plot).mark_line(point=True).encode(
            x=alt.X('Categoria:N', title='Ordem e Dia da Semana', sort=None),
            y=alt.Y('Percentual:Q', title='Percentual (%)'),
            color=alt.Color('Tipo:N', scale=color_scale, legend=alt.Legend(title="Legenda")),
            tooltip=['Categoria', 'Tipo', alt.Tooltip('Percentual', format='.2f')]
        ).properties(width=800, height=350).interactive()

        st.subheader("📈 Evolução em Gráfico de Linha")
        st.altair_chart(chart_comp, use_container_width=True)

        # --- Gráfico diário da projeção ---
        df_curva_dia = None
        if not df_proj.empty:
            df_mes_proj = df_proj.copy()
        else:
            df_mes_proj = df_prev.copy() if 'df_prev' in locals() else None

        if df_mes_proj is not None and not df_mes_proj.empty:
            total_mes = df_mes_proj['y'].sum()
            if total_mes > 0:
                df_curva_dia = df_mes_proj[['ds', 'y']].copy()
                df_curva_dia['percentual'] = df_curva_dia['y'] / total_mes * 100
                df_curva_dia = df_curva_dia.set_index('ds').sort_index()

        if df_curva_dia is not None and not df_curva_dia.empty:
            df_dia_plot = df_curva_dia.reset_index()
            chart_dia = alt.Chart(df_dia_plot).mark_line(point=True, color=cor_azul_escuro).encode(
                x=alt.X('ds:T', title='Data'),
                y=alt.Y('percentual:Q', title='Percentual Diário (%)'),
                tooltip=[alt.Tooltip('ds:T', title='Data'), alt.Tooltip('percentual:Q', format='.2f')]
            ).properties(width=800, height=350).interactive()

            st.subheader(f"📅 Curva diária da projeção para {mes_proj.strftime('%m/%Y')}")
            st.altair_chart(chart_dia, use_container_width=True)

        # --- Exportação com duas abas ---
        st.subheader("📥 Exportar Resultado")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            curva_comparativa.reset_index().to_excel(writer, index=False, sheet_name="Comparativo")

            if df_curva_dia is not None and not df_curva_dia.empty:
                df_curva_dia_export = df_curva_dia.reset_index()
                df_curva_dia_export.columns = ['Data', 'Quantidade', 'Percentual (%)']
                df_curva_dia_export['Percentual (%)'] = df_curva_dia_export['Percentual (%)'].round(2)
                df_curva_dia_export.to_excel(writer, index=False, sheet_name="Curva Diária Projeção")

        st.download_button("📄 Baixar Excel", data=buffer.getvalue(), file_name="comparativo_projecao_ligacoes_SERCOM.xlsx")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
