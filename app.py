# cp1_uber_dashboard_multitab.py
# Execução: streamlit run cp1_uber_dashboard_multitab.py

import textwrap
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
from PIL import Image
from pathlib import Path

# =========================
# CONFIG (tem que ser a 1ª chamada Streamlit e só 1x)
# =========================
st.set_page_config(page_title="CP1 – Dashboard Profissional + Análise de Dados", layout="wide")

st.title("CP1 - Backlog: DATA SCIENCE And STATISTICAL COMPUTING")

# =========================
# --------- ABAS ----------
# =========================
tabs = st.tabs(["🏠 Home", "🎓 Formação e Experiência", "🛠 Skills", "📊 Análise Uber"])

# =========================
# Funções auxiliares (usadas na aba 4)
# =========================
def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        lc = col.lower()
        if lc in {"date", "data"}:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif lc in {"time", "hora"}:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.time
        elif any(k in lc for k in ["distance", "value", "avg", "rating", "fare", "km", "amount"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def basic_schema(df: pd.DataFrame) -> pd.DataFrame:
    info = []
    for col in df.columns:
        s = df[col]
        info.append({
            "coluna": col,
            "tipo": str(s.dtype),
            "ausentes": int(s.isna().sum()),
            "#valores_unicos": int(s.nunique(dropna=True)),
            "exemplos": ", ".join(map(lambda x: str(x)[:25], s.dropna().unique()[:6]))
        })
    return pd.DataFrame(info)

def proportion_confint_wilson(successes: int, nobs: int, alpha: float = 0.05) -> Tuple[float, float]:
    if nobs == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / nobs
    denom = 1 + z**2 / nobs
    centre = phat + z**2/(2*nobs)
    margin = z*np.sqrt((phat*(1-phat) + z**2/(4*nobs))/nobs)
    return ((centre - margin)/denom, (centre + margin)/denom)

def mean_confint_t(data: pd.Series, alpha: float = 0.05) -> Tuple[float, float, float, int]:
    data = pd.to_numeric(data, errors="coerce").dropna()
    n = len(data)
    if n < 2:
        return (np.nan, np.nan, np.nan, n)
    mean = data.mean()
    sd = data.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
    return (mean, mean - tcrit*se, mean + tcrit*se, n)

def choose_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.select_dtypes(include=["number", "float", "int"]).columns if df[c].notna().sum() > 0]

def choose_categorical_columns(df: pd.DataFrame, max_card: int = 40) -> List[str]:
    cats = []
    for c in df.columns:
        if df[c].dtype == "object" or df[c].dtype.name == "category":
            if df[c].nunique(dropna=True) <= max_card:
                cats.append(c)
    return cats

def clean_booking_status(s: pd.Series) -> pd.Series:
    def map_status(x):
        if pd.isna(x): return np.nan
        x = str(x).strip().lower()
        if "complete" in x: return "Completed"
        if "customer" in x and "cancel" in x: return "Cancelled by Customer"
        if "driver" in x and "cancel" in x: return "Cancelled by Driver"
        if "incomplete" in x: return "Incomplete"
        return x.title()
    return s.apply(map_status)

# =========================
# ABA 1: Home
# =========================
with tabs[0]:
    st.title("🏠 Home")
    img_path = Path("Mídia.jpg")
    if img_path.exists():
        st.image(Image.open(img_path), caption="Minha Imagem", width=150)
    else:
        st.warning("Imagem 'Mídia.jpg' não encontrada na pasta do app.")
    st.markdown(
        """
        **Apresentação:**

        - Meu nome é Nickolas, tenho 19 anos e estou no segundo ano do Ensino Superior cursando Engenharia de Software na Fiap.
        - Este projeto é um dashboard interativo em Streamlit que também serve como portfólio.
        - Abas: Introdução • Formação/Experiências • Soft Skills • **Análise de dados da Uber**.
        """
    )

# =========================
# ABA 2: Formação e Experiência
# =========================
with tabs[1]:
    st.title("🎓 Formação e Experiência")
    st.markdown(
        """
        **Formação Acadêmica:**
        - Ensino Médio Técnico <TI> - <Senac>
        - Curso Superior <Engenharia de Software> – <Fiap> (2024-2027)
        - Cursos Online relevantes: <Cultura Inglêsa>, <SAGA>
        
        **Experiências:**
        - Projeto Tech Mahindra (Fiap/2024) – Website e propostas de melhoria, com apresentação à empresa parceira.
        - Desafio Lixo Eletrônico (Senac/2023) – Turma vencedora; 519 kg coletados por mim (total 746,8 kg).
        - ONU – COP Geopolítica das Águas (Senac/2023) – Simulação representando a República do Quênia.
        """
    )

# =========================
# ABA 3: Skills
# =========================
with tabs[2]:
    st.title("🛠 Skills")
    skills = {
        "Habilidade": [
            "Microsoft Office","Lógica de Programação","Python","SQL","Java","JavaScript",
            "Trabalho em Equipe","Pontualidade","Proatividade","Organização","Power BI","Inglês",
            "Front-End","Back-End","Html","Cisco Packet Tracer","Comunicação","Visual Studio Code","IntelliJ"
        ],
        "Categoria": [
            "Desenvolvimento","Programação","Programação","Desenvolvimento","Programação","Programação",
            "Pessoal","Pessoal","Pessoal","Pessoal","Desenvolvimento","Pessoal","Programação","Programação",
            "Programação","Desenvolvimento","Pessoal","Desenvolvimento","Desenvolvimento"
        ],
        "Nível (autoavaliação)": [6,7,6,7,8,8,8,9,8,8,5,10,6,6,6,9,10,7,7]
    }
    df_skills = pd.DataFrame(skills)
    st.subheader("📋 Lista de Habilidades")
    st.dataframe(df_skills)

# =========================
# ABA 4: Análise Uber (CSV carregado automaticamente)
# =========================
# -------------------------
# Leitura do CSV
# -------------------------
DATA_PATHS = [Path("uber.csv"), Path("./uber.csv"), Path("/mnt/data/uber.csv")]
csv_path = next((p for p in DATA_PATHS if p.exists()), None)

if csv_path is None:
    st.error("Arquivo 'uber.csv' não encontrado. Coloque o arquivo na mesma pasta do script ou em /mnt/data/uber.csv")
    st.stop()

df_raw = pd.read_csv(csv_path, encoding="utf-8")
# cópia que vamos manipular
df = df_raw.copy()

# -------------------------
# Funções utilitárias
# -------------------------
def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        lc = col.lower()
        if lc in {"date", "data"}:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif lc in {"time", "hora"}:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.time
        elif any(k in lc for k in ["distance", "value", "avg", "rating", "fare", "km", "amount", "booking value", "ride distance"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def clean_booking_status(s: pd.Series) -> pd.Series:
    def map_status(x):
        if pd.isna(x): return np.nan
        x = str(x).strip().lower()
        if "complete" in x: return "Completed"
        if "no driver" in x or "no_driver" in x: return "No Driver Found"
        if "incomplete" in x: return "Incomplete"
        if "cancel" in x and "customer" in x: return "Cancelled by Customer"
        if "cancel" in x and "driver" in x: return "Cancelled by Driver"
        return x.title()
    return s.apply(map_status)

def choose_numeric_columns(df: pd.DataFrame):
    return [c for c in df.select_dtypes(include=["number"]).columns if df[c].notna().sum() > 0]

# -------------------------
# Pré-processamento
# -------------------------
df = infer_types(df)

# padroniza Booking Status se existir coluna com esse nome (ou variações)
possible_status_cols = [c for c in df.columns if c.lower().strip() in ("booking status", "status", "booking_status")]
if possible_status_cols:
    st_col = possible_status_cols[0]
    df[st_col] = clean_booking_status(df[st_col]).astype("category")
    booking_status_col = st_col
else:
    booking_status_col = None

# -------------------------
# UI / Layout
# -------------------------
st.title("📊 Análise Uber 2024 — Eficiência Logística & Aspectos Financeiros")
st.caption(f"Arquivo usado: {csv_path} — registros: {len(df):,}")

tabs = st.tabs(["🏁 Introdução", "🚦 Eficiência Logística", "💰 Aspectos Financeiros"])

# -------------------------
# Aba: Introdução
# -------------------------
with tabs[0]:
    st.header("Resumo do dataset")

    st.markdown(
        """
        **Introdução:**

        O presente estudo tem como objetivo analisar um conjunto de dados de corridas da Uber, explorando aspectos que impactam diretamente tanto a eficiência logística quanto o desempenho financeiro da plataforma.
        A análise será dividida em dois eixos principais:
        Eficiência Logística – avaliando fatores que influenciam a operação, como taxas de cancelamento, indisponibilidade de motoristas, tempos médios de atendimento (VTAT) e de deslocamento até o cliente (CTAT), além dos principais motivos de falhas nas corridas. Esse eixo busca compreender como a Uber pode otimizar sua malha de motoristas, reduzir falhas operacionais e aumentar a confiabilidade do serviço.
        Aspectos Financeiros – investigando o comportamento do faturamento das corridas, valores médios por tipo de veículo, relação entre distância percorrida e valor pago, além das preferências de métodos de pagamento utilizados pelos clientes. Esse eixo permitirá identificar padrões de receita, ticket médio e potenciais perdas financeiras relacionadas a cancelamentos e corridas incompletas.
        Com essa abordagem, será possível não apenas compreender como os fatores logísticos impactam diretamente a qualidade do serviço, mas também como esses elementos se refletem na sustentabilidade financeira da operação.
        """
    )

    st.markdown(f"- Registros: **{len(df):,}**\n- Colunas: **{len(df.columns)}**")
    st.markdown("Colunas principais detectadas:")
    st.dataframe(pd.DataFrame({
        "coluna": df.columns,
        "tipo_detectado": [str(df[c].dtype) for c in df.columns],
        "nulos": [int(df[c].isna().sum()) for c in df.columns]
    }).reset_index(drop=True))

# -------------------------
# Aba: Eficiência Logística
# -------------------------
with tabs[1]:
    st.header("🚦 Eficiência Logística")
    # status geral
    if booking_status_col:
        st.subheader("Taxa de Atendimento (Status das corridas)")
        counts = df[booking_status_col].value_counts(dropna=False)
        fig = px.pie(values=counts.values, names=counts.index, title="Distribuição por Booking Status")
        st.plotly_chart(fig, use_container_width=True)

    # motivos de cancelamento (cliente / driver)
    st.subheader("Principais motivos de cancelamento / incompletude")
    # tenta detectar colunas de motivo com nomes comuns
    motivo_cols = [c for c in df.columns if "reason" in c.lower() or "cancel" in c.lower() and "reason" in c.lower()]
    if len(motivo_cols) == 0:
        # fallback por nomes vistos no CSV
        for name in ["Reason for cancelling by Customer","Driver Cancellation Reason","Reason for Incomplete Rides","Reason for Cancelled Rides by Customer"]:
            if name in df.columns:
                motivo_cols.append(name)
    for col in motivo_cols:
        top = df[col].dropna().value_counts().head(8)
        if len(top) > 0:
            st.markdown(f"**{col}** (top motivos)")
            fig = px.bar(top, x=top.index, y=top.values, labels={'x':'Motivo','y':'Qtd'})
            fig.update_layout(xaxis_tickangle= -45)
            st.plotly_chart(fig, use_container_width=True)

    # VTAT / CTAT
    st.subheader("Tempos VTAT / CTAT (chegada do veículo e deslocamento)")
    vt_col = next((c for c in df.columns if "vtat" in c.lower()), None)
    ct_col = next((c for c in df.columns if "ctat" in c.lower()), None)
    if vt_col and ct_col:
        st.markdown(f"Colunas usadas: **{vt_col}**, **{ct_col}**")
        st.plotly_chart(px.box(df[[vt_col, ct_col]].melt(var_name="tipo", value_name="tempo").dropna(),
                               x="tipo", y="tempo", title="Distribuição VTAT vs CTAT"), use_container_width=True)
        st.dataframe(df[[vt_col, ct_col]].describe().T)
    else:
        st.info("Colunas VTAT/CTAT não encontradas no dataset.")

    # Cancelamentos por Vehicle Type (se existir)
    veh_col = next((c for c in df.columns if "vehicle" in c.lower()), None)
    if veh_col and booking_status_col:
        st.subheader("Cancelamentos por tipo de veículo")
        df_cancel = df[df[booking_status_col] != "Completed"]
        if len(df_cancel) > 0:
            counts = df_cancel[veh_col].value_counts().head(20)
            st.plotly_chart(px.bar(counts, x=counts.index, y=counts.values, title="Cancelamentos por Vehicle Type"), use_container_width=True)
        else:
            st.info("Sem registros de cancelamento para plotar.")

    # padrões por horário (se existir Date/Time)
    date_col = next((c for c in df.columns if c.lower() in ("date", "data")), None)
    time_col = next((c for c in df.columns if c.lower() in ("time", "hora")), None)
    if date_col:
        st.subheader("Padrões por dia da semana / horário")
        df['_date'] = pd.to_datetime(df[date_col], errors="coerce")
        if df['_date'].notna().any():
            df['_dow'] = df['_date'].dt.day_name()
            agg = df.groupby('_dow')[booking_status_col].apply(lambda s: (s=="Completed").mean() if booking_status_col else np.nan).sort_index()
            st.markdown("**Taxa de conclusão por dia da semana (proporção concluída)**")
            st.plotly_chart(px.bar(agg, x=agg.index, y=agg.values, labels={'x':'Dia','y':'Proporção Concluída'}), use_container_width=True)
        df.drop(columns=['_date','_dow'], errors='ignore')

# -------------------------
# Aba: Aspectos Financeiros
# -------------------------
with tabs[2]:
    st.header("💰 Aspectos Financeiros")
    # Booking Value e Ride Distance
    bk_col = next((c for c in df.columns if "booking" in c.lower() and "value" in c.lower()), "Booking Value" if "Booking Value" in df.columns else None)
    dist_col = next((c for c in df.columns if "distance" in c.lower()), "Ride Distance" if "Ride Distance" in df.columns else None)

    if bk_col and bk_col in df.columns:
        st.subheader("Distribuição do valor das corridas")
        st.plotly_chart(px.histogram(df, x=bk_col, nbins=40, title="Booking Value Distribution").update_layout(xaxis_title="Valor"), use_container_width=True)
        st.dataframe(df[bk_col].describe().T)

    if bk_col in df.columns and dist_col in df.columns:
        st.subheader("Correlação: Distância x Valor")
        st.plotly_chart(px.scatter(df, x=dist_col, y=bk_col, trendline="ols", title="Ride Distance vs Booking Value"), use_container_width=True)
        corr = df[[dist_col, bk_col]].corr().iloc[0,1]
        st.markdown(f"**Correlação pearson (distância, valor):** {corr:.3f}")

    # Ticket médio por vehicle type
    if veh_col and bk_col in df.columns:
        st.subheader("Ticket médio por tipo de veículo")
        ticket = df.groupby(veh_col)[bk_col].mean().sort_values(ascending=False).dropna()
        st.plotly_chart(px.bar(ticket, x=ticket.index, y=ticket.values, title="Ticket Médio por Vehicle Type"), use_container_width=True)

    # Métodos de pagamento
    pay_col = next((c for c in df.columns if "payment" in c.lower()), None)
    if pay_col:
        st.subheader("Métodos de pagamento")
        pay = df[pay_col].value_counts(normalize=True).mul(100).round(2)
        st.plotly_chart(px.pie(values=pay.values, names=pay.index, title="Distribuição de Métodos de Pagamento (%)"), use_container_width=True)
        st.dataframe(pay)

    # Receita perdida (corridas não concluídas)
    if bk_col in df.columns and booking_status_col:
        lost = df.loc[df[booking_status_col] != "Completed", bk_col].sum(min_count=1)
        st.metric("Receita estimada perdida (corridas não concluídas)", f"{lost:,.2f}")