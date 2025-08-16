import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Interactive Data Portfolio | Yossi", page_icon="✨", layout="wide")

with st.sidebar:
    st.header("👤 Profile")
    name = st.text_input("Name", value="Yossi")
    role = st.text_input("Role", value="Data Scientist & Finance Manager (Ready-mix Concrete)")
    bio = st.text_area("Short bio", value="Data scientist & finance manager; fokus pada analisis bisnis, visualisasi, dan otomasi laporan.")
    contact = st.text_input("Contact (email/LinkedIn)", value="your.name@example.com | linkedin.com/in/yourprofile")

    st.markdown("---")
    st.subheader("📦 Data Source")
    data_choice = st.radio("Choose data source:", ["Use sample data", "Upload CSV"], index=0)
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) if data_choice == "Upload CSV" else None

st.title("✨ Interactive Data Portfolio")
st.caption("Built with Streamlit — interaktif, rapi, dan siap dibagikan.")

@st.cache_data(show_spinner=False)
def make_sample(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-08-10", freq="D")
    df = pd.DataFrame({
        "order_id": np.arange(1, n+1),
        "order_date": rng.choice(dates, size=n, replace=True),
        "customer_segment": rng.choice(["Consumer","Corporate","Home Office","Small Business"], size=n),
        "product_category": rng.choice(["Cement","Aggregate","Ready-Mix","Additives","Equipment","Logistics"], size=n),
        "city": rng.choice(["Jakarta","Surabaya","Bandung","Medan","Semarang","Makassar","Bekasi","Depok","Tangerang","Bogor"], size=n),
        "quantity": rng.integers(1, 20, size=n),
        "unit_price": rng.choice([50000,75000,100000,125000,150000,175000,200000], size=n),
        "discount": rng.choice([0.0,0.05,0.1,0.15], size=n, p=[0.6,0.2,0.15,0.05])
    })
    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)
    return df

def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df

if data_choice == "Use sample data":
    df = make_sample()
else:
    if uploaded is None:
        st.info("Silakan upload file CSV, atau pilih **Use sample data** di sidebar.")
        st.stop()
    df = pd.read_csv(uploaded)
    df = try_parse_dates(df)

st.subheader("🔎 Data Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df))
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", int(df.isna().sum().sum()))
c4.metric("Numeric cols", df.select_dtypes(include=np.number).shape[1])

with st.expander("Preview (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

st.subheader("🎛️ Interactive Filters")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
date_cols = [c for c in df.columns if "date" in c.lower()] + df.select_dtypes(include="datetime").columns.tolist()
seen = set(); date_cols = [x for x in date_cols if not (x in seen or seen.add(x))]

left, right = st.columns(2)
chosen_cat = left.selectbox("Categorical column", options=(cat_cols or ["-"]), index=0 if cat_cols else None)
chosen_num = right.selectbox("Numeric column", options=(num_cols or ["-"]), index=0 if num_cols else None)

chosen_date = None
if len(date_cols) > 0:
    chosen_date = st.selectbox("Date/time column (optional)", options=(date_cols or ["-"]), index=0)
    if chosen_date in df.columns and np.issubdtype(df[chosen_date].dtype, np.datetime64):
        dmin, dmax = df[chosen_date].min(), df[chosen_date].max()
        start, end = st.slider("Date range", min_value=dmin.to_pydatetime(), max_value=dmax.to_pydatetime(),
                               value=(dmin.to_pydatetime(), dmax.to_pydatetime()))
        mask = (df[chosen_date] >= pd.to_datetime(start)) & (df[chosen_date] <= pd.to_datetime(end))
        df = df.loc[mask]

st.subheader("📊 Visualizations & Insights")
if chosen_cat in df.columns:
    top_n = st.slider("Top N categories", 3, 20, 10)
    vc = df[chosen_cat].fillna("Missing").value_counts().head(top_n).reset_index()
    vc.columns = [chosen_cat, "count"]
    fig1 = px.bar(vc, x=chosen_cat, y="count", text="count", title=f"Top {top_n} by {chosen_cat}")
    fig1.update_layout(xaxis_title=chosen_cat, yaxis_title="Count", title_x=0.1)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(f"**Insight:** {chosen_cat} terbanyak: **{vc.iloc[0][chosen_cat]}** ({int(vc.iloc[0]['count'])} baris).")

if chosen_num in df.columns:
    bins = st.slider("Histogram bins", 5, 80, 30)
    fig2 = px.histogram(df, x=chosen_num, nbins=bins, title=f"Distribution of {chosen_num}")
    fig2.update_layout(xaxis_title=chosen_num, yaxis_title="Frequency", title_x=0.1)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"**Insight:** mean={df[chosen_num].mean():.2f}, median={df[chosen_num].median():.2f}, std={df[chosen_num].std():.2f}.")

if (chosen_num in df.columns) and (chosen_date in df.columns if chosen_date is not None else False):
    ts = (df.dropna(subset=[chosen_date, chosen_num])
            .groupby(pd.Grouper(key=chosen_date, freq="D"))[chosen_num]
            .sum().reset_index())
    fig3 = px.line(ts, x=chosen_date, y=chosen_num, markers=True, title=f"Daily Sum of {chosen_num}")
    fig3.update_layout(xaxis_title="Date", yaxis_title=f"Sum of {chosen_num}", title_x=0.1)
    st.plotly_chart(fig3, use_container_width=True)
    if len(ts):
        mx = ts.loc[ts[chosen_num].idxmax()]
        st.markdown(f"**Insight:** puncak {chosen_num} pada `{mx[chosen_date].date()}` = `{mx[chosen_num]:,.2f}`.")

st.subheader(" Export")
st.download_button("Download filtered data as CSV", data=df.to_csv(index=False), file_name="filtered_data.csv", mime="text/csv")

st.subheader(" Project Narrative")
st.markdown("""
**Tujuan:** Menampilkan tren penjualan beton readymix serta disajikan dengan visualisasi data secara interaktif.

**Catatan:** -.
""")

st.markdown("---")
st.caption(f"© {pd.Timestamp.today().year} · {name} — {role} · {contact}")
