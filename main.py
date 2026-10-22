import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Analiza campaniilor de marketing", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Configurare: tipurile de CSV pe care aplicația le recunoaște după coloane
# ----------------------------------------------------------------------------
SCHEME = {
    "produse": {"produs_id", "produs", "categorie", "pret", "cost"},
    "campanii": {"campanie_id", "produs_id", "canal", "data_start", "data_sfarsit",
                 "buget", "afisari", "click_uri", "conversii", "venituri"},
    "vanzari": {"data", "produs_id", "unitati_vandute", "venit", "promovat"},
}
COL_DATA = {"campanii": ["data_start", "data_sfarsit"], "vanzari": ["data"]}
COL_NUM = {
    "produse": ["pret", "cost"],
    "campanii": ["buget", "afisari", "click_uri", "conversii", "venituri"],
    "vanzari": ["unitati_vandute", "venit", "promovat"],
}
ETICHETE = {"produse": "Produse", "campanii": "Campanii", "vanzari": "Vânzări zilnice"}
NUME_ZILE = ["Luni", "Marți", "Miercuri", "Joi", "Vineri", "Sâmbătă", "Duminică"]


# ----------------------------------------------------------------------------
# Funcții de procesare (fără UI)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def citeste_csv(continut: bytes) -> pd.DataFrame:
    """Citește un CSV, detectând automat separatorul (, sau ;) și encodingul."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(continut), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Fișierul nu poate fi citit (encoding necunoscut).")


def normalizeaza_coloane(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def detecteaza_tip(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    for tip, necesare in SCHEME.items():
        if necesare <= cols:
            return tip
    return "altele"


def curata(df: pd.DataFrame, tip: str):
    """Elimină duplicatele, convertește tipurile și completează lipsurile cu mediana."""
    n_init = len(df)
    df = df.drop_duplicates().copy()
    duplicate = n_init - len(df)

    for c in COL_DATA.get(tip, []):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    if COL_DATA.get(tip):
        df = df.dropna(subset=COL_DATA[tip])
    for c in COL_NUM.get(tip, []):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    coloane_num = df.select_dtypes(include="number").columns
    lipsuri = int(df[coloane_num].isna().sum().sum())
    for c in coloane_num:
        df[c] = df[c].fillna(df[c].median())

    if tip == "vanzari":
        df["promovat"] = (df["promovat"] > 0).astype(int)
    return df.reset_index(drop=True), {"duplicate": duplicate, "lipsuri": lipsuri}


def imparte(a, b):
    """Împărțire sigură: 0 sau NaN la numitor => NaN."""
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def construieste_campanii(date: dict):
    """Calculează metricile de marketing și leagă campaniile de produse."""
    c = date.get("campanii")
    if c is None:
        return None
    c = c.copy()
    c["ctr"] = imparte(c["click_uri"], c["afisari"]) * 100
    c["cvr"] = imparte(c["conversii"], c["click_uri"]) * 100
    c["cpc"] = imparte(c["buget"], c["click_uri"])
    c["cpa"] = imparte(c["buget"], c["conversii"])
    c["roas"] = imparte(c["venituri"], c["buget"])
    c["roi_pct"] = imparte(c["venituri"] - c["buget"], c["buget"]) * 100
    c["durata_zile"] = (c["data_sfarsit"] - c["data_start"]).dt.days + 1

    p = date.get("produse")
    if p is not None:
        c = c.merge(p[["produs_id", "produs", "categorie", "cost"]], on="produs_id", how="left")
        c["produs"] = c["produs"].fillna(c["produs_id"].astype(str))
        c["categorie"] = c["categorie"].fillna("Necunoscut")
        c["profit_net"] = c["venituri"] - c["conversii"] * c["cost"] - c["buget"]
    else:
        c["produs"] = c["produs_id"].astype(str)
        c["categorie"] = "Necunoscut"
    return c


def agrega(c: pd.DataFrame, by) -> pd.DataFrame:
    g = c.groupby(by).agg(
        campanii=("campanie_id", "count"), buget=("buget", "sum"), afisari=("afisari", "sum"),
        click_uri=("click_uri", "sum"), conversii=("conversii", "sum"), venituri=("venituri", "sum"),
    ).reset_index()
    if "profit_net" in c.columns:
        g["profit_net"] = c.groupby(by)["profit_net"].sum().values
    g["ctr"] = imparte(g["click_uri"], g["afisari"]) * 100
    g["cvr"] = imparte(g["conversii"], g["click_uri"]) * 100
    g["cpa"] = imparte(g["buget"], g["conversii"])
    g["roas"] = imparte(g["venituri"], g["buget"])
    g["roi_pct"] = imparte(g["venituri"] - g["buget"], g["buget"]) * 100
    return g


def uplift_promo(v: pd.DataFrame):
    """Compară vânzările medii zilnice din zilele cu și fără promovare, pe produs."""
    g = v.groupby(["produs_id", "promovat"])["unitati_vandute"].mean().unstack("promovat")
    if not {0, 1} <= set(g.columns):
        return None
    g = g.rename(columns={0: "medie_fara_promo", 1: "medie_cu_promo"}).dropna()
    g["uplift_pct"] = imparte(g["medie_cu_promo"] - g["medie_fara_promo"], g["medie_fara_promo"]) * 100
    return g.reset_index()


def limite_iqr(serie: pd.Series):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def ron(x) -> str:
    return "–" if pd.isna(x) else f"{x:,.0f} RON".replace(",", " ")


# ----------------------------------------------------------------------------
# Filtre globale (sidebar)
# ----------------------------------------------------------------------------
def aplica_filtre(c: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtre campanii")
    canale = sorted(c["canal"].dropna().unique())
    alese = st.sidebar.multiselect("Canal", canale, default=canale)
    categorii = sorted(c["categorie"].dropna().unique())
    alese_cat = st.sidebar.multiselect("Categorie produs", categorii, default=categorii)
    d_min, d_max = c["data_start"].min().date(), c["data_start"].max().date()
    interval = st.sidebar.date_input("Început campanie între", (d_min, d_max), min_value=d_min, max_value=d_max)

    f = c[c["canal"].isin(alese) & c["categorie"].isin(alese_cat)]
    if isinstance(interval, (tuple, list)) and len(interval) == 2:
        f = f[(f["data_start"].dt.date >= interval[0]) & (f["data_start"].dt.date <= interval[1])]
    return f


# ----------------------------------------------------------------------------
# Pagini
# ----------------------------------------------------------------------------
def pagina_home():
    st.subheader("Descrierea proiectului")
    st.write(
        "Aplicația este destinată utilizatorilor de business care promovează produse prin mai multe canale "
        "de marketing. Încarci unul sau mai multe fișiere CSV, aplicația le recunoaște automat după coloane, "
        "le curăță și îți calculează indicatori de performanță, grafice și analize statistice."
    )
    st.subheader("Cum folosești aplicația")
    st.markdown(
        "1. Mergi la **Încarcă date** și selectează toate CSV-urile deodată.\n"
        "2. Verifică raportul de calitate a datelor (duplicate eliminate, valori lipsă completate).\n"
        "3. Explorează **Dashboard**, **Canale**, **Produse**, **Vânzări în timp** și **Analize avansate**."
    )
    st.subheader("Fișierele recunoscute")
    st.caption("Numele coloanelor trebuie să fie exact cele de mai jos. Orice alt CSV se încarcă și el, "
               "dar poate fi folosit doar în „Analize avansate”.")
    descrieri = {
        "campanii": "O linie = o campanie de promovare (buget, afișări, click-uri, conversii, venituri).",
        "produse": "Catalogul de produse, cu preț și cost. Permite calcularea profitului net.",
        "vanzari": "Vânzările zilnice pe produs; coloana `promovat` (0/1) arată dacă produsul era în promoție.",
    }
    for tip, cols in SCHEME.items():
        with st.expander(f"{ETICHETE[tip]} – {descrieri[tip]}"):
            st.code(", ".join(_ordine(tip)))
    st.subheader("Indicatorii calculați")
    with st.expander("Definiții"):
        st.markdown(
            "- **CTR** = click-uri / afișări\n- **CVR** = conversii / click-uri\n- **CPC** = buget / click-uri\n"
            "- **CPA** = buget / conversii (cât costă o vânzare)\n- **ROAS** = venituri / buget "
            "(sub 1 = campania nu și-a acoperit costul)\n- **ROI %** = (venituri − buget) / buget × 100\n"
            "- **Profit net** = venituri − conversii × cost produs − buget (aproximare)"
        )


def _ordine(tip):
    return {
        "campanii": ["campanie_id", "produs_id", "canal", "data_start", "data_sfarsit", "buget", "afisari",
                     "click_uri", "conversii", "venituri"],
        "produse": ["produs_id", "produs", "categorie", "pret", "cost"],
        "vanzari": ["data", "produs_id", "unitati_vandute", "venit", "promovat"],
    }[tip]


def pagina_incarcare():
    st.subheader("Încarcă fișierele CSV")
    fisiere = st.file_uploader("Poți selecta mai multe fișiere deodată", type=["csv"], accept_multiple_files=True)

    if fisiere:
        date, altele, raport = {}, {}, []
        for f in fisiere:
            try:
                brut = normalizeaza_coloane(citeste_csv(f.getvalue()))
            except Exception as e:
                st.error(f"Nu am putut citi `{f.name}`: {e}")
                continue
            tip = detecteaza_tip(brut)
            df, r = curata(brut, tip)
            if tip == "altele":
                altele[f.name] = df
            elif tip in date:  # mai multe fișiere de același tip se combină
                date[tip] = pd.concat([date[tip], df]).drop_duplicates().reset_index(drop=True)
            else:
                date[tip] = df
            raport.append({"Fișier": f.name, "Tip detectat": ETICHETE.get(tip, "Necunoscut (generic)"),
                           "Rânduri": len(df), "Coloane": df.shape[1],
                           "Duplicate eliminate": r["duplicate"], "Valori lipsă completate": r["lipsuri"]})
        st.session_state.date, st.session_state.altele = date, altele
        st.session_state.raport = pd.DataFrame(raport)

    date, altele = st.session_state.get("date", {}), st.session_state.get("altele", {})
    if not date and not altele:
        st.info("Nu ai încărcat încă niciun fișier. Poți folosi CSV-urile din folderul `date_test`.")
        return

    st.success("Datele sunt încărcate și rămân disponibile în toate paginile.")
    st.markdown("**Raport de calitate a datelor**")
    st.dataframe(st.session_state.raport, hide_index=True)
    st.caption("Valorile lipsă numerice au fost completate cu mediana coloanei; rândurile fără dată validă au fost eliminate.")

    seturi = {ETICHETE[t]: d for t, d in date.items()} | {n: d for n, d in altele.items()}
    taburi = st.tabs(list(seturi))
    for tab, (nume, d) in zip(taburi, seturi.items()):
        with tab:
            st.dataframe(d.head(200))
            st.caption(f"Se afișează primele 200 de rânduri din {len(d)}.")
            st.dataframe(d.describe(include="number").T.round(2))

    if st.button("Șterge datele încărcate"):
        for k in ("date", "altele", "raport"):
            st.session_state.pop(k, None)
        st.rerun()

    lipsa = [ETICHETE[t] for t in SCHEME if t not in date]
    if lipsa:
        st.warning("Nu ai încărcat: " + ", ".join(lipsa) + ". Unele pagini vor fi limitate.")


def pagina_dashboard(c: pd.DataFrame):
    st.subheader("Dashboard – performanța generală")
    if c.empty:
        st.warning("Filtrele curente nu lasă nicio campanie.")
        return
    buget, venit, conv = c["buget"].sum(), c["venituri"].sum(), c["conversii"].sum()
    afis, clk = c["afisari"].sum(), c["click_uri"].sum()

    k = st.columns(4)
    k[0].metric("Buget total", ron(buget))
    k[1].metric("Venituri totale", ron(venit))
    k[2].metric("ROAS", f"{venit / buget:.2f}")
    k[3].metric("ROI", f"{(venit - buget) / buget * 100:.0f}%")
    k = st.columns(4)
    k[0].metric("Conversii", f"{conv:,.0f}".replace(",", " "))
    k[1].metric("CTR", f"{clk / afis * 100:.2f}%" if afis else "–")
    k[2].metric("CVR", f"{conv / clk * 100:.2f}%" if clk else "–")
    k[3].metric("CPA", ron(buget / conv) if conv else "–")
    if "profit_net" in c.columns:
        st.metric("Profit net estimat", ron(c["profit_net"].sum()))

    st.markdown("### Concluzii automate")
    a = agrega(c, "canal").dropna(subset=["roas"])
    if not a.empty:
        best, worst = a.loc[a["roas"].idxmax()], a.loc[a["roas"].idxmin()]
        st.success(f"Cel mai eficient canal: **{best['canal']}** (ROAS {best['roas']:.2f}, ROI {best['roi_pct']:.0f}%).")
        if worst["roas"] < 1:
            st.error(f"**{worst['canal']}** nu își acoperă costurile (ROAS {worst['roas']:.2f}). Merită revizuit bugetul.")
        else:
            st.info(f"Cel mai slab canal este **{worst['canal']}** (ROAS {worst['roas']:.2f}).")
    neren = int((c["roas"] < 1).sum())
    st.warning(f"{neren} din {len(c)} campanii ({neren / len(c) * 100:.0f}%) nu și-au acoperit bugetul (ROAS < 1).")

    st.markdown("### Buget vs. venituri pe canal")
    m = a.melt(id_vars="canal", value_vars=["buget", "venituri"], var_name="indicator", value_name="RON")
    st.plotly_chart(px.bar(m, x="canal", y="RON", color="indicator", barmode="group"))

    st.markdown("### Evoluție lunară")
    lunar = c.assign(luna=c["data_start"].dt.to_period("M").astype(str)).groupby("luna")[["buget", "venituri"]].sum().reset_index()
    st.plotly_chart(px.line(lunar, x="luna", y=["buget", "venituri"], markers=True))

    st.markdown("### Top 10 campanii după ROI")
    cols = ["campanie_id", "produs", "canal", "buget", "venituri", "roas", "roi_pct"]
    st.dataframe(c.nlargest(10, "roi_pct")[cols].round(2), hide_index=True)

    st.download_button("Descarcă campaniile cu metrici (CSV)", c.round(3).to_csv(index=False).encode("utf-8"),
                       "campanii_cu_metrici.csv", "text/csv")


def pagina_canale(c: pd.DataFrame):
    st.subheader("Analiza pe canale de marketing")
    if c.empty:
        st.warning("Filtrele curente nu lasă nicio campanie.")
        return
    a = agrega(c, "canal").sort_values("roas", ascending=False)
    st.dataframe(a.round(2), hide_index=True)

    st.markdown("### ROAS pe canal")
    fig = px.bar(a, x="canal", y="roas", color="roas", color_continuous_scale="RdYlGn")
    fig.add_hline(y=1, line_dash="dash", annotation_text="prag de rentabilitate")
    st.plotly_chart(fig)

    st.markdown("### Buget vs. venituri pe campanie")
    fig = px.scatter(c, x="buget", y="venituri", size="conversii", color="canal", hover_data=["campanie_id", "produs"])
    mx = float(max(c["buget"].max(), c["venituri"].max()))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="venituri = buget",
                             line=dict(dash="dash", color="gray")))
    st.plotly_chart(fig)

    st.markdown("### Pâlnia de conversie")
    alege = st.selectbox("Canal", ["Toate"] + sorted(c["canal"].unique()))
    s = c if alege == "Toate" else c[c["canal"] == alege]
    st.plotly_chart(go.Figure(go.Funnel(y=["Afișări", "Click-uri", "Conversii"],
                                        x=[s["afisari"].sum(), s["click_uri"].sum(), s["conversii"].sum()],
                                        textinfo="value+percent initial")))

    st.markdown("### Variația ROAS între campanii")
    st.plotly_chart(px.box(c, x="canal", y="roas", points="all", hover_data=["campanie_id"]))


def pagina_produse(c: pd.DataFrame):
    st.subheader("Analiza pe produse și categorii")
    if c.empty:
        st.warning("Filtrele curente nu lasă nicio campanie.")
        return
    p = agrega(c, "produs")
    optiuni = [m for m in ["roas", "roi_pct", "venituri", "conversii", "buget", "profit_net"] if m in p.columns]
    metrica = st.radio("Clasament după", optiuni, horizontal=True)
    n = min(10, len(p))
    s1, s2 = st.columns(2)
    s1.markdown(f"**Top {n}**")
    s1.plotly_chart(px.bar(p.nlargest(n, metrica), x=metrica, y="produs", orientation="h").update_yaxes(autorange="reversed"))
    s2.markdown(f"**Ultimele {n}**")
    s2.plotly_chart(px.bar(p.nsmallest(n, metrica), x=metrica, y="produs", orientation="h").update_yaxes(autorange="reversed"))

    st.markdown("### Pe categorii")
    cat = agrega(c, "categorie")
    s1, s2 = st.columns(2)
    s1.plotly_chart(px.pie(cat, names="categorie", values="buget", title="Distribuția bugetului"))
    s2.plotly_chart(px.pie(cat, names="categorie", values="venituri", title="Distribuția veniturilor"))
    st.dataframe(cat.round(2), hide_index=True)

    st.markdown("### Toate produsele")
    st.dataframe(p.sort_values(metrica, ascending=False).round(2), hide_index=True)


def pagina_vanzari(date: dict):
    st.subheader("Vânzări în timp și efectul promovării")
    v = date.get("vanzari")
    if v is None:
        st.info("Încarcă fișierul de vânzări zilnice (`vanzari_zilnice.csv`) pentru această pagină.")
        return
    p = date.get("produse")
    nume = dict(zip(p["produs_id"], p["produs"])) if p is not None else {}
    v = v.assign(produs=v["produs_id"].map(nume).fillna(v["produs_id"].astype(str)))

    alese = st.multiselect("Produse (gol = toate)", sorted(v["produs"].unique()))
    if alese:
        v = v[v["produs"].isin(alese)]
    if v.empty:
        return

    zi = v.groupby("data")[["venit", "unitati_vandute"]].sum().reset_index()
    zi["medie_7_zile"] = zi["venit"].rolling(7, min_periods=1).mean()
    k = st.columns(3)
    k[0].metric("Venit total", ron(v["venit"].sum()))
    k[1].metric("Unități vândute", f"{v['unitati_vandute'].sum():,.0f}".replace(",", " "))
    k[2].metric("Zile-produs promovate", f"{v['promovat'].mean() * 100:.0f}%")

    st.markdown("### Venit zilnic")
    st.plotly_chart(px.line(zi, x="data", y=["venit", "medie_7_zile"]))

    st.markdown("### Efectul promovării")
    u = uplift_promo(v)
    if u is None:
        st.info("Nu există atât zile promovate, cât și nepromovate pentru comparație.")
    else:
        u["produs"] = u["produs_id"].map(nume).fillna(u["produs_id"].astype(str))
        st.success(f"În medie, o zi promovată aduce cu **{u['uplift_pct'].mean():.0f}%** mai multe unități vândute decât una nepromovată.")
        st.plotly_chart(px.bar(u.sort_values("uplift_pct"), x="uplift_pct", y="produs", orientation="h",
                               labels={"uplift_pct": "Creștere vânzări (%)"}))
        st.dataframe(u.drop(columns="produs_id").round(2), hide_index=True)

    s1, s2 = st.columns(2)
    zs = v.assign(zi=v["data"].dt.dayofweek).groupby("zi")["venit"].sum().reindex(range(7)).reset_index()
    zs["zi"] = zs["zi"].map(dict(enumerate(NUME_ZILE)))
    s1.markdown("**Venit pe zilele săptămânii**")
    s1.plotly_chart(px.bar(zs, x="zi", y="venit"))
    lun = v.assign(luna=v["data"].dt.to_period("M").astype(str)).groupby("luna")["venit"].sum().reset_index()
    s2.markdown("**Venit lunar**")
    s2.plotly_chart(px.bar(lun, x="luna", y="venit"))


def pagina_avansat(seturi: dict):
    st.subheader("Analize statistice avansate")
    if not seturi:
        st.info("Încarcă date pentru a folosi această pagină.")
        return
    nume = st.selectbox("Setul de date analizat", list(seturi))
    df = seturi[nume]
    num = [c for c in df.select_dtypes(include="number").columns if df[c].notna().any()]
    if len(num) < 2:
        st.warning("Setul ales are mai puțin de 2 coloane numerice.")
        return
    t_cor, t_reg, t_clu, t_out = st.tabs(["Corelații", "Regresie", "Clustering (K-Means + PCA)", "Outlieri"])

    with t_cor:
        cor = df[num].corr()
        st.plotly_chart(px.imshow(cor, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"))
        perechi = cor.where(np.triu(np.ones(cor.shape, dtype=bool), 1)).stack().reset_index()
        perechi.columns = ["variabila_1", "variabila_2", "corelatie"]
        perechi["abs"] = perechi["corelatie"].abs()
        st.markdown("**Cele mai puternice 5 corelații**")
        st.dataframe(perechi.nlargest(5, "abs").drop(columns="abs").round(3), hide_index=True)

    with t_reg:
        y_col = st.selectbox("Variabila țintă (Y)", num, key="reg_y")
        x_cols = st.multiselect("Variabile explicative (X)", [c for c in num if c != y_col], key="reg_x")
        if x_cols:
            d = df[x_cols + [y_col]].replace([np.inf, -np.inf], np.nan)
            d = d.fillna(d.median()).astype(float)
            model = sm.OLS(d[y_col], sm.add_constant(d[x_cols])).fit()
            k = st.columns(3)
            k[0].metric("R²", f"{model.rsquared:.3f}")
            k[1].metric("R² ajustat", f"{model.rsquared_adj:.3f}")
            k[2].metric("Observații", int(model.nobs))
            coef = pd.DataFrame({"variabila": model.params.index, "coeficient": model.params.values,
                                 "p_value": model.pvalues.values})
            coef["semnificativ (p<0.05)"] = np.where(coef["p_value"] < 0.05, "da", "nu")
            st.dataframe(coef.round(4), hide_index=True)
            comp = pd.DataFrame({"real": d[y_col], "prezis": model.fittedvalues})
            fig = px.scatter(comp, x="real", y="prezis")
            lo, hi = float(comp.min().min()), float(comp.max().max())
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="predicție perfectă",
                                     line=dict(dash="dash", color="gray")))
            st.plotly_chart(fig)
            with st.expander("Rezumat statsmodels complet"):
                st.text(model.summary())
        else:
            st.info("Alege cel puțin o variabilă explicativă.")

    with t_clu:
        feat = st.multiselect("Variabile folosite", num, default=num[:3], key="clu_f")
        k_cl = st.slider("Număr de clustere", 2, 6, 3)
        if len(feat) >= 2 and len(df) > k_cl:
            d = df[feat].replace([np.inf, -np.inf], np.nan)
            d = d.fillna(d.median())
            z = StandardScaler().fit_transform(d)
            et = KMeans(n_clusters=k_cl, n_init=10, random_state=42).fit_predict(z)
            pca = PCA(n_components=2).fit(z)
            proj = pca.transform(z)
            plot = pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "cluster": et.astype(str)})
            st.plotly_chart(px.scatter(plot, x="PC1", y="PC2", color="cluster"))
            st.caption(f"Varianța explicată de primele 2 componente: {pca.explained_variance_ratio_.sum() * 100:.1f}%")
            medii = d.assign(cluster=et).groupby("cluster").mean().round(2)
            medii["număr_rânduri"] = pd.Series(et).value_counts().sort_index().values
            st.markdown("**Profilul mediu al fiecărui cluster**")
            st.dataframe(medii)
        else:
            st.info("Alege cel puțin 2 variabile.")

    with t_out:
        coloane = st.multiselect("Coloane analizate (metoda IQR)", num, key="out_c")
        for col in coloane:
            lo, hi = limite_iqr(df[col].dropna())
            out = df[(df[col] < lo) | (df[col] > hi)]
            st.markdown(f"**{col}** – limite: {lo:.2f} … {hi:.2f} | outlieri: {len(out)}")
            st.plotly_chart(px.box(df, y=col, points="outliers"))
            if len(out):
                st.dataframe(out.head(50))


# ----------------------------------------------------------------------------
# Aplicația
# ----------------------------------------------------------------------------
st.markdown(
    "<style>.custom-title {color: #CC5500 !important; font-size: 40px; text-align: center;}</style>",
    unsafe_allow_html=True,
)
st.markdown('<h1 class="custom-title">Analiza campaniilor de marketing</h1>', unsafe_allow_html=True)

MENIU = ["Home", "Încarcă date", "Dashboard", "Canale", "Produse", "Vânzări în timp", "Analize avansate"]
if "pagina_curenta" not in st.session_state:
    st.session_state.pagina_curenta = "Home"
pagina = st.sidebar.selectbox("Meniu", MENIU, index=MENIU.index(st.session_state.pagina_curenta))
st.session_state.pagina_curenta = pagina

date = st.session_state.get("date", {})
altele = st.session_state.get("altele", {})
campanii = construieste_campanii(date)

if pagina == "Home":
    pagina_home()
elif pagina == "Încarcă date":
    pagina_incarcare()
elif pagina in ("Dashboard", "Canale", "Produse"):
    if campanii is None:
        st.info("Încarcă fișierul de campanii (`campanii.csv`) din pagina „Încarcă date”.")
    else:
        filtrat = aplica_filtre(campanii)
        {"Dashboard": pagina_dashboard, "Canale": pagina_canale, "Produse": pagina_produse}[pagina](filtrat)
elif pagina == "Vânzări în timp":
    pagina_vanzari(date)
elif pagina == "Analize avansate":
    seturi = {}
    if campanii is not None:
        seturi["Campanii (cu metrici)"] = campanii
    if "vanzari" in date:
        seturi["Vânzări zilnice"] = date["vanzari"]
    if "produse" in date:
        seturi["Produse"] = date["produse"]
    seturi.update({f"Fișier: {n}": d for n, d in altele.items()})
    pagina_avansat(seturi)
