import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# ────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atlantic Music Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_MAP = {
    "album":       "#3498db",
    "single":      "#e74c3c",
    "compilation": "#2ecc71",
}

# ────────────────────────────────────────────────────────────
#  LOAD & CLEAN DATA
# ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process():

    # ── Find the CSV automatically ───────────────────────────
    csv_name = "Atlantic_United_States.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_name)

    if not os.path.exists(csv_path):
        st.error(
            f"CSV file not found!\n\n"
            f"Please place  **{csv_name}**  in the same folder as app.py\n\n"
            f"Looking in: `{script_dir}`"
        )
        st.stop()

    # ── Load ─────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # ── Clean & type-cast ─────────────────────────────────────
    df["date"]         = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["position"]     = pd.to_numeric(df["position"],     errors="coerce")
    df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce")
    df["duration_ms"]  = pd.to_numeric(df["duration_ms"],  errors="coerce")
    df["total_tracks"] = pd.to_numeric(df["total_tracks"], errors="coerce")
    df["is_explicit"]  = df["is_explicit"].astype(bool)
    df["song"]         = df["song"].str.strip()
    df["artist"]       = df["artist"].str.strip()
    df["album_type"]   = df["album_type"].str.strip().str.lower()

    df = df[df["position"].between(1, 50)]
    df = df[~df.duplicated(subset=["date", "song", "artist"])]
    df = df.dropna(subset=["date", "position", "song", "artist"])

    # ── Feature engineering ───────────────────────────────────
    df["duration_min"] = (df["duration_ms"] / 60_000).round(2)

    song_stats = (
        df.groupby("song")
        .agg(
            days_on_chart   = ("date",       "nunique"),
            avg_rank        = ("position",   "mean"),
            best_rank       = ("position",   "min"),
            rank_volatility = ("position",   "std"),
            avg_popularity  = ("popularity", "mean"),
            peak_popularity = ("popularity", "max"),
        )
        .round(2)
        .reset_index()
    )
    song_stats["rank_volatility"] = song_stats["rank_volatility"].fillna(0)
    df = df.merge(song_stats, on="song", how="left")

    df = df.sort_values(["song", "date"])
    df["popularity_trend"] = (
        df.groupby("song")["popularity"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .round(2)
    )
    df["rank_change"] = df.groupby("song")["position"].diff().fillna(0)

    df["rank_tier"] = pd.cut(
        df["position"],
        bins=[0, 10, 20, 50],
        labels=["Top 10", "Top 11-20", "Top 21-50"],
    )
    df["duration_bucket"] = pd.cut(
        df["duration_min"],
        bins=[0, 2.5, 3.5, 4.5, 100],
        labels=["<2.5 min", "2.5-3.5 min", "3.5-4.5 min", ">4.5 min"],
    )

    df = df.sort_values(["date", "position"]).reset_index(drop=True)
    return df


# ── Song KPI table ────────────────────────────────────────────
def get_song_kpis(df):
    kpis = (
        df.groupby(["song", "artist"])
        .agg(
            days_on_chart   = ("date",         "nunique"),
            avg_rank        = ("position",     "mean"),
            best_rank       = ("position",     "min"),
            rank_volatility = ("position",     "std"),
            avg_popularity  = ("popularity",   "mean"),
            peak_popularity = ("popularity",   "max"),
            is_explicit     = ("is_explicit",  "first"),
            album_type      = ("album_type",   "first"),
            duration_min    = ("duration_min", "first"),
            total_tracks    = ("total_tracks", "first"),
        )
        .round(2)
        .reset_index()
    )
    kpis["rank_volatility"] = kpis["rank_volatility"].fillna(0)
    max_days = kpis["days_on_chart"].max()
    kpis["longevity_score"] = (
        (kpis["days_on_chart"] / max_days) * 0.6
        + ((51 - kpis["avg_rank"]) / 50) * 0.4
    ).round(3)
    return kpis.sort_values("longevity_score", ascending=False).reset_index(drop=True)


# ── Artist KPI table ──────────────────────────────────────────
def get_artist_kpis(df):
    agg = (
        df.groupby("artist")
        .agg(
            total_days        = ("date",       "nunique"),
            unique_songs      = ("song",       "nunique"),
            avg_rank          = ("position",   "mean"),
            best_rank         = ("position",   "min"),
            avg_popularity    = ("popularity", "mean"),
            total_appearances = ("position",   "count"),
        )
        .round(2)
        .reset_index()
    )
    agg["dominance_pct"] = (agg["total_appearances"] / len(df) * 100).round(2)
    return agg.sort_values("dominance_pct", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
df        = load_and_process()
song_kpis = get_song_kpis(df)
art_kpis  = get_artist_kpis(df)

all_artists = sorted(df["artist"].unique())
all_songs   = sorted(df["song"].unique())

# ────────────────────────────────────────────────────────────
#  SIDEBAR FILTERS
# ────────────────────────────────────────────────────────────
st.sidebar.title("🎛️ Filters")

date_min = df["date"].min().date()
date_max = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

rank_range = st.sidebar.slider("Position range", 1, 50, (1, 50))

sel_artists = st.sidebar.multiselect("Filter by artist", all_artists)
sel_songs   = st.sidebar.multiselect("Filter by song",   all_songs)

album_types_all = sorted(df["album_type"].unique().tolist())
sel_album_types = st.sidebar.multiselect(
    "Album type",
    album_types_all,
    default=album_types_all,
)

explicit_filter = st.sidebar.radio(
    "Explicit content",
    ["All", "Explicit only", "Clean only"],
)

# ── Apply filters ─────────────────────────────────────────────
fdf = df.copy()
fdf = fdf[fdf["date"].dt.date.between(*date_range)]
fdf = fdf[fdf["position"].between(*rank_range)]
if sel_artists:
    fdf = fdf[fdf["artist"].isin(sel_artists)]
if sel_songs:
    fdf = fdf[fdf["song"].isin(sel_songs)]
if sel_album_types:
    fdf = fdf[fdf["album_type"].isin(sel_album_types)]
if explicit_filter == "Explicit only":
    fdf = fdf[fdf["is_explicit"] == True]
elif explicit_filter == "Clean only":
    fdf = fdf[fdf["is_explicit"] == False]

song_kpis_f = song_kpis[song_kpis["song"].isin(fdf["song"].unique())]

# ────────────────────────────────────────────────────────────
#  HEADER + KPI TILES
# ────────────────────────────────────────────────────────────
st.title("🎵 Atlantic Recording Corporation — US Playlist Analytics")
st.caption(
    f"📅 {df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}  ·  "
    f"🎵 {df['song'].nunique()} songs  ·  "
    f"👤 {df['artist'].nunique()} artists  ·  "
    f"💿 Types: album, single, compilation"
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total records",  f"{len(fdf):,}")
k2.metric("Unique songs",   fdf["song"].nunique())
k3.metric("Unique artists", fdf["artist"].nunique())
k4.metric("Avg popularity", f"{fdf['popularity'].mean():.1f}")
k5.metric("Explicit %",     f"{fdf['is_explicit'].mean()*100:.1f}%")
k6.metric("Days in range",  fdf["date"].nunique())

st.divider()

# ────────────────────────────────────────────────────────────
#  TABS
# ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📅 Playlist Timeline",
    "🎵 Song Performance",
    "👤 Artist Dominance",
    "📊 Popularity Analytics",
    "🎭 Content Attributes",
    "🔬 Advanced EDA",
])

# ════════════════════════════════════════════════════════════
#  TAB 1 — PLAYLIST TIMELINE
# ════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📅 Playlist Timeline Explorer")

    # Rank movement
    col1, col2 = st.columns(2)
    with col1:
        mv = fdf.copy().sort_values(["song", "date"])
        mv["rc"] = mv.groupby("song")["position"].diff()
        mv = mv.dropna(subset=["rc"]).copy()
        mv["movement"] = pd.cut(mv["rc"], bins=[-100,-1,0,100],
                                 labels=["Moved Up","Stable","Moved Down"])
        mvmt = mv.groupby(["date","movement"], observed=True).size().reset_index(name="count")
        if not mvmt.empty:
            fig = px.area(mvmt, x="date", y="count", color="movement",
                           title="Daily rank movement (up / stable / down)",
                           color_discrete_map={"Moved Up":"#2ecc71","Stable":"#95a5a6","Moved Down":"#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        ee = (
            fdf.groupby("song")
            .agg(entry_date=("date","min"), exit_date=("date","max"))
            .assign(chart_run=lambda x:(x["exit_date"]-x["entry_date"]).dt.days+1)
            .reset_index()
        )
        fig2 = px.scatter(ee, x="entry_date", y="chart_run",
                           hover_name="song", color="chart_run",
                           color_continuous_scale="Plasma",
                           title="Song longevity vs first entry date",
                           labels={"entry_date":"First appeared","chart_run":"Days on chart"})
        st.plotly_chart(fig2, use_container_width=True)

    # Fast risers
    st.subheader("⬆️ Fast Risers — Biggest Single-Day Rank Jumps")
    tmp = fdf.copy().sort_values(["song","date"])
    tmp["rc"] = tmp.groupby("song")["position"].diff()
    risers = tmp[tmp["rc"] < 0].copy()
    risers["rank_gain"] = (-risers["rc"]).astype(int)
    st.dataframe(
        risers.nlargest(10,"rank_gain")[["date","song","artist","position","rank_gain"]].reset_index(drop=True),
        use_container_width=True,
    )

    # Stable songs
    st.subheader("🐢 Most Stable Songs — Lowest Rank Volatility")
    stable = (
        fdf.groupby("song")
        .agg(days=("date","nunique"), rank_vol=("position","std"), avg_rank=("position","mean"))
        .reset_index()
    )
    stable["rank_vol"] = stable["rank_vol"].fillna(0)
    st.dataframe(
        stable[stable["days"] >= 7].nsmallest(10,"rank_vol").reset_index(drop=True),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════
#  TAB 2 — SONG PERFORMANCE
# ════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🎵 Song-Level KPIs")

    col1, col2 = st.columns(2)
    with col1:
        top15 = song_kpis_f.nlargest(15, "days_on_chart")
        fig = px.bar(top15, x="days_on_chart", y="song", orientation="h",
                      color="avg_rank", color_continuous_scale="RdYlGn_r",
                      title="Top 15 Songs by Days on Chart",
                      labels={"days_on_chart":"Days","song":""})
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(song_kpis_f, x="best_rank", y="days_on_chart",
                           size="avg_popularity", color="album_type",
                           hover_name="song", hover_data=["artist"],
                           color_discrete_map=COLOR_MAP,
                           title="Peak Rank vs Longevity (bubble = avg popularity)",
                           labels={"best_rank":"Best rank","days_on_chart":"Days on chart"})
        fig2.update_xaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

    # Rank heatmap
    st.subheader("🗺️ Rank Heatmap — Top 20 Longest Charting Songs")
    top20 = fdf.groupby("song")["date"].nunique().nlargest(20).index.tolist()
    sub   = fdf[fdf["song"].isin(top20)]
    pivot = sub.pivot_table(index="song", columns="date", values="position", aggfunc="first")
    fig3  = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                       title="Rank heatmap (green=high rank, red=low rank)",
                       labels={"color":"Position"}, aspect="auto")
    st.plotly_chart(fig3, use_container_width=True)

    # Song trend comparison
    st.subheader("📈 Compare Rank Trends")
    trend_songs = st.multiselect("Select up to 8 songs to compare", all_songs, max_selections=8)
    if trend_songs:
        sub2 = fdf[fdf["song"].isin(trend_songs)]
        fig4 = px.line(sub2, x="date", y="position", color="song",
                        title="Rank over time",
                        labels={"position":"Chart position","date":"Date"},
                        color_discrete_sequence=px.colors.qualitative.Bold)
        fig4.update_yaxes(autorange="reversed")
        st.plotly_chart(fig4, use_container_width=True)

    # Full table
    st.subheader("📋 Full Song KPI Table")
    st.dataframe(
        song_kpis_f.sort_values("days_on_chart", ascending=False),
        use_container_width=True,
        column_config={
            "avg_popularity":  st.column_config.ProgressColumn("Avg popularity",  max_value=100),
            "longevity_score": st.column_config.ProgressColumn("Longevity score", max_value=1),
            "rank_volatility": st.column_config.NumberColumn("Rank volatility"),
        }
    )

# ════════════════════════════════════════════════════════════
#  TAB 3 — ARTIST DOMINANCE
# ════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("👤 Artist Dominance Leaderboard")

    top_n = st.slider("Number of top artists to show", 5, 30, 15)
    art_f = get_artist_kpis(fdf).head(top_n)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.bar(art_f, x="dominance_pct", y="artist", orientation="h",
                      title=f"Top {top_n} Artists by Chart Dominance (%)",
                      color="dominance_pct", color_continuous_scale="Blues",
                      labels={"dominance_pct":"% of chart slots","artist":"Artist"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            art_f[["artist","unique_songs","total_days","avg_rank","dominance_pct"]],
            use_container_width=True,
        )

    # Concurrent songs
    st.subheader("🎶 Max Songs Charted on Same Day per Artist")
    overlap = (
        fdf.groupby(["date","artist"])["song"].nunique().reset_index()
        .rename(columns={"song":"songs_on_date"})
    )
    overlap2 = (
        overlap.groupby("artist")["songs_on_date"].max().reset_index()
        .rename(columns={"songs_on_date":"max_concurrent_songs"})
        .sort_values("max_concurrent_songs", ascending=False)
        .head(20).reset_index(drop=True)
    )
    st.dataframe(overlap2, use_container_width=True)

    # Artist timeline
    st.subheader("📆 Artist Daily Rank Timeline")
    drill = st.selectbox("Select an artist to explore", all_artists)
    if drill:
        tl = fdf[fdf["artist"] == drill][["date","song","position","popularity"]].sort_values("date")
        fig2 = px.line(tl, x="date", y="position", color="song",
                        title=f"{drill} — daily rank per song",
                        labels={"position":"Chart rank"})
        fig2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 4 — POPULARITY ANALYTICS
# ════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📊 Popularity Analytics")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(fdf, x="position", y="popularity",
                          color="album_type", hover_name="song",
                          hover_data=["artist","is_explicit"],
                          title="Popularity vs Chart Rank",
                          color_discrete_map=COLOR_MAP,
                          labels={"position":"Rank","popularity":"Popularity"},
                          opacity=0.6)
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.violin(fdf, x="rank_tier", y="popularity", color="rank_tier",
                          box=True, points="outliers",
                          title="Popularity Distribution by Rank Tier",
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          labels={"rank_tier":"Tier","popularity":"Popularity"})
        st.plotly_chart(fig2, use_container_width=True)

    # Daily trend
    st.subheader("📈 Daily Average Popularity Trend")
    pop_trend = fdf.groupby("date")["popularity"].mean().reset_index()
    fig3 = px.line(pop_trend, x="date", y="popularity",
                    title="Daily average popularity (May 2024 – Nov 2025)",
                    labels={"popularity":"Avg popularity"})
    st.plotly_chart(fig3, use_container_width=True)

    # Stats table
    st.subheader("📋 Popularity Stats by Rank Tier")
    st.dataframe(
        fdf.groupby("rank_tier", observed=True)["popularity"].describe().round(2),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════
#  TAB 5 — CONTENT ATTRIBUTES
# ════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🎭 Content Attribute Analysis")

    col1, col2 = st.columns(2)
    with col1:
        # Explicit vs clean bar
        exp_agg = fdf.groupby("is_explicit")[["popularity","position"]].mean().round(2).reset_index()
        exp_agg["label"] = exp_agg["is_explicit"].map({True:"Explicit",False:"Clean"})
        fig = go.Figure([
            go.Bar(name="Avg popularity", x=exp_agg["label"], y=exp_agg["popularity"],
                   marker_color=["#3498db","#e74c3c"])
        ])
        fig.update_layout(title="Explicit vs Clean: Avg Popularity")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart — album types
        counts = fdf.groupby("album_type")["song"].nunique().reset_index()
        fig2 = px.pie(counts, names="album_type", values="song",
                       title="Unique Songs by Album Type",
                       color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Duration correlation
        clean = fdf[["duration_min","popularity"]].dropna()
        corr, pval = stats.pearsonr(clean["duration_min"], clean["popularity"])
        st.metric("Duration vs Popularity Correlation",
                   f"r = {round(corr,4)}", f"p-value = {round(pval,4)}")
        bucket_avg = (
            fdf.groupby("duration_bucket", observed=True)
            .agg(avg_popularity=("popularity","mean"), count=("song","count"))
            .round(2).reset_index()
        )
        fig3 = px.bar(bucket_avg, x="duration_bucket", y="avg_popularity",
                       color="avg_popularity", color_continuous_scale="Viridis",
                       title="Avg Popularity by Song Duration",
                       labels={"duration_bucket":"Duration","avg_popularity":"Avg popularity"},
                       text_auto=".1f")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Total tracks impact
        tmp = fdf.copy()
        tmp["track_bucket"] = pd.cut(
            tmp["total_tracks"],
            bins=[0,1,3,10,20,9999],
            labels=["Single (1)","EP (2-3)","Small LP (4-10)","LP (11-20)","Large LP (20+)"],
        )
        tt = (
            tmp.groupby("track_bucket", observed=True)
            .agg(avg_popularity=("popularity","mean"), count=("song","count"))
            .round(2).reset_index()
        )
        fig4 = px.bar(tt, x="track_bucket", y="avg_popularity",
                       color="count", color_continuous_scale="Teal",
                       title="Avg Popularity by Album Size",
                       labels={"track_bucket":"Album size","avg_popularity":"Avg popularity"},
                       text_auto=".1f")
        st.plotly_chart(fig4, use_container_width=True)

    # Album type bar
    st.subheader("💿 Album Type Performance — album / single / compilation")
    at_agg = fdf.groupby("album_type")[["popularity","position"]].mean().round(2).reset_index()
    fig5 = px.bar(at_agg, x="album_type", y="popularity", color="album_type",
                   color_discrete_map=COLOR_MAP,
                   title="Avg Popularity by Album Type",
                   labels={"album_type":"Album type","popularity":"Avg popularity"},
                   text_auto=".1f")
    st.plotly_chart(fig5, use_container_width=True)

    # Detailed tables
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Explicit vs Clean — Detailed Stats")
        exp_detail = (
            fdf.groupby("is_explicit")
            .agg(song_count=("song","nunique"), avg_popularity=("popularity","mean"),
                 avg_rank=("position","mean"), best_rank=("position","min"))
            .round(2).reset_index()
        )
        exp_detail["label"] = exp_detail["is_explicit"].map({True:"Explicit",False:"Clean"})
        st.dataframe(exp_detail, use_container_width=True)

    with col6:
        st.subheader("Album Type — Detailed Stats")
        at_detail = (
            fdf.groupby("album_type")
            .agg(song_count=("song","nunique"), avg_popularity=("popularity","mean"),
                 avg_rank=("position","mean"), best_rank=("position","min"),
                 total_entries=("position","count"))
            .round(2).reset_index()
        )
        st.dataframe(at_detail, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 6 — ADVANCED EDA
# ════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🔬 Advanced Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        # Correlation matrix
        num_cols = ["position","popularity","duration_min","total_tracks",
                    "days_on_chart","rank_volatility"]
        avail = [c for c in num_cols if c in fdf.columns]
        corr = fdf[avail].corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         title="Feature Correlation Matrix", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Explicit trend over time
        te = fdf.groupby(["date","is_explicit"])["popularity"].mean().reset_index()
        te["type"] = te["is_explicit"].map({True:"Explicit",False:"Clean"})
        fig2 = px.line(te, x="date", y="popularity", color="type",
                        title="Popularity Trend: Explicit vs Clean Over Time",
                        color_discrete_map={"Explicit":"#e74c3c","Clean":"#3498db"})
        st.plotly_chart(fig2, use_container_width=True)

    # Volatility histogram
    st.subheader("📊 Rank Volatility Distribution by Album Type")
    fig3 = px.histogram(song_kpis, x="rank_volatility", nbins=30, color="album_type",
                         color_discrete_map=COLOR_MAP,
                         title="Rank Volatility Index Distribution",
                         barmode="overlay", opacity=0.7)
    st.plotly_chart(fig3, use_container_width=True)

    # 3D scatter
    st.subheader("🌐 3D View — Peak Rank × Popularity × Longevity")
    fig4 = px.scatter_3d(
        song_kpis.head(100),
        x="best_rank", y="avg_popularity", z="days_on_chart",
        color="album_type", hover_name="song", size="rank_volatility",
        color_discrete_map=COLOR_MAP,
        title="Top 100 Songs: Peak Rank × Popularity × Days on Chart",
        labels={"best_rank":"Best rank","avg_popularity":"Avg popularity","days_on_chart":"Days on chart"},
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Top 10 insights summary
    st.subheader("🏆 Top 10 Key Insights from Your Data")
    top_song    = song_kpis.iloc[0]
    top_artist  = art_kpis.iloc[0]
    single_pop  = fdf[fdf["album_type"]=="single"]["popularity"].mean()
    album_pop   = fdf[fdf["album_type"]=="album"]["popularity"].mean()
    explicit_pop= fdf[fdf["is_explicit"]==True]["popularity"].mean()
    clean_pop   = fdf[fdf["is_explicit"]==False]["popularity"].mean()

    insights = [
        f"🥇 **Most longevity:** '{top_song['song']}' by {top_song['artist']} — {top_song['days_on_chart']} days on chart",
        f"👑 **Top artist:** {top_artist['artist']} with {top_artist['unique_songs']} songs and {top_artist['dominance_pct']}% chart dominance",
        f"💿 **Singles vs Albums:** Singles avg popularity {single_pop:.1f} vs Albums {album_pop:.1f}",
        f"🎤 **Explicit vs Clean:** Explicit avg {explicit_pop:.1f} vs Clean avg {clean_pop:.1f}",
        f"📊 **Duration correlation:** r = {round(corr.loc['duration_min','popularity'] if 'duration_min' in corr.columns else 0, 3)} — duration has {'weak' if abs(corr.loc['duration_min','popularity'] if 'duration_min' in corr.columns else 0) < 0.3 else 'strong'} impact on popularity",
        f"📅 **Date range covered:** {df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')} ({df['date'].nunique()} days)",
        f"🎵 **Total unique songs:** {df['song'].nunique()} across {df['artist'].nunique()} artists",
        f"🔥 **Most popular position:** Rank 1 songs average popularity {fdf[fdf['position']==1]['popularity'].mean():.1f}",
        f"📈 **Highest avg popularity tier:** Top 10 songs avg {fdf[fdf['position']<=10]['popularity'].mean():.1f}",
        f"🎭 **Explicit content share:** {fdf['is_explicit'].mean()*100:.1f}% of all chart appearances are explicit",
    ]

    for insight in insights:
        st.markdown(f"- {insight}")

