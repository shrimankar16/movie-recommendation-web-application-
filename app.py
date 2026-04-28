import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO


st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --bg:        #0a0a0f;
        --surface:   #12121a;
        --card:      #1a1a26;
        --card-hover:#22223a;
        --gold:      #f5c518;
        --gold-dim:  #c9a11a;
        --accent:    #e84393;
        --text:      #e8e8f0;
        --muted:     #8888aa;
        --border:    #2a2a40;
        --radius:    14px;
    }

    /* ── Global reset ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"]          { display: none !important; }
    [data-testid="stSidebar"]          { background: var(--surface) !important; }
    [data-testid="stMainBlockContainer"]{ padding: 2rem 3rem !important; }

    /* ── Hero title ── */
    .hero {
        text-align: center;
        padding: 3.5rem 1rem 2rem;
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #f5c518 0%, #e84393 60%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 .4rem;
        line-height: 1.1;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        font-weight: 300;
        margin: 0;
    }

    /* ── Divider ── */
    .gold-divider {
        width: 80px; height: 3px;
        background: linear-gradient(90deg, var(--gold), var(--accent));
        border-radius: 2px;
        margin: 1rem auto 2.5rem;
    }

    /* ── Search container ── */
    .search-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.6rem 2rem;
        max-width: 780px;
        margin: 0 auto 2.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,.5);
    }
    .search-box label {
        font-size: .85rem !important;
        font-weight: 500 !important;
        color: var(--muted) !important;
        letter-spacing: .05em !important;
        text-transform: uppercase !important;
    }

    /* ── Streamlit selectbox & button overrides ── */
    [data-testid="stSelectbox"] > div > div {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }
    [data-testid="stSelectbox"] svg { fill: var(--gold) !important; }

    div.stButton > button {
        width: 100%;
        margin-top: 1.1rem;
        padding: .85rem 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: .08em;
        color: #0a0a0f !important;
        background: linear-gradient(135deg, #f5c518, #e8a210) !important;
        border: none !important;
        border-radius: 10px !important;
        cursor: pointer;
        transition: filter .2s, transform .2s;
    }
    div.stButton > button:hover {
        filter: brightness(1.12) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Section heading ── */
    .section-heading {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 1.2rem;
        padding-bottom: .5rem;
        border-bottom: 1px solid var(--border);
    }
    .section-heading span {
        color: var(--gold);
    }

    /* ── Movie card ── */
    .movie-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.3rem 1.4rem;
        height: 100%;
        min-height: 170px;
        transition: background .22s, border-color .22s, transform .22s, box-shadow .22s;
        position: relative;
        overflow: hidden;
    }
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--gold), var(--accent));
        opacity: 0;
        transition: opacity .22s;
    }
    .movie-card:hover {
        background: var(--card-hover) !important;
        border-color: #3a3a58 !important;
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(0,0,0,.55);
    }
    .movie-card:hover::before { opacity: 1; }

    .card-rank {
        font-size: .72rem;
        font-weight: 500;
        color: var(--gold);
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: .5rem;
        line-height: 1.3;
    }
    .card-meta {
        font-size: .78rem;
        color: var(--accent);
        font-weight: 500;
        margin-bottom: .55rem;
        letter-spacing: .04em;
    }
    .card-overview {
        font-size: .82rem;
        color: var(--muted);
        line-height: 1.55;
    }

    /* ── Warning / info ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] { color: var(--gold) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar            { width: 6px; }
    ::-webkit-scrollbar-track      { background: var(--bg); }
    ::-webkit-scrollbar-thumb      { background: #2a2a40; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover{ background: var(--gold); }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2.5rem 0 1rem;
        color: #444466;
        font-size: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


RAW_URL = (
    "https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/"
    "master/movie_dataset.csv"
)

@st.cache_data(show_spinner=False)
def load_and_preprocess() -> pd.DataFrame:
    """Download dataset, clean it, and return a tidy DataFrame."""
    try:
        resp = requests.get(RAW_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
    except Exception:
        # Fallback: try direct read (works if running locally with file present)
        df = pd.read_csv("movie_dataset.csv")

    # Keep only the columns we need; handle missing column names gracefully
    desired = ["title", "keywords", "cast", "genres", "director", "overview"]
    available = [c for c in desired if c in df.columns]
    df = df[available].copy()

    # Fill NaN with empty string in every text column
    for col in available:
        df[col] = df[col].fillna("").astype(str)

    # ── Text normalization helper ──
    def clean(text: str) -> str:
        """Lowercase + collapse whitespace."""
        return " ".join(text.lower().split())

    # Cast & keywords: collapse multi-word tokens (e.g. "Sam Neill" → "samneill")
    def squash(text: str) -> str:
        return " ".join(w.replace(" ", "") for w in text.split(","))

    for col in ["cast", "keywords", "genres"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: squash(clean(x)))

    if "director" in df.columns:
        df["director"] = df["director"].apply(
            lambda x: (x.replace(" ", "").lower() + " ") * 3  # boost director weight
        )

    if "overview" in df.columns:
        df["overview"] = df["overview"].apply(clean)

    # Build the combined feature bag
    feature_cols = [c for c in ["keywords", "cast", "genres", "director", "overview"]
                    if c in df.columns]
    df["soup"] = df[feature_cols].apply(lambda row: " ".join(row.values), axis=1)

    # Drop duplicates on title
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def build_similarity(_df: pd.DataFrame):
    """Vectorise the 'soup' column and compute the cosine similarity matrix."""
    cv = CountVectorizer(stop_words="english", max_features=12_000)
    matrix = cv.fit_transform(_df["soup"])
    sim = cosine_similarity(matrix, matrix)
    return sim



def recommend(movie_title: str, df: pd.DataFrame, sim: np.ndarray, n: int = 10):
    """
    Return top-n similar movies for *movie_title*.
    Returns a list of dicts or raises ValueError if movie not found.
    """
    title_lower = movie_title.strip().lower()
    idx_series = df.index[df["title"].str.lower() == title_lower]

    if idx_series.empty:
        raise ValueError(f"'{movie_title}' not found in the dataset.")

    idx = idx_series[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # Skip index 0 → that's the movie itself
    top = [s for s in scores if s[0] != idx][:n]

    results = []
    for rank, (i, score) in enumerate(top, start=1):
        row = df.iloc[i]
        genres_raw = row.get("genres", "")
        # Reverse squash for display: separate by space → title case
        genres_display = ", ".join(
            w.title() for w in genres_raw.split()
        ) if genres_raw else "—"

        overview = row.get("overview", "")
        if len(overview) > 200:
            overview = overview[:197] + "…"

        results.append({
            "rank":     rank,
            "title":    row["title"],
            "genres":   genres_display,
            "overview": overview or "No overview available.",
        })
    return results




def render_card(movie: dict) -> str:
    return f"""
    <div class="movie-card">
        <div class="card-rank">#{movie['rank']} match</div>
        <div class="card-title">{movie['title']}</div>
        <div class="card-meta">{movie['genres']}</div>
        <div class="card-overview">{movie['overview']}</div>
    </div>
    """




def main():
    # ── Hero ──
    st.markdown(
        """
        <div class="hero">
            <h1>🎬 CineMatch</h1>
            <p>Discover films you'll love — powered by content-based intelligence</p>
        </div>
        <div class="gold-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ──
    with st.spinner("Loading movie database…"):
        df = load_and_preprocess()

    with st.spinner("Building similarity engine…"):
        sim = build_similarity(df)

    movie_titles = sorted(df["title"].tolist())

    # ── Search box ──
    st.markdown('<div class="search-box">', unsafe_allow_html=True)

    selected = st.selectbox(
        "SEARCH FOR A MOVIE",
        options=[""] + movie_titles,
        format_func=lambda x: "— Type to search a movie title —" if x == "" else x,
        key="movie_select",
    )

    run = st.button("✦  Get Recommendations", key="recommend_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Results ──
    if run:
        if not selected:
            st.warning("⚠️  Please select a movie before hitting Recommend.")
            return

        with st.spinner(f"Finding movies similar to **{selected}**…"):
            try:
                results = recommend(selected, df, sim, n=10)
            except ValueError as e:
                st.error(f"🚫  {e}")
                return

        st.markdown(
            f'<div class="section-heading">Films similar to '
            f'<span>"{selected}"</span></div>',
            unsafe_allow_html=True,
        )

        # ── Render 2-column card grid ──
        cols_per_row = 2
        for row_start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for col_idx, movie in enumerate(results[row_start: row_start + cols_per_row]):
                with cols[col_idx]:
                    st.markdown(render_card(movie), unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:.9rem'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">CineMatch · Content-Based Movie Recommender '
        '· Built with Streamlit & scikit-learn</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
