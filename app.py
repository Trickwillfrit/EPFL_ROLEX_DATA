"""
Data files (loaded from raw GitHub URLs):
    - items_app.csv          enriched book metadata + cover_url (+ optional blurb)
    - submission.csv         per-user top-10 recommendations
    - interactions_train.csv user reading history
"""

from __future__ import annotations

import random

import pandas as pd
import requests
import streamlit as st


# Page config

st.set_page_config(
    page_title="Library Recommendation App",
    page_icon="📚",
    layout="wide",
)


# Data sources

BASE = "https://raw.githubusercontent.com/Trickwillfrit/EPFL_ROLEX_DATA/main"
ITEMS_URL = f"{BASE}/items_app.csv"
SUBMISSION_URL = f"{BASE}/submission.csv"
INTERACTIONS_URL = f"{BASE}/interactions_train.csv"

HISTORY_LIMIT = 10  # how many recent interactions to display

# Minimal, text-free placeholder for books with no cover.
PLACEHOLDER_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 170'>
  <rect x='1' y='1' width='118' height='168'
        fill='#fafafa' stroke='#d0d0d4' stroke-width='1' rx='4'/>
</svg>
""".strip()

BLURB_COLUMNS = ("blurb")

# Text helpers
def clean_title(title) -> str:
    """Clean a book title: strip whitespace and useless trailing slash."""
    if pd.isna(title):
        return ""
    text = str(title).strip()
    while text.endswith("/"):
        text = text[:-1].rstrip()
    return text


def shorten(text, max_len: int = 120) -> str:
    if pd.isna(text):
        return "Not available"
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def split_subjects(raw) -> list[str]:
    """Split a Subjects string into a clean list of individual subjects."""
    if pd.isna(raw):
        return []
    s = str(raw)
    pieces: list[str] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "," in chunk and len(chunk) > 40:
            for sub in chunk.split(","):
                sub = sub.strip()
                if sub:
                    pieces.append(sub)
        else:
            pieces.append(chunk)
    seen, out = set(), []
    for p in pieces:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def has_cover(cover_url) -> bool:
    """True iff cover_url is a non-empty string."""
    return (
        pd.notna(cover_url)
        and isinstance(cover_url, str)
        and bool(cover_url.strip())
    )


def get_book_blurb(book: pd.Series) -> str:
    """Return the first non-empty blurb/description from supported columns."""
    for col in BLURB_COLUMNS:
        if col in book.index:
            val = book.get(col)
            if pd.notna(val) and isinstance(val, str) and val.strip():
                return val.strip()
    return ""



# Data loading

@st.cache_data(show_spinner="Loading data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items = pd.read_csv(ITEMS_URL)
    submission = pd.read_csv(SUBMISSION_URL)
    interactions = pd.read_csv(INTERACTIONS_URL)

    if "cover_url" not in items.columns:
        items["cover_url"] = pd.NA
    for col in ("Title", "Author", "Publisher", "Subjects"):
        if col not in items.columns:
            items[col] = pd.NA

    return items, submission, interactions


@st.cache_data(show_spinner=False, max_entries=2000)
def cover_is_reachable(url: str) -> bool:
    """HEAD-check a cover URL once per session. Avoids broken-image icons."""
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        r = requests.head(url, timeout=4, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False



# Display helpers

def render_cover(cover_url, width: int = 120) -> None:
    """Render the cover image, falling back to a placeholder if missing."""
    if has_cover(cover_url) and cover_is_reachable(cover_url):
        st.image(cover_url, width=width)
        return
    st.markdown(
        f"<div style='width:{width}px'>{PLACEHOLDER_SVG}</div>",
        unsafe_allow_html=True,
    )


def build_explanation(book: pd.Series,
                      hist_authors: set,
                      hist_publishers: set,
                      hist_subjects_low: set) -> str:
    """Return a short 'why it may fit' string, or empty string if none."""
    subjects = split_subjects(book.get("Subjects"))
    shared = [s for s in subjects if s.lower() in hist_subjects_low]
    if shared:
        return f"Why it may fit: shares subject(s): {', '.join(shared[:3])}"

    author = book.get("Author")
    if pd.notna(author) and str(author).strip() in hist_authors:
        return "Why it may fit: author already appears in this user's history"

    publisher = book.get("Publisher")
    if pd.notna(publisher) and str(publisher).strip() in hist_publishers:
        return "Why it may fit: same publisher as previous reading"

    return ""


def select_book(book_id: int) -> None:
    """Callback: store the clicked book id in session state."""
    st.session_state["selected_book_id"] = int(book_id)


def display_book_card(book: pd.Series, *,
                      context: str,
                      rank: int | None = None,
                      interaction_date: pd.Timestamp | None = None,
                      explanation: str = "") -> None:
    """Render one book card with a clickable title that opens a detail panel.

    `context` is a short tag ('hist' or 'rec') so the button keys stay unique
    even when the same book appears on both sides. We also append the row's
    pandas index (`book.name`) so duplicates within one side can never collide.
    """
    img_col, text_col = st.columns([1, 4])

    with img_col:
        render_cover(book.get("cover_url"))

    with text_col:
        title = clean_title(book.get("Title")) or f"Unknown item (ID {book['i']})"
        author = book["Author"] if pd.notna(book.get("Author")) else "Unknown author"
        publisher = book["Publisher"] if pd.notna(book.get("Publisher")) else "Unknown publisher"
        subjects_raw = book.get("Subjects")
        subjects_preview = shorten(subjects_raw, 120) if pd.notna(subjects_raw) else "No subjects"

        label = f"#{rank} — {shorten(title, 90)}" if rank is not None else shorten(title, 100)

        # Button key: context + book id + row index = guaranteed unique
        st.button(
            label,
            key=f"open_{context}_{book['i']}_{book.name}",
            on_click=select_book,
            args=(book["i"],),
            use_container_width=True,
        )

        st.markdown(f"**Author:** {author}  \n**Publisher:** {publisher}")
        st.caption(f"Subjects: {subjects_preview}")

        if interaction_date is not None and pd.notna(interaction_date):
            st.caption(f"Read on {interaction_date.strftime('%Y-%m-%d')}")
        if explanation:
            st.caption(f"💡 {explanation}")

    st.markdown("---")


def display_book_details(book: pd.Series) -> None:
    """Render the full detailed view of one book."""
    title = clean_title(book.get("Title")) or f"Unknown item (ID {book.get('i')})"

    img_col, text_col = st.columns([1, 3])

    with img_col:
        render_cover(book.get("cover_url"), width=180)

    with text_col:
        st.markdown(f"### {title}")

        fields = [
            ("Item ID", book.get("i")),
            ("Author", book.get("Author")),
            ("Publisher", book.get("Publisher")),
            ("ISBN Valid", book.get("ISBN Valid")),
            ("ISBN (clean)", book.get("isbn_clean") if "isbn_clean" in book.index else None),
        ]
        for label, value in fields:
            if pd.notna(value) and str(value).strip():
                st.markdown(f"**{label}:** {value}")

        subjects = book.get("Subjects")
        if pd.notna(subjects) and str(subjects).strip():
            st.markdown("**Subjects:**")
            subj_list = split_subjects(subjects)
            if subj_list:
                st.markdown(", ".join(subj_list))
            else:
                st.markdown(str(subjects))

        cover_url = book.get("cover_url")
        if has_cover(cover_url):
            st.markdown(f"**Cover URL:** [{cover_url}]({cover_url})")

    st.markdown("---")
    st.markdown("**Blurb**")
    blurb = get_book_blurb(book)
    if blurb:
        st.write(blurb)
    else:
        st.caption("No blurb available.")



# Filter logic

def apply_filters(df: pd.DataFrame,
                  title_query: str,
                  authors: list[str],
                  publishers: list[str],
                  subjects: list[str],
                  only_with_cover: bool) -> pd.DataFrame:
    """Apply the sidebar book filters to a dataframe of books."""
    if df.empty:
        return df
    out = df.copy()

    if title_query:
        q = title_query.lower().strip()
        cleaned = out["Title"].apply(clean_title).str.lower()
        out = out[cleaned.str.contains(q, na=False, regex=False)]

    if authors:
        out = out[out["Author"].isin(authors)]

    if publishers:
        out = out[out["Publisher"].isin(publishers)]

    if subjects:
        selected_low = {s.lower() for s in subjects}

        def has_any_subject(raw) -> bool:
            for s in split_subjects(raw):
                if s.lower() in selected_low:
                    return True
            return False

        out = out[out["Subjects"].apply(has_any_subject)]

    if only_with_cover:
        out = out[out["cover_url"].apply(has_cover)]

    return out



# Load

items, submission, interactions = load_data()


# Title

st.title("📚 Library Recommendation App")
st.write(
    "Browse a user's recent reading history alongside their top-10 "
    "recommended books. Click on any book title to see full details."
)


# Sidebar: settings

st.sidebar.header("Settings")

user_list = sorted(submission["user_id"].unique().tolist())
st.sidebar.write(f"**{len(user_list):,}** users available.")

mode = st.sidebar.radio(
    "Pick a user by:",
    options=["Dropdown", "ID input", "Random"],
    index=0,
)

if mode == "Dropdown":
    user_id = st.sidebar.selectbox("User ID", user_list)
elif mode == "ID input":
    user_id = st.sidebar.number_input(
        "User ID",
        min_value=int(min(user_list)),
        max_value=int(max(user_list)),
        value=int(user_list[0]),
        step=1,
    )
    if user_id not in user_list:
        st.sidebar.warning(f"User {user_id} is not in the submission file.")
else:  # Random
    if "random_user" not in st.session_state:
        st.session_state.random_user = user_list[0]
    if st.sidebar.button("🎲 Pick another random user"):
        st.session_state.random_user = random.choice(user_list)
        st.session_state.pop("selected_book_id", None)
    user_id = st.session_state.random_user
    st.sidebar.info(f"Random user: **{user_id}**")

# Clear selected book if the user changed (dropdown / ID input)
if st.session_state.get("active_user_id") != user_id:
    st.session_state["active_user_id"] = user_id
    st.session_state.pop("selected_book_id", None)


# Compute user history + recommendations BEFORE filters
# Reading history: deduplicate by book id, keeping the most recent interaction.
# This avoids showing the same book multiple times in the history column.
user_history = (
    interactions[interactions["u"] == user_id]
    .merge(items, on="i", how="left")
    .assign(datetime=lambda d: pd.to_datetime(d["t"], unit="s", errors="coerce"))
    .sort_values("datetime", ascending=False)
    .drop_duplicates(subset="i", keep="first")  # keep most recent read per book
    .reset_index(drop=True)
)

sub_row = submission.loc[submission["user_id"] == user_id, "recommendation"]
if sub_row.empty:
    st.error(f"No recommendation found for user {user_id} in submission.csv.")
    st.stop()

rec_string = sub_row.iloc[0]
rec_ids: list[int] = []
for x in str(rec_string).split():
    try:
        rec_ids.append(int(x))
    except ValueError:
        continue

rec_books = (
    pd.DataFrame({"i": rec_ids, "rank": range(1, len(rec_ids) + 1)})
    .merge(items, on="i", how="left")
    .reset_index(drop=True)
)

history_ids = set(user_history["i"].tolist())

# Sidebar: book filters

st.sidebar.header("Book filters")

union_df = pd.concat([user_history, rec_books], ignore_index=True)

authors_pool = sorted(
    {str(a).strip() for a in union_df["Author"].dropna() if str(a).strip()}
)
publishers_pool = sorted(
    {str(p).strip() for p in union_df["Publisher"].dropna() if str(p).strip()}
)
subjects_pool_set: set[str] = set()
for raw in union_df["Subjects"].dropna():
    subjects_pool_set.update(split_subjects(raw))
subjects_pool = sorted(subjects_pool_set, key=str.lower)

title_query = st.sidebar.text_input("Search by title")
selected_authors = st.sidebar.multiselect("Author", authors_pool)
selected_publishers = st.sidebar.multiselect("Publisher", publishers_pool)
selected_subjects = st.sidebar.multiselect("Subject", subjects_pool)
only_with_cover = st.sidebar.checkbox("Only show books with cover", value=False)
hide_already_read = st.sidebar.checkbox("Hide already-read recommendations", value=True)

st.sidebar.caption(
    "Filters apply to both history and recommendations, except "
    "*hide already-read*, which only affects recommendations."
)


# Apply filters

filtered_history = apply_filters(
    user_history, title_query, selected_authors,
    selected_publishers, selected_subjects, only_with_cover,
)

filtered_recs = apply_filters(
    rec_books, title_query, selected_authors,
    selected_publishers, selected_subjects, only_with_cover,
)

if hide_already_read and not filtered_recs.empty:
    filtered_recs = filtered_recs[~filtered_recs["i"].isin(history_ids)]

filters_active = bool(
    title_query or selected_authors or selected_publishers
    or selected_subjects or only_with_cover
)


# Header

st.subheader(f"User #{user_id}")

n_history_total = len(user_history)  # number of distinct books read

if filters_active:
    st.caption("ℹ️ Filters are applied to both history and recommendations.")


# Selected book detail panel (shown when a title was clicked)

selected_id = st.session_state.get("selected_book_id")
if selected_id is not None:
    match = items[items["i"] == selected_id]
    if not match.empty:
        with st.container():
            top_l, top_r = st.columns([6, 1])
            with top_l:
                st.markdown("### 📖 Selected book details")
            with top_r:
                if st.button("✖ Close", key="close_details"):
                    st.session_state.pop("selected_book_id", None)
                    st.rerun()
            display_book_details(match.iloc[0])
        st.markdown("---")


# Precompute "why it may fit" context from the FULL (unfiltered) history

hist_authors = {
    str(a).strip() for a in user_history["Author"].dropna() if str(a).strip()
}
hist_publishers = {
    str(p).strip() for p in user_history["Publisher"].dropna() if str(p).strip()
}
hist_subjects_low: set[str] = set()
for raw in user_history["Subjects"].dropna():
    hist_subjects_low.update(s.lower() for s in split_subjects(raw))


# Two-column layout

left, right = st.columns(2)

with left:
    st.subheader("Recent reading history")

    if n_history_total == 0:
        st.info("No reading history found for this user.")
    elif filtered_history.empty:
        st.info("No history items match the current filters.")
    else:
        history_to_show = filtered_history.head(HISTORY_LIMIT)
        for _, book in history_to_show.iterrows():
            display_book_card(
                book,
                context="hist",
                interaction_date=book["datetime"],
            )

        if filters_active:
            st.caption(
                f"Showing {len(history_to_show)} filtered interactions "
                f"from {n_history_total} total interactions."
            )
        elif n_history_total > HISTORY_LIMIT:
            st.caption(
                f"Showing {HISTORY_LIMIT} of {n_history_total} interactions "
                "before filters."
            )

with right:
    st.subheader("Top-10 recommendations")

    if filtered_recs.empty:
        st.info("No recommendations match the current filters.")
    else:
        for _, book in filtered_recs.iterrows():
            explanation = build_explanation(
                book, hist_authors, hist_publishers, hist_subjects_low
            )
            display_book_card(
                book,
                context="rec",
                rank=int(book["rank"]),
                explanation=explanation,
            )

        if filters_active or hide_already_read:
            st.caption(
                f"Showing {len(filtered_recs)} of {len(rec_books)} "
                "recommendations after filters."
            )

# Footer

st.markdown("---")
st.caption(
    "EPFL recommendation systems project. "
    "Recommendations are precomputed; this app only displays them."
)