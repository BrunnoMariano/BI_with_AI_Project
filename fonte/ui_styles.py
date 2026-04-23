import streamlit as st


GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --app-bg: #07090d;
    --surface-0: rgba(17, 22, 29, 0.72);
    --surface-1: rgba(20, 25, 33, 0.82);
    --surface-2: rgba(28, 34, 44, 0.88);
    --surface-strong: rgba(32, 39, 51, 0.94);
    --border-soft: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.14);
    --text-primary: #f5f7fb;
    --text-secondary: #b6c0cf;
    --text-muted: #8490a3;
    --accent: #d8dee8;
    --accent-soft: rgba(216, 222, 232, 0.14);
    --success: #9ce8c3;
    --warning: #ffd59e;
    --danger: #ffb2b2;
    --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.30);
    --shadow-md: 0 16px 40px rgba(0, 0, 0, 0.22);
    --radius-xl: 28px;
    --radius-lg: 22px;
    --radius-md: 16px;
    --radius-sm: 12px;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
}

body {
    color: var(--text-primary);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(127, 147, 173, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 255, 255, 0.07), transparent 22%),
        linear-gradient(180deg, #0c1016 0%, #090c11 50%, #07090d 100%);
    color: var(--text-primary);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.04), transparent 32%),
        radial-gradient(circle at 15% 20%, rgba(255, 255, 255, 0.06), transparent 18%);
    opacity: 0.9;
}

footer {visibility: hidden;}
.stDeployButton {display:none;}
#MainMenu {visibility: hidden;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stToolbarActions"] {display: none !important;}
.stAppToolbar {
    visibility: visible !important;
    pointer-events: auto !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

[data-testid="stHeader"] {
    background: rgba(7, 9, 13, 0.36);
    backdrop-filter: blur(18px);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(12, 15, 20, 0.95), rgba(14, 18, 24, 0.84)),
        rgba(12, 15, 20, 0.88);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.03);
}

[data-testid="stSidebar"] > div:first-child {
    backdrop-filter: blur(26px);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1440px;
}

.ui-centered-hero {
    max-width: 46rem;
    margin: 0 auto 1.4rem;
}

h1, h2, h3, h4 {
    color: var(--text-primary);
    letter-spacing: -0.03em;
}

h1 {
    font-weight: 800 !important;
    font-size: clamp(2.2rem, 3vw, 3.4rem) !important;
    margin-bottom: 0.75rem !important;
}

h2 {
    font-weight: 700 !important;
    font-size: clamp(1.45rem, 1.5vw, 2rem) !important;
}

h3 {
    font-weight: 700 !important;
}

p, label, span, div, small {
    color: inherit;
}

[data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

.ui-shell-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.78rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

.ui-shell-title {
    margin: 0.3rem 0 0.2rem;
    color: var(--text-primary);
    font-size: clamp(1.8rem, 2.4vw, 2.65rem);
    font-weight: 800;
    line-height: 1.05;
}

.ui-shell-subtitle {
    max-width: 840px;
    margin: 0;
    color: var(--text-secondary);
    font-size: 0.98rem;
    line-height: 1.65;
}

.ui-section-label {
    margin: 0 0 0.35rem;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.ui-section-title {
    margin: 0;
    color: var(--text-primary);
    font-size: 1.45rem;
    font-weight: 750;
}

.ui-section-text {
    margin: 0.4rem 0 1.1rem;
    color: var(--text-secondary);
    font-size: 0.98rem;
    line-height: 1.6;
}

[data-testid="stTabs"] {
    gap: 0.45rem;
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 0.55rem;
    padding: 0.35rem;
    margin-bottom: 1.25rem;
    border: 1px solid var(--border-soft);
    border-radius: 999px;
    background: rgba(18, 24, 31, 0.7);
    backdrop-filter: blur(18px);
}

[data-testid="stTabs"] [role="tab"] {
    height: auto;
    padding: 0.72rem 1.05rem;
    border-radius: 999px;
    color: var(--text-secondary);
    font-weight: 600;
    transition: background 180ms ease, color 180ms ease, transform 180ms ease;
}

[data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-primary);
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(240, 244, 250, 0.16), rgba(255, 255, 255, 0.06));
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

[data-testid="stForm"],
[data-testid="stAlert"],
[data-testid="stChatMessage"],
[data-testid="stMetric"],
.stDataFrame,
[data-testid="stExpander"],
.element-container div:has(> [data-testid="stPlotlyChart"]) {
    border-radius: var(--radius-lg);
}

[data-testid="stForm"],
[data-testid="stAlert"],
[data-testid="stExpander"],
.element-container div:has(> [data-testid="stPlotlyChart"]) {
    border: 1px solid var(--border-soft);
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015)),
        var(--surface-0);
    box-shadow: var(--shadow-md);
    backdrop-filter: blur(18px);
}

[data-testid="stForm"] {
    padding: 1.1rem 1rem 0.85rem;
}

/* Streamlit custom components are mounted inside an iframe. The dashboard grid
   already renders its own rounded shell, so the host iframe must stay visually invisible. */
.element-container:has(> iframe[data-testid="stCustomComponentV1"]),
.element-container div:has(> iframe[data-testid="stCustomComponentV1"]),
[data-testid="stElementContainer"]:has(iframe[data-testid="stCustomComponentV1"]) {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    overflow: visible !important;
}

iframe[data-testid="stCustomComponentV1"],
.stCustomComponentV1 {
    display: block !important;
    width: 100% !important;
    border: 0 !important;
    border-radius: var(--radius-xl) !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
    clip-path: inset(0 round var(--radius-xl));
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius-lg);
}

[data-testid="stAlert"] {
    padding: 0.9rem 1rem;
}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary);
}

[data-testid="stSpinner"] {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
}

[data-testid="stSpinner"] > div:first-child,
[data-testid="stSpinner"] svg {
    display: none !important;
}

[data-testid="stSpinner"]::before {
    content: "";
    width: 1rem;
    height: 1rem;
    border-radius: 999px;
    border: 2px solid rgba(255, 255, 255, 0.18);
    border-top-color: rgba(255, 255, 255, 0.92);
    animation: ui-spin 0.8s linear infinite;
    flex: 0 0 auto;
}

@keyframes ui-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

[data-baseweb="input"],
[data-baseweb="base-input"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

[data-baseweb="input"] > div,
[data-baseweb="base-input"] > div,
div[data-baseweb="select"] > div,
[data-testid="stFileUploader"] section {
    border-radius: var(--radius-md) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    background: rgba(15, 19, 26, 0.82) !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease, background 180ms ease;
}

[data-baseweb="input"] input,
[data-baseweb="base-input"] input,
[data-baseweb="input"] textarea,
[data-baseweb="base-input"] textarea {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0 !important;
}

[data-baseweb="input"] > div:focus-within,
[data-baseweb="base-input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(255, 255, 255, 0.92) !important;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.92) !important;
}

[data-baseweb="input"]:focus-within,
[data-baseweb="base-input"]:focus-within {
    border: 0 !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-baseweb="input"] > div:hover,
[data-baseweb="base-input"] > div:hover,
div[data-baseweb="select"] > div:hover {
    border-color: rgba(255, 255, 255, 0.14) !important;
}

input, textarea {
    color: var(--text-primary) !important;
}

input:focus,
input:focus-visible,
textarea:focus,
textarea:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}

input:invalid,
input:focus:invalid,
input:focus-visible:invalid,
textarea:invalid,
textarea:focus:invalid,
textarea:focus-visible:invalid {
    box-shadow: none !important;
    outline: none !important;
}

input[aria-invalid="true"],
textarea[aria-invalid="true"] {
    box-shadow: none !important;
}

[data-baseweb="input"] input:invalid,
[data-baseweb="base-input"] input:invalid,
[data-baseweb="input"] input:focus:invalid,
[data-baseweb="base-input"] input:focus:invalid {
    border: 0 !important;
    box-shadow: none !important;
}

label, .stRadio label, .stSelectbox label, .stTextInput label, .stFileUploader label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button {
    min-height: 2.9rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background:
        linear-gradient(180deg, rgba(237, 240, 245, 0.12), rgba(165, 174, 187, 0.07)),
        rgba(30, 36, 46, 0.9);
    color: var(--text-primary);
    font-weight: 700;
    letter-spacing: 0.01em;
    transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, border-color 180ms ease;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.22);
}

[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    border-color: rgba(255, 255, 255, 0.16);
    background:
        linear-gradient(180deg, rgba(247, 248, 250, 0.17), rgba(184, 193, 205, 0.10)),
        rgba(34, 41, 53, 0.95);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
}

[data-testid="stButton"] button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background:
        linear-gradient(180deg, rgba(242, 245, 250, 0.18), rgba(193, 202, 214, 0.10)),
        rgba(42, 49, 63, 0.98);
}

[data-testid="stChatMessage"] {
    margin-bottom: 0.95rem;
    padding: 1rem 1rem 0.85rem;
    border: 1px solid var(--border-soft);
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02)),
        rgba(15, 19, 27, 0.72);
    box-shadow: var(--shadow-md);
    backdrop-filter: blur(18px);
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    color: var(--text-secondary);
    line-height: 1.7;
}

.ui-chat-composer-spacer {
    height: 0.35rem;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) form {
    padding: 0 !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 0.45rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 999px !important;
    background: rgba(46, 46, 46, 0.96) !important;
    min-height: 3.4rem !important;
    padding: 0.28rem 0.34rem 0.28rem 0.95rem !important;
    box-shadow: none !important;
    overflow: hidden !important;
    transition: border-color 180ms ease, box-shadow 180ms ease !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-testid="stHorizontalBlock"]:focus-within {
    border-color: rgba(255, 255, 255, 0.92) !important;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.92) !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-testid="column"] {
    padding: 0 !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-baseweb="input"],
div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-baseweb="base-input"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-baseweb="input"] > div,
div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-baseweb="base-input"] > div {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: 0 !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) input {
    height: auto !important;
    padding: 0.55rem 0 !important;
    margin: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: none !important;
    color: #f5f7fb !important;
    font-size: 1rem !important;
    line-height: 1.35 !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) input::placeholder {
    color: rgba(255, 255, 255, 0.62) !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: flex-end !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-testid="stFormSubmitButton"] button {
    width: 2rem !important;
    height: 2rem !important;
    min-width: 2rem !important;
    min-height: 2rem !important;
    border-radius: 999px !important;
    border: 0 !important;
    background: rgba(255, 255, 255, 0.98) !important;
    color: #111317 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: 1rem !important;
    line-height: 1 !important;
}

div[data-testid="stForm"]:has(input[aria-label="Pergunte algo sobre seus dados..."]) [data-testid="stFormSubmitButton"] button:hover {
    transform: none !important;
    background: #ffffff !important;
    box-shadow: none !important;
}

[data-testid="stExpander"] details summary {
    padding: 0.9rem 1rem;
    color: var(--text-primary);
    font-weight: 600;
}

[data-testid="stExpanderDetails"] {
    padding: 0 1rem 1rem;
}

[data-testid="stFileUploader"] section {
    padding: 1rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

[data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.65rem;
}

[data-testid="stRadio"] [role="radiogroup"] > label {
    padding: 0.75rem 0.9rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-md);
    background: rgba(255, 255, 255, 0.03);
    width: 100%;
    min-height: 3.25rem;
    box-sizing: border-box;
    display: flex !important;
    align-items: center;
}

[data-testid="stSidebar"] div[data-testid="stRadio"]:has(input[name="sidebar_control_menu"]) {
    background: transparent !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type [role="radiogroup"] {
    gap: 0.28rem !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 0.35rem !important;
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type [role="radiogroup"] > label {
    min-height: 3rem !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type [role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.04) !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"]:first-of-type [role="radiogroup"] > label:has(input:checked) {
    background: rgba(229, 235, 242, 0.12) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06) !important;
}

[data-testid="stMetric"] {
    padding: 1rem;
    border: 1px solid var(--border-soft);
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015)),
        var(--surface-0);
    box-shadow: var(--shadow-md);
    backdrop-filter: blur(18px);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--text-primary);
}

.stDataFrame {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.stDataFrame [data-testid="stTable"] {
    background: rgba(15, 19, 26, 0.46) !important;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: var(--radius-lg);
}

[data-testid="stPlotlyChart"] > div {
    border-radius: var(--radius-lg);
    overflow: hidden;
}

.st-emotion-cache-eczf16,
.st-emotion-cache-1r6slb0 {
    border-radius: var(--radius-lg);
}

hr {
    border-color: rgba(255, 255, 255, 0.06);
}

.ui-soft-spacer {
    height: 0.35rem;
}
</style>
"""


def inject_global_styles():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_app_hero(title: str, subtitle: str | None = None, kicker: str | None = None):
    kicker_html = f'<div class="ui-shell-kicker">{kicker}</div>' if kicker else ""
    subtitle_html = f'<p class="ui-shell-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"<section class='ui-centered-hero'>{kicker_html}<h1 class='ui-shell-title'>{title}</h1>{subtitle_html}</section>",
        unsafe_allow_html=True,
    )


def render_section_intro(label: str, title: str, text: str):
    label_html = f'<p class="ui-section-label">{label}</p>' if label else ""
    title_html = f'<h2 class="ui-section-title">{title}</h2>' if title else ""
    text_html = f'<p class="ui-section-text">{text}</p>' if text else ""
    st.markdown(
        f"<div>{label_html}{title_html}{text_html}</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_brand():
    return None
