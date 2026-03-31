import time, requests, pandas as pd, streamlit as st
from datetime import datetime
from tyche.config import BINANCE_API_KEY, BINANCE_SECRET, INITIAL_CASH
API = "http://localhost:5000"

st.set_page_config(
    page_title="Tyche — HFT Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@800&display=swap');

html, body, [data-testid="stApp"] {
    background: #090b0e !important;
    color: #c8d3dc !important;
    font-family: 'JetBrains Mono', monospace !important;
}
h1,h2,h3,p,label,div { color: #c8d3dc !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #00e5ff !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #5c7a8a !important; letter-spacing: 0.1em; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Update these in your <style> block */
.pnl-pos { font-size:3.5rem; font-weight:700; color:#00e676 !important; line-height:1.1; }
.pnl-neg { font-size:3.5rem; font-weight:700; color:#ff1744 !important; line-height:1.1; }

.title  { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
           color:#00e5ff; letter-spacing:0.3em; margin-bottom:0; }
.sub    { font-size:0.7rem; color:#3a5566; letter-spacing:0.15em; margin-bottom:1rem; }

.pnl-pos { font-size:3.5rem; font-weight:700; color:#00e676; line-height:1.1; }
.pnl-neg { font-size:3.5rem; font-weight:700; color:#ff1744; line-height:1.1; }
.pnl-sub { font-size:0.8rem; color:#5c7a8a; margin-top:0.2rem; }

.section { font-size:0.65rem; color:#3a5566; letter-spacing:0.2em;
           border-bottom:1px solid #1a2433; padding-bottom:4px; margin-bottom:8px; }

.asset-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:12px; }
.asset-card { background:#0d1117; border:1px solid #1a2433; border-radius:6px;
              padding:10px 6px; text-align:center; }
.asset-sym  { font-size:0.65rem; color:#5c7a8a; }
.asset-px   { font-size:1rem; font-weight:700; color:#c8d3dc; }

.tbl-hdr { display:grid; grid-template-columns:0.6fr 0.9fr 1fr 1.2fr 1fr 0.8fr 0.7fr;
           padding:4px 6px; font-size:0.6rem; color:#3a5566; letter-spacing:0.1em;
           border-bottom:1px solid #1a2433; }
.tbl-row { display:grid; grid-template-columns:0.6fr 0.9fr 1fr 1.2fr 1fr 0.8fr 0.7fr;
           padding:5px 6px; font-size:0.72rem; border-bottom:1px solid #111820; }
.tbl-row:hover { background:#0d1117; }

.mongo-ok  { color:#00e676; font-size:0.65rem; }
.mongo-off { color:#ff6d00; font-size:0.65rem; }
.pill { display:inline-block; padding:2px 8px; border-radius:12px;
        font-size:0.6rem; letter-spacing:0.1em; }
.pill-run { background:#003320; color:#00e676; }
.pill-off { background:#1a1200; color:#ff6d00; }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def _get(ep, timeout=0.8):
    try:
        r = requests.get(f"{API}/{ep}", timeout=timeout)
        return r.json() if r.ok else {}
    except:
        return {}

def _fmt_pnl(v):
    if v is None:   return "—", "#5c7a8a"
    if v > 0:       return f"+${v:,.4f}", "#00e676"
    if v < 0:       return f"-${abs(v):,.4f}", "#ff1744"
    return "$0.0000", "#5c7a8a"


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">⚡TYCHE </div>', unsafe_allow_html=True)
st.markdown('<div class="sub">High Frequency Trading Simulator</div>', unsafe_allow_html=True)

hc1 = st.columns([4])


st.markdown("---")

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")


# ════════════════════════════════════════════════════════
#  LEFT PANEL
# ════════════════════════════════════════════════════════
with left:

    # ── PNL + METRICS ──────────────────────────────────
    @st.fragment(run_every=0.8)
    def _pnl_block():
        pf  = _get("portfolio")
        ag  = _get("agent")
        val = float(pf.get("portfolio_value", 10_000))
        pnl = float(pf.get("pnl_usd", 0.0))
        
        # ⚡ COLOR LOGIC
        # If P&L is negative, use red. If positive, use green.
        is_pos = pnl >= 0
        cls = "pnl-pos" if is_pos else "pnl-neg"
        color_hex = "#00e676" if is_pos else "#ff1744"
        sign = "▲" if is_pos else "▼"

        # 1. Main Value ($9,406.91)
        st.markdown(f'<div class="{cls}">${val:,.2f}</div>', unsafe_allow_html=True)
        
        # 2. Subtext (▼ $593.09...) - Forced to match the same color
        st.markdown(
            f'<div style="color:{color_hex}; font-size:0.8rem; margin-top:0.2rem; font-weight:700;">'
            f'{sign} ${abs(pnl):,.2f} · {pf.get("pnl_pct", 0.0):+.3f}% · LIFETIME P&L</div>',
            unsafe_allow_html=True
        )
        st.write("")

        # Metrics Row (Keep these as they are)
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("EPISODE", f'{pf.get("episode", ag.get("episode", 0)):,}')
        m2.metric("STEPS",    f'{ag.get("total_steps", pf.get("total_steps",0)):,}')
        m3.metric("TRADES",   f'{pf.get("n_trades", 0):,}')
        m4.metric("WIN %",    f'{pf.get("win_rate", 0.0):.1f}%')
        m5.metric("SHARPE",   f'{pf.get("sharpe", 0.0):.3f}')
        m6.metric("DD %",     f'{pf.get("drawdown", 0.0):.2f}%')

    _pnl_block()

    st.write("")

    # ── EQUITY CHART ────────────────────────────────────
    @st.fragment(run_every=1.5)
    def _chart():
        st.markdown('<div class="section">EQUITY PERFORMANCE</div>', unsafe_allow_html=True)
        hist = _get("pnl_history")
        if not hist or len(hist) < 5:
            st.caption("Stabilizing trajectory...")
            return
            
        try:
            df = pd.DataFrame(hist)
            # ⚡ FORCE THE CHART TO ONLY PLOT THE 'pnl' COLUMN
            # This prevents the Y-axis from scaling to 60,000 (steps)
            st.line_chart(df['pnl'], height=220, color="#ff1744" if df['pnl'].iloc[-1] < 0 else "#00e676")
        except Exception as e:
            st.caption(f"Chart Render Lag: {e}")
    _chart()

    # ── EPISODE HISTORY TABLE ───────────────────────────
    @st.fragment(run_every=3.0)
    def _ep_table():
        st.markdown('<div class="section">EPISODE HISTORY</div>', unsafe_allow_html=True)
        eps = _get("episodes")
        if not eps:
            st.caption("No episodes logged yet.")
            return
        rows = eps[-20:][::-1]
        df = pd.DataFrame(rows)
        show_cols = [c for c in ["episode","pnl_usd","pnl_pct","n_trades",
                                  "win_rate","max_drawdown","steps"] if c in df.columns]
        df = df[show_cols].copy()
        rename = {"episode":"EP","pnl_usd":"P&L $","pnl_pct":"P&L %",
                  "n_trades":"TRADES","win_rate":"WIN%",
                  "max_drawdown":"MAX DD","steps":"STEPS"}
        df = df.rename(columns=rename)
        st.dataframe(df, use_container_width=True, height=250,
                     hide_index=True,
                     column_config={
                         "P&L $": st.column_config.NumberColumn(format="$%.2f"),
                         "P&L %": st.column_config.NumberColumn(format="%.3f%%"),
                         "WIN%":  st.column_config.NumberColumn(format="%.1f%%"),
                         "MAX DD":st.column_config.NumberColumn(format="$%.2f"),
                     })

    _ep_table()

# ════════════════════════════════════════════════════════
#  RIGHT PANEL
# ════════════════════════════════════════════════════════
with right:

    # ── ASSET PRICES ────────────────────────────────────

    @st.fragment(run_every=0.8)
    def _prices():
        st.markdown('<div class="section">LIVE PRICES</div>', unsafe_allow_html=True)
        data = _get("portfolio")
        
        # ⚡ Search for the prices under multiple possible keys
        prices = data.get("prices", data.get("current_prices", data.get("px", {})))
        
        if not prices or len(prices) == 0:
            # Fallback: Try a direct endpoint if the portfolio packet is stripped
            prices = _get("prices") 
        
        if not prices:
            st.caption("Initializing market data stream...")
            return
            
        h = '<div class="asset-grid">'
        # Sort them so they don't jump around
        for sym in sorted(prices.keys()):
            val = prices[sym]
            label = sym.replace("USDT","")
            h += (f'<div class="asset-card">'
                  f'<div class="asset-sym">{label}</div>'
                  f'<div class="asset-px">${float(val):,.2f}</div>'
                  f'</div>')
        st.markdown(h + '</div>', unsafe_allow_html=True)



    _prices()


    # ── STRESS MODE ─────────────────────────────────────
    @st.fragment(run_every=1.0)
    def _stress():
        s = _get("stress")
        mode = s.get("mode", "NORMAL")
        color = "#ff6d00" if mode != "NORMAL" else "#00e676"
        st.markdown(
            f'<div style="font-size:0.65rem; color:#3a5566; margin-bottom:4px;">ADVERSARIAL MODE</div>'
            f'<div style="color:{color}; font-weight:700; font-size:0.9rem; '
            f'letter-spacing:0.15em;">{mode}</div>',
            unsafe_allow_html=True
        )

    _stress()

    st.write("")

    # ── MONGODB STATUS ──────────────────────────────────
    @st.fragment(run_every=5.0)
    def _mongo():
        st.markdown('<div class="section">DATABASE / PERSISTENCE</div>', unsafe_allow_html=True)
        ms = _get("mongo_stats")
        
        # UI Polish for the 70% Milestone
        source = ms.get("source", "memory").upper()
        color = "#00e676" if source == "MONGODB" else "#ff6d00"
        
        st.markdown(f'<div style="color:{color}; font-weight:700; font-size:0.8rem; letter-spacing:0.1em;">'
                    f'● {source} ENGINE</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("LIFETIME TRADES", ms.get("total_history_trades", 0))
        c2.metric("TOTAL EPISODES",  ms.get("n_episodes", 0))
        c3.metric("ALL-TIME BEST",   f'${ms.get("best_pnl", 0.0):+,.2f}')

    _mongo()

    st.write("")

    # ── LIVE TRADE LEDGER ───────────────────────────────
    @st.fragment(run_every=0.5)
    def _ledger():
        st.markdown('<div class="section">LIVE TRADE LEDGER</div>', unsafe_allow_html=True)
        trades = _get("trades")
        if not trades:
            st.caption("No trades yet...")
            return

        st.markdown(
            '<div class="tbl-hdr">'
            '<span>ACT</span><span>ASSET</span><span>EP</span>'
            '<span>PRICE</span><span>QTY</span><span>P&L</span><span>TIME</span>'
            '</div>',
            unsafe_allow_html=True
        )

        for t in trades[:15]:
            ac = str(t.get("action", "BUY")).upper()
            
            # ⚡ PNL LOGIC
            # If it's a SELL, we show the REALIZED profit with 4 decimals
            if "SELL" in ac:
                pnl_val = float(t.get("pnl_usd", 0.0))
                # ⚡ Precision is key here: .4f shows the fractions of a cent
                pnl_str = f"${pnl_val:+,.4f}" 
                pnl_color = "#00e676" if pnl_val > 0 else "#ff1744"
            else:
                # For BUY rows, PNL is traditionally empty or "—"
                pnl_str = "—"
                pnl_color = "#5c7a8a"

            st.markdown(f'''
                <div class="tbl-row">
                    <span style="color:{"#00e676" if "BUY" in ac else "#ff1744"}; font-weight:700;">{ac}</span>
                    <span>{str(t.get("asset","")).replace("USDT","")}</span>
                    <span style="color:#445566">{t.get("episode", "0")}</span>
                    <span>${float(t.get("price", 0)):,.2f}</span>
                    <span>{float(t.get("qty", 0)):.4f}</span>
                    <span style="color:{pnl_color}; font-weight:700;">{pnl_str}</span>
                    <span style="color:#3a5566">{t.get("ts", "")}</span>
                </div>''', unsafe_allow_html=True)

    _ledger()

