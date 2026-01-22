import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import os
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from dateutil import parser
import re
import plotly.graph_objects as go
import time
import datetime
import socket
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import math
import html
from difflib import SequenceMatcher

# ---------------------------------------------------------
# 1. 설정 및 초기화
# ---------------------------------------------------------
CSV_FILE = "my_portfolio.csv"
mobile_mode = True
chart_height = "350px"
socket.setdefaulttimeout(30)

if 'sidebar_state' not in st.session_state:
    st.session_state['sidebar_state'] = 'expanded'

st.set_page_config(
    layout="wide", 
    page_title="AI Hyper-Analyst V90 (Fixed)", 
    page_icon="📈",
    initial_sidebar_state=st.session_state['sidebar_state']
)

# [로그 시스템]
if 'log_buffer' not in st.session_state:
    st.session_state['log_buffer'] = []

def add_log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}"
    st.session_state['log_buffer'].append(log_entry)
    if len(st.session_state['log_buffer']) > 500:
        st.session_state['log_buffer'].pop(0)

# [분석 항목 정의]
opt_targets = [
    "현금건전성 지표 (FCF, 유동비율, 부채비율)", 
    "핵심 재무제표 분석 (손익, 대차대조, 현금흐름)",
    "투자기관 목표주가 및 컨센서스", 
    "호재/악재 뉴스 판단", 
    "기술적 지표 (RSI/이평선)",
    "외국인/기관 수급 분석", 
    "경쟁사 비교 및 업황", 
    "단기/중기 매매 전략",
    "투자성향별 포트폴리오 적정보유비중"
]

# 상태 변수 초기화
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = {} 
if 'is_analyzing' not in st.session_state: st.session_state['is_analyzing'] = False
if 'targets_to_run' not in st.session_state: st.session_state['targets_to_run'] = []
if 'current_mode' not in st.session_state: st.session_state['current_mode'] = "MAIN"
if 'prompt_mode' not in st.session_state: st.session_state['prompt_mode'] = False
if 'proc_index' not in st.session_state: st.session_state['proc_index'] = 0
if 'proc_stage' not in st.session_state: st.session_state['proc_stage'] = 0 
if 'temp_data' not in st.session_state: st.session_state['temp_data'] = {}
if 'select_all_state' not in st.session_state: st.session_state['select_all_state'] = False
if 'new_ticker_input' not in st.session_state: st.session_state['new_ticker_input'] = ""

for opt in opt_targets:
    if f"focus_{opt}" not in st.session_state: st.session_state[f"focus_{opt}"] = True
if 'focus_all' not in st.session_state: st.session_state['focus_all'] = True

# ---------------------------------------------------------
# 2. 데이터 관리 함수
# ---------------------------------------------------------
def load_data_to_state():
    if 'portfolio_df' not in st.session_state:
        add_log("📥 [INIT] 포트폴리오 데이터 로드 시도...")
        if os.path.exists(CSV_FILE):
            try:
                df = pd.read_csv(CSV_FILE)
                if df.empty:
                    st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
                else:
                    st.session_state['portfolio_df'] = df.reset_index(drop=True)
                    add_log(f"✅ [INIT] 데이터 로드 완료: {len(df)}개.")
            except Exception as e:
                st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
                add_log(f"❌ [INIT] 로드 에러: {str(e)}")
        else:
            st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])

def save_state_to_csv():
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        df = df.reset_index(drop=True)
        st.session_state['portfolio_df'] = df 
        try:
            with open(CSV_FILE, 'w', encoding='utf-8', newline='') as f:
                df.to_csv(f, index=False)
                f.flush()
                os.fsync(f.fileno()) 
            add_log(f"💾 [SAVE] 저장 완료.")
        except Exception as e:
            add_log(f"❌ [SAVE] 저장 실패: {str(e)}")

def add_ticker_logic():
    raw_input = st.session_state.get('new_ticker_input', '')
    if raw_input:
        add_log(f"➕ [ADD] 요청: '{raw_input}'")
        tickers = [t.strip().upper() for t in raw_input.split(',')]
        df = st.session_state['portfolio_df']
        existing_tickers = df['ticker'].values
        
        new_rows = []
        for ticker in tickers:
            if ticker and ticker not in existing_tickers:
                # 여기서도 메타데이터 확보 시도
                meta = get_metadata_robust(ticker)
                name = meta['name']
                new_rows.append({'ticker': ticker, 'name': name})
                add_log(f"   -> 추가: {ticker} ({name})")
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            st.session_state['portfolio_df'] = df
            save_state_to_csv()
            
    st.session_state['new_ticker_input'] = ""

load_data_to_state()

if 'del_ticker' in st.query_params:
    del_ticker = st.query_params['del_ticker']
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        df = df[df['ticker'] != del_ticker]
        st.session_state['portfolio_df'] = df
        save_state_to_csv()
        if f"chk_{del_ticker}" in st.session_state:
            del st.session_state[f"chk_{del_ticker}"]
    st.query_params.clear()
    st.rerun()

# ---------------------------------------------------------
# 3. 유틸리티 및 강력한 메타데이터 수집기
# ---------------------------------------------------------
def get_robust_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def run_with_timeout(func, args=(), timeout=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try: return future.result(timeout=timeout)
        except: return None

def _fetch_history(ticker, period): return yf.Ticker(ticker).history(period=period)
def _fetch_info(ticker): return yf.Ticker(ticker).info

# [핵심 솔루션 1] 강력한 메타데이터 수집 함수
def get_metadata_robust(ticker):
    """
    yfinance의 여러 속성을 뒤져서 이름, 섹터, 산업을 찾아내는 함수.
    실패 시 'Unknown'을 반환하지만, 포트폴리오에 저장된 이름이 있다면 우선 사용.
    """
    # 기본값
    result = {
        'name': ticker,
        'sector': "Unknown",
        'industry': "Unknown"
    }

    # 1. 포트폴리오에 저장된 이름 확인
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        row = df[df['ticker'] == ticker]
        if not row.empty:
            result['name'] = row.iloc[0]['name']

    # 2. yfinance 정보 조회 (Timeout 적용)
    try:
        info = run_with_timeout(_fetch_info, args=(ticker,), timeout=6)
        if info:
            # 이름 찾기 (longName -> shortName -> symbol)
            fetched_name = info.get('longName') or info.get('shortName')
            if fetched_name:
                result['name'] = fetched_name
            
            # 섹터 찾기 (sector -> category -> gicsSector)
            sector = info.get('sector') or info.get('category') or info.get('gicsSector')
            if sector:
                result['sector'] = sector
            
            # 산업 찾기 (industry -> industryKey -> gicsIndustry)
            industry = info.get('industry') or info.get('industryKey') or info.get('gicsIndustry')
            if industry:
                result['industry'] = industry
                
    except Exception as e:
        add_log(f"⚠️ [META] {ticker} 메타데이터 조회 실패: {e}")

    return result

def clean_html_text(text):
    if not text: return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = html.unescape(clean)
    clean = " ".join(clean.split())
    return clean

def is_similar(a, b, threshold=0.7):
    if not a or not b: return False
    return SequenceMatcher(None, a, b).ratio() > threshold

def fetch_rss_realtime(url, limit=10):
    try:
        session = get_robust_session()
        response = session.get(url, timeout=5)
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall('./channel/item')[:limit]:
            title = item.find('title').text
            try: dt = parser.parse(item.find('pubDate').text); date_str = dt.strftime("%m-%d %H:%M")
            except: date_str = "최신"
            desc = clean_html_text(item.find('description').text) if item.find('description') is not None else ""
            items.append({'title': title, 'link': item.find('link').text, 'date_str': date_str, 'summary': desc})
        return items
    except: return []

def get_realtime_news(ticker, name):
    add_log(f"📰 [NEWS] {ticker} 뉴스 수집 시작")
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    if not is_kr:
        # Yahoo Finance RSS
        try:
            items = fetch_rss_realtime(f"https://finance.yahoo.com/rss/headline?s={ticker}", limit=5)
            for i in items: i['source'] = "Yahoo"; news_items.append(i)
        except: pass
        
        # yfinance Library Fallback
        if not news_items:
            try:
                yf_news = yf.Ticker(ticker).news
                for item in yf_news:
                    ts = item.get('providerPublishTime', 0)
                    d_str = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "최신"
                    news_items.append({'title': item.get('title'), 'link': item.get('link'), 'date_str': d_str, 'source': "YahooLib", 'summary': ""})
            except: pass

    # Google News Fallback
    query = f'"{name}"' if is_kr else f'{ticker} stock'
    q_enc = urllib.parse.quote(query)
    g_items = fetch_rss_realtime(f"https://news.google.com/rss/search?q={q_enc}&hl=ko&gl=KR&ceid=KR:ko", limit=5)
    for i in g_items: i['source'] = "Google"; news_items.append(i)
    
    return news_items[:7]

def get_financial_metrics(ticker):
    # 재무데이터도 별도로 조회 시도
    info = run_with_timeout(_fetch_info, args=(ticker,), timeout=5)
    if not info: return {}
    def fmt(val): return f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A"
    return {
        "Free Cash Flow": fmt(info.get('freeCashflow')),
        "Current Ratio": fmt(info.get('currentRatio')),
        "Debt to Equity": fmt(info.get('debtToEquity')),
        "ROE": fmt(info.get('returnOnEquity')),
        "Revenue": fmt(info.get('totalRevenue'))
    }

def sanitize_text(text):
    text = text.replace('$', '\$'); text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    return text

def collapse_sidebar():
    js = """<script>var closeBtn = window.parent.document.querySelector('[data-testid="stSidebarExpandedControl"]');if (closeBtn) {closeBtn.click();}</script>"""
    st.components.v1.html(js, height=0, width=0)

def start_analysis_process(targets, mode, is_prompt_only):
    st.session_state['is_analyzing'] = True
    st.session_state['targets_to_run'] = targets
    st.session_state['current_mode'] = mode
    st.session_state['prompt_mode'] = is_prompt_only
    st.session_state['analysis_results'] = {} 
    st.session_state['proc_index'] = 0
    st.session_state['proc_stage'] = 1 

def generate_with_fallback(prompt, api_key, start_model):
    genai.configure(api_key=api_key)
    chain = [start_model] + [m for m in ["gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.0-pro"] if m != start_model]
    
    for model_name in chain:
        try:
            add_log(f"🧠 [AI] 모델 시도: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name
        except Exception as e:
            add_log(f"⚠️ [AI] 실패 ({model_name}): {e}")
            time.sleep(0.5)
    raise Exception("All models failed.")

def handle_search_click(mode, is_prompt):
    raw = st.session_state.get("s_input", "")
    if raw: start_analysis_process([t.strip() for t in raw.split(',') if t.strip()], mode, is_prompt)
    else: st.warning("티커 입력 필요")

def step_fetch_data(ticker, mode):
    add_log(f"📦 [STEP 1] 데이터 수집: {ticker}")
    
    # [솔루션 적용] 강력한 메타데이터 수집
    meta = get_metadata_robust(ticker)
    stock_name = meta['name']
    sector_info = meta['sector']
    industry_info = meta['industry']
    
    # [솔루션 2] 메타데이터가 Unknown일 경우를 위한 프롬프트 인젝션 준비
    metadata_injection = ""
    if sector_info == "Unknown" or industry_info == "Unknown":
        metadata_injection = f"""
        **[중요] 현재 데이터 소스에서 이 기업의 섹터와 산업 정보를 가져오지 못했습니다.**
        당신의 지식 베이스를 활용하여 '{ticker}' ({stock_name})의 정확한 **섹터(Sector)**와 **산업(Industry)**을 스스로 판단하고, 
        보고서의 [기업 개요] 섹션에 명시하십시오.
        """

    clean_code = re.sub(r'[^0-9]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker

    try:
        # 주가 데이터
        period = st.session_state.get('selected_period_str', '1y')
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=10)
        if df is None: df = pd.DataFrame()

        # 데이터 요약
        data_summary = "No Data"
        if not df.empty:
            curr = df['Close'].iloc[-1]; high = df['High'].max(); low = df['Low'].min()
            data_summary = f"Current: {curr:.2f}, High: {high:.2f}, Low: {low:.2f}\n{df.tail(60).to_string()}"

        # 재무 및 뉴스
        fin_str = "N/A"; news_text = "N/A"
        if mode not in ["10K", "10Q", "8K"]:
            fm = get_financial_metrics(ticker)
            fin_str = str(fm) if fm else "N/A"
            if st.session_state.get('use_news', True):
                news = get_realtime_news(ticker, stock_name)
                if news: news_text = "\n".join([f"- [{n['source']}] {n['title']} ({n['date_str']})" for n in news])

        # 프롬프트 조립
        focus_list = [opt for opt in opt_targets if st.session_state.get(f"focus_{opt}", True)]
        focus = ", ".join(focus_list)
        viewpoint = st.session_state.get('selected_viewpoint', 'General')
        depth = st.session_state.get('analysis_depth', "2. 표준")
        
        # [솔루션 3] 시나리오 확률 및 근거 요청 로직 강화
        level_instruction = ""
        if "5." in depth:
            level_instruction = """
            \n[시나리오 분석 필수 지침]
            '시나리오 분석'을 수행할 때는 반드시 다음 3가지 항목을 포함해야 합니다:
            1. **시나리오 명**: (예: 낙관적, 기본, 비관적)
            2. **실현 확률(Probability)**: 각 시나리오가 발생할 확률을 %로 추산하여 명시하십시오. (예: 60%)
            3. **판단 근거(Rationale)**: 왜 그 확률을 부여했는지 구체적인 논거를 설명하십시오.
            """

        if "투자성향별 포트폴리오 적정보유비중" in focus:
            level_instruction += """
            \n[특별 지시: 투자성향별 비중]
            보고서 결론에 다음 3가지 성향별 권장 보유 비중(%)과 논리를 각각 서술하십시오:
            1. 🦁 공격적 (Aggressive)
            2. ⚖️ 중립적 (Moderate)
            3. 🛡️ 보수적 (Conservative)
            """

        growth_value_logic = """
        [핵심: 성장주 vs 가치주 판단]
        먼저 이 기업이 성장주인지 가치주인지 규정하고, 그에 맞춰 분석하십시오.
        (성장주: 매출성장, 현금흐름, 지속성 위주 / 가치주: 점유율, 배당, 이익률 위주)
        """
        level_instruction += growth_value_logic

        # 기본 프롬프트 템플릿
        base_prompt = f"""
        [역할] 월가 수석 애널리스트
        [대상 티커] {ticker}
        [공식 기업명] {stock_name}
        [섹터(Sector)] {sector_info}
        [산업(Industry)] {industry_info}
        [모드] {mode}
        [중점 분석] {focus}
        [관점] {viewpoint}
        
        {metadata_injection}
        
        {level_instruction}
        
        [데이터 요약]
        {data_summary}
        [재무 지표]
        {fin_str}
        [뉴스]
        {news_text}
        
        [지시사항]
        위 데이터를 바탕으로 전문적인 투자 보고서를 작성하십시오.
        **반드시 자연스러운 한국어로 작성하십시오.**
        결론에는 [매수 / 매도 / 관망] 중 하나의 의견을 제시하십시오.
        """
        
        # 공시 모드별 프롬프트 분기 (필요 시 내용 추가 가능)
        prompt = base_prompt # 기본적으로 base 사용, 공시 모드면 아래 덮어쓰기

        if mode == "10K":
             prompt = base_prompt + "\n[특수 모드] 10-K(연차보고서) 관점에서 장기적 비전, 리스크, 재무 건전성을 심층 분석하십시오."
        elif mode == "10Q":
             prompt = base_prompt + "\n[특수 모드] 10-Q(분기보고서) 관점에서 직전 분기 대비 실적 변화와 가이던스 추이를 중점 분석하십시오."
        elif mode == "8K":
             prompt = base_prompt + "\n[특수 모드] 8-K(수시공시) 관점에서 최근 발생한 특정 이벤트가 주가에 미칠 단기적 영향을 분석하십시오."

        st.session_state['temp_data'] = {
            'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': []
        }
        return True

    except Exception as e:
        add_log(f"❌ [STEP 1 Error] {e}")
        st.error(e)
        return False

# ---------------------------------------------------------
# 5. UI 구성
# ---------------------------------------------------------
st.sidebar.subheader("🎯 분석 옵션")

viewpoint_mapping = {"단기": "3mo", "스윙": "6mo", "중기": "2y", "장기": "5y"}
sel_vp = st.sidebar.select_slider("", list(viewpoint_mapping.keys()), value="중기", label_visibility="collapsed")
st.session_state['selected_period_str'] = viewpoint_mapping[sel_vp]
st.session_state['selected_viewpoint'] = sel_vp

levels = ["1.요약", "2.표준", "3.심층", "4.전문가", "5.시나리오"]
sel_depth = st.sidebar.select_slider("", levels, value="5.시나리오", label_visibility="collapsed")
st.session_state['analysis_depth'] = sel_depth

st.session_state['use_news'] = st.sidebar.toggle("뉴스 데이터 반영", value=True)

def toggle_focus_all():
    val = st.session_state['focus_all']
    for opt in opt_targets: st.session_state[f"focus_{opt}"] = val

with st.sidebar.expander("☑️ 중점 분석 항목", expanded=False):
    st.checkbox("전체 선택", key="focus_all", on_change=toggle_focus_all)
    for opt in opt_targets: st.checkbox(opt, key=f"focus_{opt}")

# API 키
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.sidebar.error("Secrets에 GEMINI_API_KEY 필요")

# 검색/포트폴리오 탭
t1, t2 = st.sidebar.tabs(["⚡ 검색", "⭐ 포트폴리오"])

with t1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.text_input("티커 (예: SOUN, 005930.KS)", key="s_input")
    c1, c2 = st.columns(2)
    chk_p = c1.checkbox("프롬프트만", key="chk_p_s", value=False)
    if c2.button("🔍 시작", key="btn_s"):
        handle_search_click("MAIN", chk_p)
    
    st.markdown("##### 📑 공시")
    b1, b2, b3 = st.columns(3)
    if b1.button("10-K"): handle_search_click("10K", chk_p)
    if b2.button("10-Q"): handle_search_click("10Q", chk_p)
    if b3.button("8-K"): handle_search_click("8K", chk_p)

with t2:
    st.markdown("<br>", unsafe_allow_html=True)
    c_add1, c_add2 = st.columns([0.75, 0.25])
    c_add1.text_input("추가", key="new_ticker_input", label_visibility="collapsed")
    c_add2.button("➕", on_click=add_ticker_logic)
    
    # 포트폴리오 UI (HTML/JS)
    fav_df = st.session_state.get('portfolio_df', pd.DataFrame())
    sel_tickers = []
    if 'selected' in st.query_params:
        sel_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]

    import json
    t_data = []
    if not fav_df.empty:
        for _, r in fav_df.iterrows():
            t_data.append({'ticker': r['ticker'], 'name': str(r['name'])})
    
    t_json = json.dumps(t_data)
    sel_json = json.dumps(sel_tickers)
    
    html = f"""
    <style>
        .pf-box {{border:1px solid #ddd; padding:10px; border-radius:8px; max-height:300px; overflow-y:auto;}}
        .pf-item {{display:flex; justify-content:space-between; align-items:center; padding:5px; border-bottom:1px solid #eee;}}
        .pf-item:hover {{background:#f9f9f9;}}
        .pf-item.active {{background:#e6f3ff;}}
        .pf-btn {{border:none; background:none; cursor:pointer; font-size:16px;}}
        .pf-del:hover {{color:red;}}
    </style>
    <div class="pf-box" id="pfBox"></div>
    <script>
        const data = {t_json};
        let selected = {sel_json};
        
        function render() {{
            const box = document.getElementById('pfBox');
            box.innerHTML = '';
            if(data.length === 0) {{ box.innerHTML = '<div style="color:#999;text-align:center">비어있음</div>'; return; }}
            
            data.forEach(item => {{
                const div = document.createElement('div');
                const isActive = selected.includes(item.ticker);
                div.className = 'pf-item' + (isActive ? ' active' : '');
                div.onclick = (e) => {{
                    if(e.target.className.includes('pf-del')) return;
                    if(isActive) selected = selected.filter(t => t !== item.ticker);
                    else selected.push(item.ticker);
                    sync(); render();
                }};
                
                div.innerHTML = `
                    <div>
                        <div style="font-weight:bold; font-size:12px">${{item.ticker}}</div>
                        <div style="font-size:10px; color:#666">${{item.name}}</div>
                    </div>
                    <button class="pf-btn pf-del" onclick="del('${{item.ticker}}')">×</button>
                `;
                box.appendChild(div);
            }});
        }}
        
        function del(t) {{
            const url = new URL(window.parent.location.href);
            url.searchParams.set('del_ticker', t);
            window.parent.location.href = url.toString();
        }}
        
        function sync() {{
            const url = new URL(window.parent.location.href);
            if(selected.length > 0) url.searchParams.set('selected', selected.join(','));
            else url.searchParams.delete('selected');
            window.parent.history.replaceState(null, '', url.toString());
        }}
        render();
    </script>
    """
    st.components.v1.html(html, height=320)
    
    chk_p_fav = st.checkbox("프롬프트만", key="chk_p_fav")
    if st.button("🚀 종합 분석", type="primary"):
        if sel_tickers: start_analysis_process(sel_tickers, "MAIN", chk_p_fav)
        else: st.warning("선택된 종목 없음")

# 모델 선택 및 로그
st.sidebar.markdown('---')
st.session_state['selected_model'] = st.sidebar.selectbox("모델", [
    "gemini-1.5-pro", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash"
])

with st.sidebar.expander("📜 로그"):
    st.text_area("", "\n".join(st.session_state['log_buffer']), height=200)

# ---------------------------------------------------------
# 6. 실행 로직
# ---------------------------------------------------------
st.title("📈 AI Hyper-Analyst V90")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    idx = st.session_state['proc_index']
    stage = st.session_state['proc_stage']
    
    if idx >= len(targets):
        st.success("완료!")
        st.session_state['is_analyzing'] = False
        st.rerun()

    curr = targets[idx]
    st.progress((idx * 2 + (1 if stage > 1 else 0)) / (len(targets)*2), f"분석 중: {curr}")

    if stage == 1:
        collapse_sidebar()
        with st.spinner(f"데이터 수집: {curr}..."):
            if step_fetch_data(curr, st.session_state['current_mode']):
                st.session_state['proc_stage'] = 2
            else:
                st.session_state['analysis_results'][curr] = {'status': 'error', 'report': '데이터 실패'}
                st.session_state['proc_index'] += 1
            st.rerun()

    elif stage == 2:
        temp = st.session_state['temp_data']
        if st.session_state['prompt_mode']:
            res = {'status': 'manual', 'prompt': temp['prompt'], 'report': "프롬프트 생성됨", 'name': temp['name'], 'df': temp['df'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr']}
        else:
            with st.spinner("AI 분석 중..."):
                try:
                    rep, model = generate_with_fallback(temp['prompt'], api_key, st.session_state['selected_model'])
                    res = {'status': 'success', 'report': sanitize_text(rep), 'model': model, 'name': temp['name'], 'df': temp['df'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr']}
                except Exception as e:
                    res = {'status': 'error', 'report': str(e), 'name': temp['name'], 'df': pd.DataFrame()}
        
        st.session_state['analysis_results'][curr] = res
        st.session_state['proc_index'] += 1
        st.session_state['proc_stage'] = 1
        st.rerun()

# ---------------------------------------------------------
# 7. 결과 표시
# ---------------------------------------------------------
if not st.session_state['is_analyzing'] and st.session_state['analysis_results']:
    for ticker, data in st.session_state['analysis_results'].items():
        with st.expander(f"📊 {data.get('name', ticker)} ({ticker})", expanded=True):
            if not data.get('df', pd.DataFrame()).empty:
                if data['is_kr']:
                    st.line_chart(data['df']['Close'])
                else:
                    st.components.v1.html(f"""<div id="c_{ticker}" style="height:350px"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"{data['tv_symbol']}","interval":"D","timezone":"Asia/Seoul","theme":"light","style":"1","locale":"ko","container_id":"c_{ticker}"}});</script>""", height=360)
            
            if data['status'] == 'manual':
                st.code(data['prompt'])
            else:
                st.markdown(data['report'])
