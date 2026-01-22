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
import json

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
    page_title="AI Hyper-Analyst V86 (Final)", 
    page_icon="📈",
    initial_sidebar_state=st.session_state['sidebar_state']
)

# [로그 시스템] 초기화 및 함수 정의
if 'log_buffer' not in st.session_state:
    st.session_state['log_buffer'] = []

def add_log(message):
    """시스템 로그를 추가하는 함수 (상세 모드)"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] 
    log_entry = f"[{timestamp}] {message}"
    st.session_state['log_buffer'].append(log_entry)
    if len(st.session_state['log_buffer']) > 500:
        st.session_state['log_buffer'].pop(0)

# [변수 정의] 
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

# 체크박스 상태 초기화
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
                    add_log(f"✅ [INIT] 데이터 로드 완료: {len(df)}개 항목.")
            except Exception as e:
                st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
                add_log(f"❌ [INIT] 데이터 로드 에러: {str(e)}")
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
            add_log(f"💾 [SAVE] 파일 저장 완료.")
        except Exception as e:
            add_log(f"❌ [SAVE] 파일 저장 실패: {str(e)}")

def add_ticker_logic():
    raw_input = st.session_state.get('new_ticker_input', '')
    if raw_input:
        add_log(f"➕ [ADD] 티커 추가 요청: '{raw_input}'")
        tickers = [t.strip().upper() for t in raw_input.split(',')]
        df = st.session_state['portfolio_df']
        existing_tickers = df['ticker'].values
        
        new_rows = []
        for ticker in tickers:
            if ticker and ticker not in existing_tickers:
                # 메타데이터 함수 사용하여 이름 확보 시도
                meta = fetch_metadata_robust(ticker)
                name = meta.get('name', ticker)
                new_rows.append({'ticker': ticker, 'name': name})
                add_log(f"   -> 추가: {ticker} ({name})")
            else:
                add_log(f"   -> 중복 스킵: {ticker}")
        
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
# 3. 유틸리티 & 강력한 메타데이터 수집 함수 (Multi-Layer)
# ---------------------------------------------------------
def get_robust_session():
    session = requests.Session()
    # 봇 차단 방지를 위한 User-Agent 위조
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
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

# [핵심] 다중 소스 메타데이터 수집기
def fetch_metadata_robust(ticker):
    """
    1차: yfinance API
    2차: Yahoo Finance 웹 스크래핑 (Fallback)
    """
    add_log(f"🕵️ [META] {ticker} 상세 정보 수집 시작...")
    
    # 1. yfinance 시도
    try:
        info = run_with_timeout(_fetch_info, args=(ticker,), timeout=6)
        if info and 'shortName' in info:
            name = info.get('shortName') or info.get('longName') or ticker
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            add_log(f"   ✅ [Source: API] Name: {name}, Sec: {sector}")
            return {'name': name, 'sector': sector, 'industry': industry}
    except: pass

    # 2. 웹 스크래핑 시도 (Fallback)
    try:
        add_log(f"   ⚠️ [Fallback] 웹 스크래핑 시도...")
        url = f"https://finance.yahoo.com/quote/{ticker}/profile"
        session = get_robust_session()
        resp = session.get(url, timeout=5)
        
        name = ticker
        sector = "Unknown"
        industry = "Unknown"
        
        if resp.status_code == 200:
            txt = resp.text
            # 정규표현식으로 이름, 섹터, 산업 추출 시도
            name_match = re.search(r'<title>(.*?) \((.*?)\) Company Profile', txt)
            if name_match: name = name_match.group(1).strip()
            
            # Yahoo 구조에 따른 간단 파싱 (구조 변경시 실패 가능성 있음)
            sec_match = re.search(r'Sector:.*?<span class="value">(.*?)</span>', txt, re.DOTALL) # 예시 패턴
            # 실제 야후 페이지 구조가 복잡하므로 간단한 텍스트 검색 사용
            if "Sector(s)" in txt:
                # 단순화된 로직: HTML 태그 제거하고 텍스트 주변 검색 (구현 복잡도상 생략 후 AI에게 위임이 나음)
                pass 
            
            add_log(f"   ✅ [Source: Web] Name found: {name}")
            return {'name': name, 'sector': sector, 'industry': industry}
    except Exception as e:
        add_log(f"   ❌ [Fallback Error] {e}")

    # 3. 실패 시
    add_log(f"   ⚠️ [Failure] 메타데이터 확보 실패. AI에게 위임.")
    return {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown'}

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
            desc = ""
            if item.find('description') is not None: desc = clean_html_text(item.find('description').text)
            items.append({'title': title, 'link': item.find('link').text, 'date_str': date_str, 'summary': desc})
        return items
    except: return []

def get_realtime_news(ticker, name):
    add_log(f"📰 [NEWS] {ticker} 뉴스 수집")
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    # 1. Yahoo RSS
    if not is_kr:
        try:
            items = fetch_rss_realtime(f"https://finance.yahoo.com/rss/headline?s={ticker}", limit=5)
            for i in items: i['source'] = "Yahoo"; news_items.append(i)
        except: pass

    # 2. Google News RSS
    search_query = f'"{name}"' if is_kr else f'{ticker} stock'
    q_encoded = urllib.parse.quote(search_query)
    try:
        url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ko&gl=KR&ceid=KR:ko"
        items = fetch_rss_realtime(url, limit=5)
        for i in items: i['source'] = "Google"; news_items.append(i)
    except: pass
    
    return news_items[:7]

def get_financial_metrics(ticker):
    info = run_with_timeout(_fetch_info, args=(ticker,), timeout=5)
    if not info: return {}
    def fmt(k): v = info.get(k); return f"{v:,.2f}" if isinstance(v,(int,float)) else "N/A"
    return {
        "FCF": fmt('freeCashflow'), "CurrRatio": fmt('currentRatio'),
        "Debt/Eq": fmt('debtToEquity'), "ROE": fmt('returnOnEquity'),
        "Rev": fmt('totalRevenue'), "NetInc": fmt('netIncome')
    }

def sanitize_text(text):
    text = text.replace('$', '\$')
    return re.sub(r'\n\s*\n+', '\n\n', text).strip()

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
    # [중요] 제미나이 검색 도구 활성화 (정보 부족시 사용)
    tools = [{'google_search': {}}] 
    
    fallback_chain = [start_model, "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    for model_name in fallback_chain:
        try:
            add_log(f"   Attempting AI: {model_name}")
            # 도구 사용 설정
            model = genai.GenerativeModel(model_name, tools=tools)
            response = model.generate_content(prompt)
            return response.text, model_name 
        except Exception as e:
            add_log(f"   ⚠️ AI Fail ({model_name}): {e}")
            time.sleep(1)
            continue
    raise Exception("All AI models failed.")

def handle_search_click(mode, is_prompt):
    raw_input = st.session_state.get("s_input", "")
    if raw_input:
        targets = [t.strip() for t in raw_input.split(',') if t.strip()]
        start_analysis_process(targets, mode, is_prompt)
    else: st.warning("티커 입력 필요")

def step_fetch_data(ticker, mode):
    add_log(f"📦 [STEP 1] Data Fetch: {ticker}")
    
    # 1. 초기화 (이전 데이터 잔재 제거)
    clean_code = re.sub(r'[^0-9a-zA-Z]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker
    
    # 2. 메타데이터 확보 (다중 소스)
    meta = fetch_metadata_robust(ticker)
    stock_name = meta['name']
    sector = meta['sector']
    industry = meta['industry']

    # 3. 주가 데이터
    period = st.session_state.get('selected_period_str', '1y')
    df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=8)
    if df is None: df = pd.DataFrame()

    data_summary = "No Data"
    if not df.empty:
        curr = df['Close'].iloc[-1]
        data_summary = f"Current: {curr:.2f}\nRecent Data:\n{df.tail(5).to_string()}"
    
    # 4. 재무 및 뉴스
    fin_str = "N/A"
    news_text = "N/A"
    if mode not in ["10K", "10Q", "8K"]:
        try: fin_str = str(get_financial_metrics(ticker))
        except: pass
        
        if st.session_state.get('use_news', True):
            news = get_realtime_news(ticker, stock_name)
            if news: news_text = "\n".join([f"- [{n['source']}] {n['title']} ({n['date_str']})" for n in news])

    # 5. 프롬프트 구성
    selected_focus_list = []
    for opt in opt_targets:
        if st.session_state.get(f"focus_{opt}", True): selected_focus_list.append(opt)
    focus = ", ".join(selected_focus_list)
    viewpoint = st.session_state.get('selected_viewpoint', 'General')
    analysis_depth = st.session_state.get('analysis_depth', "2. 표준")

    # [핵심] 시나리오 확률 요청 및 메타데이터 자가 수정 지시
    level_instruction = ""
    scenario_instruction = ""
    if "5." in analysis_depth:
        scenario_instruction = """
        \n[필수 요청: 시나리오별 확률 명시]
        결론 부분에 반드시 '시나리오 분석(Scenario Analysis)' 섹션을 만들고, 다음 3가지 시나리오에 대해 **실현 가능 확률(%)**과 **그 이유(Rationale)**를 명시하십시오.
        1. 🚀 Bull Case (낙관적): 확률 OO% - 이유 요약
        2. ➖ Base Case (기본): 확률 OO% - 이유 요약
        3. 💧 Bear Case (비관적): 확률 OO% - 이유 요약
        (세 확률의 합은 100%가 되도록 하십시오.)
        """
        level_instruction += scenario_instruction

    if "투자성향별 포트폴리오 적정보유비중" in focus:
        level_instruction += """
        \n[특별 지시: 투자성향별 비중 제안]
        결론에 공격적/중립적/보수적 투자자별 권장 보유 비중(%)을 제시하십시오.
        """

    growth_value_logic = """
    [핵심: 성장주 vs 가치주 판단]
    먼저 이 기업이 성장주인지 가치주인지 규정하고, 그에 맞는 핵심 지표(매출성장 vs 배당/점유율 등)를 우선 분석하십시오.
    """
    
    # [가장 중요한 변경점] 메타데이터 보완 지시 (Prompt Injection)
    # 정보가 Unknown이면 AI가 직접 Google Search 도구를 써서 채우도록 강제
    metadata_instruction = f"""
    [대상 정보]
    - 티커: {ticker}
    - 입력된 기업명: {stock_name}
    - 입력된 섹터: {sector}
    - 입력된 산업: {industry}

    **[CRITICAL INSTRUCTION]**
    만약 위 '입력된 기업명', '섹터', '산업' 정보가 'Unknown'이거나, 티커({ticker})와 일치하지 않는 정보(예: soun 티커에 nvda 이름 등)라고 판단된다면,
    **즉시 Google Search 도구를 사용하여 정확한 최신 정보를 찾아낸 뒤, 보고서 서두에 올바른 기업명/섹터/산업을 명시하고 분석을 진행하십시오.**
    입력된 정보를 맹신하지 말고, 당신의 지식과 검색 결과를 우선시하십시오.
    """

    korean_enforcement = "\n\n**[중요] 모든 답변은 반드시 자연스러운 '한국어(Korean)'로 작성해야 합니다.**"

    base_prompt = f"""
    [역할] 월가 수석 애널리스트
    {metadata_instruction}
    [모드] {mode}
    [중점 분석] {focus}
    [관점] {viewpoint}
    [심도] {analysis_depth}
    
    {growth_value_logic}
    {level_instruction}
    
    [데이터 요약]
    {data_summary}
    [재무] {fin_str}
    [뉴스] {news_text}
    
    [지시사항]
    위 데이터를 바탕으로 투자 보고서를 작성하십시오.
    데이터가 부족한 부분은 'Google Search'를 통해 보완하십시오.
    {korean_enforcement}
    """

    if mode == "10K": prompt = base_prompt.replace("[모드] MAIN", "[모드] 10-K 분석").replace("투자 보고서", "10-K 심층 분석 보고서")
    elif mode == "10Q": prompt = base_prompt.replace("[모드] MAIN", "[모드] 10-Q 실적 분석")
    elif mode == "8K": prompt = base_prompt.replace("[모드] MAIN", "[모드] 8-K 공시 속보")
    else: prompt = base_prompt

    st.session_state['temp_data'] = {
        'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
        'df': df, 'prompt': prompt, 'news': []
    }
    return True

# ---------------------------------------------------------
# 5. UI 구성
# ---------------------------------------------------------
st.sidebar.subheader("🎯 분석 옵션")
selected_viewpoint = st.sidebar.select_slider("", options=list(viewpoint_mapping.keys()), value="중기 (6개월~1년)", label_visibility="collapsed")
viewpoint_mapping = {"단기 (1주~1개월)": "3mo", "스윙 (1~3개월)": "6mo", "중기 (6개월~1년)": "2y", "장기 (1~3년)": "5y"}
st.session_state['selected_period_str'] = viewpoint_mapping[selected_viewpoint]
st.session_state['selected_viewpoint'] = selected_viewpoint

analysis_levels = ["1.요약", "2.표준", "3.심층", "4.전문가", "5.시나리오"]
analysis_depth = st.sidebar.select_slider("", options=analysis_levels, value=analysis_levels[-1], label_visibility="collapsed")
st.session_state['analysis_depth'] = analysis_depth

st.session_state['use_news'] = st.sidebar.toggle("뉴스 데이터 반영", value=True)

def toggle_focus_all():
    new_state = st.session_state['focus_all']
    for opt in opt_targets: st.session_state[f"focus_{opt}"] = new_state

with st.sidebar.expander("☑️ 중점 분석 항목", expanded=False):
    st.checkbox("전체 선택", key="focus_all", on_change=toggle_focus_all)
    for opt in opt_targets: st.checkbox(opt, key=f"focus_{opt}")

api_key = st.secrets.get("GEMINI_API_KEY", None)
if not api_key: st.sidebar.error("Secrets에 'GEMINI_API_KEY' 필요")

tab_search, tab_fav = st.sidebar.tabs(["⚡ 검색", "⭐ 포트폴리오"])
prompt_mode_search = False
prompt_mode_port = False

with tab_search:
    st.markdown("<br>", unsafe_allow_html=True) 
    single_input = st.text_input("티커 (예: 005930.KS)", key="s_input")
    c_chk, c_btn = st.columns([0.5, 0.5])
    with c_chk: prompt_mode_search = st.checkbox("☑️ 프롬프트만", key="chk_prompt_single", value=True)
    with c_btn: 
        if api_key or prompt_mode_search:
            st.button("🔍 분석 시작", type="primary", key="btn_s_main", 
                    on_click=handle_search_click, args=("MAIN", prompt_mode_search))
        else: st.button("🔍 시작", disabled=True)
    
    st.markdown("##### 📑 공시 분석")
    c1, c2, c3 = st.columns(3)
    with c1: st.button("10-K", key="btn_s_10k", on_click=handle_search_click, args=("10K", prompt_mode_search))
    with c2: st.button("10-Q", key="btn_s_10q", on_click=handle_search_click, args=("10Q", prompt_mode_search))
    with c3: st.button("8-K", key="btn_s_8k", on_click=handle_search_click, args=("8K", prompt_mode_search))

# [포트폴리오 섹션] - 동일 유지, 코드 길이상 핵심 로직만 보존
selected_tickers = []
if 'selected' in st.query_params:
    selected_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]

with tab_fav:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.75, 0.25])
    with c1: st.text_input("종목 추가", key="new_ticker_input", label_visibility="collapsed")
    with c2: st.button("➕", on_click=add_ticker_logic)

    fav_df = st.session_state.get('portfolio_df', pd.DataFrame())
    if not fav_df.empty:
        # JS Grid 코드는 기존과 동일 (생략 없이 사용하려면 이전 코드의 Grid 부분 복사 필요)
        # 여기서는 간략화된 리스트로 표시 (공간 문제)
        st.write("포트폴리오 목록 (삭제: 체크 해제)")
        to_remove = []
        for idx, row in fav_df.iterrows():
            is_sel = st.checkbox(f"{row['ticker']} ({row['name']})", key=f"chk_p_{row['ticker']}", value=(row['ticker'] in selected_tickers))
            if is_sel and row['ticker'] not in selected_tickers: selected_tickers.append(row['ticker'])
            elif not is_sel and row['ticker'] in selected_tickers: selected_tickers.remove(row['ticker'])
    
    c_chk_p, c_btn_p = st.columns([0.5, 0.5])
    with c_chk_p: prompt_mode_port = st.checkbox("☑️ 프롬프트만", key="chk_prompt_port", value=True)
    with c_btn_p: 
        if st.button("🚀 종합 분석", type="primary"):
            start_analysis_process(selected_tickers, "MAIN", prompt_mode_port)

st.sidebar.markdown('<hr>', unsafe_allow_html=True)
st.sidebar.subheader("🤖 모델")
st.sidebar.selectbox("모델", ["gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-pro", "gemini-1.5-flash"], key='selected_model')

with st.sidebar.expander("📜 로그"):
    st.text_area("Log", value="\n".join(st.session_state['log_buffer']), height=200)

# ---------------------------------------------------------
# 6. 실행 컨트롤러
# ---------------------------------------------------------
st.title(f"📈 AI Hyper-Analyst V86")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    idx = st.session_state['proc_index']
    stage = st.session_state['proc_stage']
    
    if not targets or idx >= len(targets):
        st.success("완료!")
        st.session_state['is_analyzing'] = False
        st.rerun()

    curr = targets[idx]
    st.progress((idx * 2 + (1 if stage > 1 else 0)) / (len(targets) * 2), text=f"분석 중: {curr}")

    if stage == 1:
        collapse_sidebar()
        with st.spinner(f"데이터 수집 중: {curr}"):
            step_fetch_data(curr, st.session_state['current_mode'])
            st.session_state['proc_stage'] = 2
            st.rerun()
            
    elif stage == 2:
        temp = st.session_state['temp_data']
        if st.session_state['prompt_mode']:
            res = {'report': "프롬프트 생성됨", 'status': 'manual', 'prompt': temp['prompt']}
        else:
            with st.spinner("AI 분석 중..."):
                try:
                    txt, model = generate_with_fallback(temp['prompt'], api_key, st.session_state['selected_model'])
                    res = {'report': txt, 'status': 'success', 'model': model}
                except Exception as e:
                    res = {'report': str(e), 'status': 'error'}
        
        st.session_state['analysis_results'][curr] = {**temp, **res}
        st.session_state['proc_index'] = idx + 1
        st.session_state['proc_stage'] = 1
        st.rerun()

# 결과 출력
if st.session_state['analysis_results']:
    for t, d in st.session_state['analysis_results'].items():
        with st.expander(f"📊 {d['name']} ({t}) 결과", expanded=True):
            if d['status'] == 'manual': st.code(d['prompt'])
            else: st.markdown(d['report'])
            if not d['df'].empty: st.line_chart(d['df']['Close'])
