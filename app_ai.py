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
    page_title="AI Hyper-Analyst V84", 
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

# [변수 정의]
opt_targets = [
    "현금건전성 지표 (FCF, 유동비율, 부채비율)", 
    "핵심 재무제표 분석 (손익, 대차대조, 현금흐름)",
    "투자기관 목표주가 및 컨센서스", 
    "호재/악재 뉴스 판단", 
    "기술적 지표 (RSI/이평선)",
    "외국인/기관 수급 분석", 
    "경쟁사 비교 및 업황", 
    "단기/중기 매매 전략"
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

# 체크박스 상태
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
                try: 
                    # [수정] 추가 시에도 longName 우선 확보 시도
                    add_log(f"🔍 [ADD] {ticker} 정보 조회 중...")
                    t_obj = yf.Ticker(ticker)
                    t_info = t_obj.info
                    # longName(공식명) -> shortName -> ticker 순서
                    name = t_info.get('longName') or t_info.get('shortName') or ticker
                    add_log(f"   -> 이름 식별: {name}")
                except Exception as e: 
                    name = ticker
                    add_log(f"   ⚠️ [ADD] 정보 조회 실패, 티커 사용. Error: {e}")
                
                new_rows.append({'ticker': ticker, 'name': name})
            
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            st.session_state['portfolio_df'] = df
            save_state_to_csv()
            add_log("✅ [ADD] 신규 티커 저장 완료.")
            
    st.session_state['new_ticker_input'] = ""

load_data_to_state()

# ---------------------------------------------------------
# [삭제 로직]
# ---------------------------------------------------------
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
# 3. 유틸리티 함수 (핵심 수정 포함)
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

def _fetch_info_robust(ticker):
    """[신규] yfinance info를 더 확실하게 가져오기 위한 래퍼"""
    try:
        return yf.Ticker(ticker).info
    except:
        return None

def get_official_company_name(ticker):
    """
    [핵심 수정] 공식 기업명 가져오기 로직 강화
    1. 포트폴리오에 저장된 이름 확인
    2. yfinance의 longName (공식 법인명) 최우선 조회
    3. 실패 시 shortName 조회
    4. 모두 실패 시 None 반환 (티커 반환 X)
    """
    # 1. 포트폴리오 확인
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        row = df[df['ticker'] == ticker]
        if not row.empty:
            saved_name = row.iloc[0]['name']
            # 저장된 이름이 티커와 다르면(즉, 유효한 이름이면) 사용
            if saved_name != ticker:
                return saved_name

    # 2. yfinance 조회 (타임아웃 8초로 넉넉하게)
    info = run_with_timeout(_fetch_info_robust, args=(ticker,), timeout=8)
    
    if info:
        # longName이 가장 공식적인 이름 (예: Apple Inc.)
        long_name = info.get('longName')
        if long_name: return long_name
        
        # 없으면 shortName (예: Apple)
        short_name = info.get('shortName')
        if short_name: return short_name

    return None  # 이름을 못 찾음

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
    add_log(f"   🌐 [RSS] URL 요청: {url}")
    try:
        session = get_robust_session()
        response = session.get(url, timeout=5)
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall('./channel/item')[:limit]:
            title = item.find('title').text
            link = item.find('link').text
            pubDate = item.find('pubDate').text
            description = ""
            desc_elem = item.find('description')
            if desc_elem is not None and desc_elem.text:
                description = clean_html_text(desc_elem.text)
            try: dt = parser.parse(pubDate); date_str = dt.strftime("%m-%d %H:%M")
            except: date_str = "최신"
            items.append({'title': title, 'link': link, 'date_str': date_str, 'summary': description})
        return items
    except Exception as e:
        add_log(f"   ❌ [RSS] 파싱 에러: {e}")
        return []

def get_realtime_news_by_name(company_name):
    """
    [핵심 수정] 오직 '공식 기업명'으로만 뉴스를 검색합니다. (티커 사용 안함)
    """
    if not company_name:
        add_log("   ⚠️ 기업명이 식별되지 않아 뉴스 검색을 건너뜁니다.")
        return []

    add_log(f"📰 [NEWS] 공식 기업명으로 뉴스 검색 시작: '{company_name}'")
    
    # 검색어에 따옴표를 붙여 정확한 구문 검색 ("Apple Inc.")
    search_query = f'"{company_name}"'
    
    try:
        q_encoded = urllib.parse.quote(search_query)
        # Google News RSS (한국어/한국 설정)
        url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ko&gl=KR&ceid=KR:ko"
        
        google_news = fetch_rss_realtime(url, limit=7)
        for n in google_news: n['source'] = "Google News"
        
        if not google_news:
            add_log(f"   ⚠️ '{company_name}'에 대한 검색 결과 없음.")
            
        return google_news

    except Exception as e:
        add_log(f"   ❌ 뉴스 검색 실패: {e}")
        return []

def get_financial_metrics(ticker):
    add_log(f"📊 [FIN] 재무 지표 조회: {ticker}")
    info = run_with_timeout(_fetch_info_robust, args=(ticker,), timeout=5)
    if not info: return {}
    try:
        def get_fmt(key): val = info.get(key); return f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A"
        metrics = {
            "Free Cash Flow": get_fmt('freeCashflow'), "Current Ratio": get_fmt('currentRatio'),
            "Quick Ratio": get_fmt('quickRatio'), "Debt to Equity": get_fmt('debtToEquity'),
            "Return on Equity (ROE)": get_fmt('returnOnEquity'), "Total Revenue": get_fmt('totalRevenue'),
            "Net Income": get_fmt('netIncome')
        }
        return metrics
    except: return {}

def sanitize_text(text):
    text = text.replace('$', '\$'); text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    return text

def collapse_sidebar():
    js = """<script>var closeBtn = window.parent.document.querySelector('[data-testid="stSidebarExpandedControl"]');if (closeBtn) {closeBtn.click();}</script>"""
    st.components.v1.html(js, height=0, width=0)

# ---------------------------------------------------------
# 4. 분석 프로세스 로직
# ---------------------------------------------------------
def start_analysis_process(targets, mode, is_prompt_only):
    add_log(f"▶️ [PROCESS] 분석 시작: {len(targets)}개 종목")
    st.session_state['is_analyzing'] = True
    st.session_state['targets_to_run'] = targets
    st.session_state['current_mode'] = mode
    st.session_state['prompt_mode'] = is_prompt_only
    st.session_state['analysis_results'] = {} 
    st.session_state['proc_index'] = 0
    st.session_state['proc_stage'] = 1 

def generate_with_fallback(prompt, api_key, start_model):
    genai.configure(api_key=api_key)
    fallback_chain = [start_model, "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.0-pro"]
    
    for model_name in fallback_chain:
        try:
            add_log(f"   🤖 Model: {model_name} 시도...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name 
        except Exception as e:
            add_log(f"   ⚠️ 실패 ({model_name}): {e}")
            time.sleep(0.5)
            continue
    raise Exception("모든 모델 시도 실패")

def step_fetch_data(ticker, mode):
    add_log(f"==========================================")
    add_log(f"📦 [STEP 1] 데이터 수집: {ticker}")
    
    # [수정] 공식 기업명 가져오기 (실패 시 티커가 아닌 None 반환됨)
    official_name = get_official_company_name(ticker)
    
    # 만약 공식 이름을 못 찾았다면, 어쩔 수 없이 티커를 쓰되, 뉴스 검색은 안 함
    display_name = official_name if official_name else ticker
    
    add_log(f"   🏷️ 식별된 공식 기업명: {official_name if official_name else '식별 실패 (티커 사용)'}")

    clean_code = re.sub(r'[^0-9]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker

    try:
        # 주가 데이터
        period = st.session_state.get('selected_period_str', '1y')
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=10)
        
        if df is None: df = pd.DataFrame()
        
        data_summary = "No Data"
        if not df.empty:
            curr = df['Close'].iloc[-1]; high_val = df['High'].max(); low_val = df['Low'].min()
            stats_str = f"High: {high_val:.2f}, Low: {low_val:.2f}, Current: {curr:.2f}"
            display_df = df.tail(60); recent_days = df.tail(5)
            data_summary = f"[Stats] {stats_str}\n[Trend]\n{display_df.to_string()}\n[Recent]\n{recent_days.to_string()}"

        fin_str = "N/A"; news_text = "N/A"
        
        if mode not in ["10K", "10Q", "8K"]:
            try: 
                fm = get_financial_metrics(ticker)
                fin_str = str(fm) if fm else "N/A"
            except: pass
            
            if st.session_state.get('use_news', True):
                if official_name:
                    # [수정] 무조건 공식 이름으로만 검색
                    news = get_realtime_news_by_name(official_name)
                    if news: 
                        formatted_news = []
                        for n in news:
                            title = n['title']
                            summary = n.get('summary', '')
                            if is_similar(title, summary): summary = ""
                            elif len(summary) > 200: summary = summary[:200] + "..."
                            item_str = f"- [{n['source']}] {title} ({n['date_str']})"
                            if summary: item_str += f"\n  > 요약: {summary}"
                            formatted_news.append(item_str)
                        news_text = "\n".join(formatted_news)
                    else: news_text = f"'{official_name}'에 대한 최신 뉴스가 없습니다."
                else:
                    news_text = "⚠️ 공식 기업명을 식별하지 못해 뉴스 검색을 생략했습니다."

        selected_focus_list = []
        for opt in opt_targets:
            if st.session_state.get(f"focus_{opt}", True): selected_focus_list.append(opt)
        focus = ", ".join(selected_focus_list)
        viewpoint = st.session_state.get('selected_viewpoint', 'General')
        analysis_depth = st.session_state.get('analysis_depth', "2. 표준 브리핑 (Standard)")
        
        # 프롬프트 조립
        prompt = f"""
        [역할] 월가 수석 애널리스트
        [대상] {ticker} (공식명: {display_name})
        [모드] {mode}
        [중점] {focus}
        [관점] {viewpoint}
        [깊이] {analysis_depth}
        
        **주의: 분석 대상은 반드시 '{display_name}' 기업이어야 합니다.**
        
        [주가 데이터]
        {data_summary}
        
        [재무 지표]
        {fin_str}
        
        [최신 뉴스 (공식 기업명 '{display_name}' 검색 결과)]
        {news_text}
        
        [지시사항]
        위 데이터를 바탕으로 전문적인 투자 보고서를 작성하십시오.
        특히 뉴스는 기업명으로 정확히 검색된 내용이므로, 분석에 적극 반영하십시오.
        결론에는 [매수 / 매도 / 관망] 의견을 명확히 하십시오.
        """
        
        # 공시 모드 프롬프트 오버라이드 (간략화)
        if mode in ["10K", "10Q", "8K"]:
            prompt = f"""
            [역할] 전문 공시 분석가
            [대상] {ticker} (공식명: {display_name})
            [자료] SEC {mode} 보고서
            
            위 기업의 해당 공시 내용을 심층 분석하여 핵심 변경사항, 리스크, 기회 요인을 정리하십시오.
            """

        st.session_state['temp_data'] = {
            'name': display_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': []
        }
        return True

    except Exception as e:
        add_log(f"❌ [FATAL] Step 1 Error: {str(e)}")
        return False

# ---------------------------------------------------------
# 5. 사이드바 UI
# ---------------------------------------------------------
st.sidebar.subheader("🎯 분석 옵션")

viewpoint_mapping = {"단기 (1주~1개월)": "3mo", "스윙 (1~3개월)": "6mo", "중기 (6개월~1년)": "2y", "장기 (1~3년)": "5y"}
selected_viewpoint = st.sidebar.select_slider("", options=list(viewpoint_mapping.keys()), value="중기 (6개월~1년)")
st.session_state['selected_period_str'] = viewpoint_mapping[selected_viewpoint]
st.session_state['selected_viewpoint'] = selected_viewpoint

analysis_levels = ["1.요약", "2.표준", "3.심층", "4.전문가", "5.시나리오"]
analysis_depth = st.sidebar.select_slider("", options=analysis_levels, value=analysis_levels[-1])
st.session_state['analysis_depth'] = analysis_depth

st.session_state['use_news'] = st.sidebar.toggle("뉴스 데이터 반영", value=True)

with st.sidebar.expander("☑️ 중점 분석 항목"):
    st.checkbox("전체 선택", key="focus_all", value=True)
    for opt in opt_targets: st.checkbox(opt, key=f"focus_{opt}")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.sidebar.error("⚠️ GEMINI_API_KEY 필요")

tab_search, tab_fav = st.sidebar.tabs(["⚡ 검색", "⭐ 포트폴리오"])
prompt_mode = False

def handle_search_click(mode, is_prompt):
    raw_input = st.session_state.get("s_input", "")
    if raw_input:
        targets = [t.strip() for t in raw_input.split(',') if t.strip()]
        start_analysis_process(targets, mode, is_prompt)

with tab_search:
    st.markdown("<br>", unsafe_allow_html=True) 
    st.text_input("티커 (예: 005930.KS)", key="s_input")
    prompt_mode = st.checkbox("프롬프트만 생성", key="chk_prompt_s", value=True)
    if st.button("🔍 분석 시작", type="primary"):
        handle_search_click("MAIN", prompt_mode)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.button("10-K", on_click=handle_search_click, args=("10K", prompt_mode))
    with c2: st.button("10-Q", on_click=handle_search_click, args=("10Q", prompt_mode))
    with c3: st.button("8-K", on_click=handle_search_click, args=("8K", prompt_mode))

with tab_fav:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.75, 0.25])
    with c1: st.text_input("종목 추가", key="new_ticker_input", placeholder="AAPL")
    with c2: st.button("➕", on_click=add_ticker_logic)

    fav_df = st.session_state.get('portfolio_df', pd.DataFrame())
    selected_tickers = []
    
    if not fav_df.empty:
        # 간단한 리스트 형태로 표시 (HTML Grid 대신 Streamlit 네이티브 사용)
        with st.expander(f"포트폴리오 ({len(fav_df)})", expanded=True):
            for idx, row in fav_df.iterrows():
                chk = st.checkbox(f"{row['ticker']} ({row['name']})", key=f"chk_{row['ticker']}")
                if chk: selected_tickers.append(row['ticker'])
                
        prompt_mode_p = st.checkbox("프롬프트만 생성", key="chk_prompt_p", value=True)
        if st.button("🚀 선택 종목 분석", type="primary"):
            start_analysis_process(selected_tickers, "MAIN", prompt_mode_p)

# AI 모델 선택
st.sidebar.markdown('<hr>', unsafe_allow_html=True)
model_options = ["gemini-1.5-pro", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash"]
st.session_state['selected_model'] = st.sidebar.selectbox("AI 모델", model_options)

with st.sidebar.expander("📜 로그"):
    st.text_area("", value="\n".join(st.session_state['log_buffer']), height=200)

# ---------------------------------------------------------
# 6. 메인 실행 로직
# ---------------------------------------------------------
st.title(f"📈 AI Hyper-Analyst V84")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    curr_idx = st.session_state['proc_index']
    
    if not targets or curr_idx >= len(targets):
        st.session_state['is_analyzing'] = False
        st.success("분석 완료!")
        st.stop()
        
    curr_ticker = targets[curr_idx]
    stage = st.session_state['proc_stage']
    
    st.progress((curr_idx * 2 + stage) / (len(targets) * 2), text=f"Analyzing {curr_ticker}...")

    if stage == 1:
        with st.spinner(f"데이터 수집 중: {curr_ticker}"):
            if step_fetch_data(curr_ticker, st.session_state['current_mode']):
                st.session_state['proc_stage'] = 2
            else:
                st.session_state['analysis_results'][curr_ticker] = {'report': "실패", 'status': 'error'}
                st.session_state['proc_index'] += 1
            st.rerun()

    elif stage == 2:
        temp = st.session_state['temp_data']
        if st.session_state['prompt_mode']:
            res = {'report': "프롬프트 생성됨", 'prompt': temp['prompt'], 'status': 'manual', 
                   'name': temp['name'], 'df': temp['df'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr']}
        else:
            with st.spinner("보고서 작성 중..."):
                try:
                    txt, model = generate_with_fallback(temp['prompt'], api_key, st.session_state['selected_model'])
                    res = {'report': sanitize_text(txt), 'status': 'success', 'model': model,
                           'name': temp['name'], 'df': temp['df'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr']}
                except Exception as e:
                    res = {'report': f"Error: {e}", 'status': 'error'}
        
        st.session_state['analysis_results'][curr_ticker] = res
        st.session_state['proc_index'] += 1
        st.session_state['proc_stage'] = 1
        st.rerun()

# ---------------------------------------------------------
# 7. 결과 표시
# ---------------------------------------------------------
if not st.session_state['is_analyzing'] and st.session_state['analysis_results']:
    for ticker, data in st.session_state['analysis_results'].items():
        with st.expander(f"📊 {data.get('name', ticker)} 결과", expanded=True):
            if not data.get('df', pd.DataFrame()).empty:
                st.line_chart(data['df']['Close'])
            
            if data.get('status') == 'manual':
                st.code(data.get('prompt'), language='text')
                st.link_button("Gemini 열기", "https://gemini.google.com/")
            else:
                st.markdown(data.get('report'))
