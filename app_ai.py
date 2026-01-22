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
from bs4 import BeautifulSoup  # [추가] 2차 방어선(웹 크롤링)용

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
    page_title="AI Hyper-Analyst V86 (Robust)", 
    page_icon="📈",
    initial_sidebar_state=st.session_state['sidebar_state']
)

# [로그 시스템] 초기화
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
        add_log("📥 [INIT] 포트폴리오 데이터 로드...")
        if os.path.exists(CSV_FILE):
            try:
                df = pd.read_csv(CSV_FILE)
                if df.empty:
                    st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
                else:
                    st.session_state['portfolio_df'] = df.reset_index(drop=True)
                    add_log(f"✅ [INIT] 로드 완료: {len(df)}개.")
            except:
                st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
        else:
            st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])

def save_state_to_csv():
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df'].reset_index(drop=True)
        st.session_state['portfolio_df'] = df 
        try:
            df.to_csv(CSV_FILE, index=False, encoding='utf-8')
            add_log("💾 [SAVE] 저장 완료.")
        except Exception as e:
            add_log(f"❌ [SAVE] 실패: {e}")

def add_ticker_logic():
    raw_input = st.session_state.get('new_ticker_input', '')
    if raw_input:
        tickers = [t.strip().upper() for t in raw_input.split(',')]
        df = st.session_state['portfolio_df']
        existing = df['ticker'].values
        new_rows = []
        for t in tickers:
            if t and t not in existing:
                # 간단 추가 시에는 이름만 빠르게 조회
                try: 
                    meta = get_robust_metadata(t) # 아래 정의된 강력한 함수 사용
                    name = meta['name']
                except: name = t
                new_rows.append({'ticker': t, 'name': name})
        
        if new_rows:
            st.session_state['portfolio_df'] = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_state_to_csv()
    st.session_state['new_ticker_input'] = ""

load_data_to_state()

# 삭제 로직
if 'del_ticker' in st.query_params:
    del_ticker = st.query_params['del_ticker']
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        st.session_state['portfolio_df'] = df[df['ticker'] != del_ticker]
        save_state_to_csv()
        if f"chk_{del_ticker}" in st.session_state: del st.session_state[f"chk_{del_ticker}"]
    st.query_params.clear()
    st.rerun()

# ---------------------------------------------------------
# 3. 유틸리티 및 강력한 데이터 수집 함수 (핵심)
# ---------------------------------------------------------
def get_robust_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session

def run_with_timeout(func, args=(), timeout=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try: return future.result(timeout=timeout)
        except: return None

# [핵심 솔루션] 다중 방어 메타데이터 수집 함수
def get_robust_metadata(ticker):
    """
    1차: yfinance info
    2차: yfinance fast_info
    3차: Yahoo Finance HTML Title Scraping
    """
    metadata = {"name": ticker, "sector": "Unknown", "industry": "Unknown"}
    add_log(f"🕵️ [META] {ticker} 메타데이터 수집 시작 (다중 방어)")

    # [1차 시도] yfinance info (가장 상세함)
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info # 타임아웃 가능성 있음
        if info and 'shortName' in info:
            metadata['name'] = info.get('shortName') or info.get('longName') or ticker
            metadata['sector'] = info.get('sector', 'Unknown')
            metadata['industry'] = info.get('industry', 'Unknown')
            add_log(f"   ✅ [1차] yfinance 성공: {metadata['name']}")
            return metadata
    except Exception as e:
        add_log(f"   ⚠️ [1차] 실패: {e}")

    # [2차 시도] yfinance fast_info (빠르고 가벼움 - 섹터 정보는 없을 수 있으나 이름 확보용)
    try:
        fast_info = yf.Ticker(ticker).fast_info
        # fast_info는 sector 정보가 없지만 currency 등은 있음. 이름이라도 건지기.
        # fast_info에는 종목명 명시적 필드가 없을 수 있어 건너뛸 수도 있음.
        pass 
    except: pass

    # [3차 시도] Web Scraping (HTML Title 파싱) - 가장 강력한 차선책
    if metadata['name'] == ticker or metadata['sector'] == "Unknown":
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            session = get_robust_session()
            resp = session.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                title_text = soup.title.string if soup.title else ""
                # Title 예시: "SoundHound AI, Inc. (SOUN) Stock Price..."
                if title_text:
                    # 괄호 앞부분 추출
                    extracted_name = title_text.split('(')[0].strip()
                    if extracted_name:
                        metadata['name'] = extracted_name
                        add_log(f"   ✅ [3차] 웹 스크래핑으로 이름 복구: {extracted_name}")
        except Exception as e:
            add_log(f"   ⚠️ [3차] 웹 스크래핑 실패: {e}")

    return metadata

def _fetch_history(ticker, period): return yf.Ticker(ticker).history(period=period)

def clean_html_text(text):
    if not text: return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = html.unescape(clean)
    return " ".join(clean.split())

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
    except: return []

def get_realtime_news(ticker, name):
    add_log(f"📰 [NEWS] 뉴스 검색: {ticker}")
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    if not is_kr:
        try:
            rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            items = fetch_rss_realtime(rss_url, limit=5)
            for i in items: i['source'] = "Yahoo"
            news_items.extend(items)
        except: pass

    if is_kr: search_query = f'"{name}"'
    else: search_query = f'{ticker} stock'
    
    q_encoded = urllib.parse.quote(search_query)
    url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ko&gl=KR&ceid=KR:ko"
    g_items = fetch_rss_realtime(url, limit=5)
    for i in g_items: i['source'] = "Google"
    news_items.extend(g_items)
    
    return news_items[:7]

def get_financial_metrics(ticker):
    # 재무지표도 별도 타임아웃 관리
    try:
        info = run_with_timeout(lambda: yf.Ticker(ticker).info, timeout=4)
        if not info: return {}
        def get_fmt(key): val = info.get(key); return f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A"
        return {
            "Free Cash Flow": get_fmt('freeCashflow'), "Current Ratio": get_fmt('currentRatio'),
            "Debt to Equity": get_fmt('debtToEquity'), "ROE": get_fmt('returnOnEquity'),
            "Total Revenue": get_fmt('totalRevenue'), "Net Income": get_fmt('netIncome')
        }
    except: return {}

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
    # Gemini 모델 체인 (fallback)
    chain = [start_model, "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.0-pro"]
    unique_chain = []
    [unique_chain.append(x) for x in chain if x not in unique_chain]
    
    for model_name in unique_chain:
        try:
            add_log(f"🧠 [AI] 요청: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name 
        except Exception as e:
            add_log(f"   ⚠️ 실패 ({model_name}): {e}")
            time.sleep(0.5); continue
    raise Exception("All models failed")

def handle_search_click(mode, is_prompt):
    raw_input = st.session_state.get("s_input", "")
    if raw_input:
        targets = [t.strip() for t in raw_input.split(',') if t.strip()]
        start_analysis_process(targets, mode, is_prompt)
    else: st.warning("티커 입력 필요")

def step_fetch_data(ticker, mode):
    add_log(f"==========================================")
    add_log(f"📦 [STEP 1] 데이터 수집: {ticker} ({mode})")
    
    # 1. 메타데이터 확보 (다중 방어 적용)
    # 스레드풀로 감싸서 행여나 함수 내부에서 멈추는 것 방지
    meta = run_with_timeout(get_robust_metadata, args=(ticker,), timeout=8)
    if not meta: meta = {"name": ticker, "sector": "Unknown", "industry": "Unknown"}
    
    stock_name = meta['name']
    sector = meta['sector']
    industry = meta['industry']
    
    clean_code = re.sub(r'[^0-9]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker

    try:
        # 2. 주가 데이터
        period = st.session_state.get('selected_period_str', '1y')
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=8)
        
        if df is None: df = pd.DataFrame()
        
        data_summary = "No Data"
        if not df.empty:
            curr = df['Close'].iloc[-1]; high_val = df['High'].max(); low_val = df['Low'].min()
            display_df = df.tail(60)
            data_summary = f"[Price Stats] High: {high_val:.2f}, Low: {low_val:.2f}, Cur: {curr:.2f}\n[Trend]\n{display_df.to_string()}"
        
        # 3. 추가 정보
        fin_str = "N/A"; news_text = "N/A"
        if mode not in ["10K", "10Q", "8K"]:
            fm = get_financial_metrics(ticker)
            fin_str = str(fm) if fm else "N/A"
            
            if st.session_state.get('use_news', True):
                news = get_realtime_news(ticker, stock_name)
                if news:
                    news_lines = [f"- [{n['source']}] {n['title']} ({n['date_str']})" for n in news]
                    news_text = "\n".join(news_lines)
                else: news_text = "최신 뉴스 없음"

        # 4. 프롬프트 구성
        selected_focus_list = []
        for opt in opt_targets:
            if st.session_state.get(f"focus_{opt}", True): selected_focus_list.append(opt)
        focus = ", ".join(selected_focus_list)
        
        viewpoint = st.session_state.get('selected_viewpoint', 'General')
        analysis_depth = st.session_state.get('analysis_depth', "2. 표준")
        
        # [시나리오 확률 요청 추가]
        scenario_instruction = ""
        if "5." in analysis_depth:
            scenario_instruction = """
            [시나리오 분석 필수 지침]
            - 낙관적(Bull), 기본(Base), 비관적(Bear) 시나리오 3가지를 제시하십시오.
            - **각 시나리오마다 '실현 확률(%)'을 반드시 명시하고, 그 확률을 산정한 논리적 근거를 설명하십시오.**
            - 예: "낙관적 시나리오 (확률: 20%): 이유는 ~이기 때문입니다."
            """

        # [투자성향 비중 요청]
        portfolio_instruction = ""
        if "투자성향별 포트폴리오 적정보유비중" in focus:
            portfolio_instruction = """
            [투자성향별 비중 제안]
            결론부에 다음 3가지 성향별 권장 보유 비중(%)과 논리를 서술하십시오:
            1. 🦁 공격적 (Aggressive)
            2. ⚖️ 중립적 (Moderate)
            3. 🛡️ 보수적 (Conservative)
            """

        # [성장주/가치주 구분 로직]
        growth_value_logic = """
        [성장주 vs 가치주 판단 및 분석]
        1. 이 기업이 '성장주'인지 '가치주'인지 규정하고 이유를 설명하십시오.
        2. 성장주라면: 매출성장률, Cash Flow 증가, ROI 개선, 마진 흑자전환 여부를 중점 분석.
        3. 가치주라면: 시장점유율, 배당 안정성, 주가 변동성, 이익률 추이를 중점 분석.
        """
        
        # [최종병기] 메타데이터 보정 지시사항 (Gemini에게 위임)
        metadata_instruction = f"""
        [대상 정보]
        - 티커: {ticker}
        - 기업명(Python 추출): {stock_name}
        - 섹터(Python 추출): {sector}
        - 산업(Python 추출): {industry}
        
        **[중요] 만약 위 '기업명', '섹터', '산업' 정보가 'Unknown'이거나 티커와 동일하다면, 귀하(AI)의 지식 베이스를 활용하여 정확한 정보로 대체하여 분석하십시오. 절대 'Unknown'이라고 출력하지 마십시오.**
        """

        korean_enforcement = "\n\n**[중요] 모든 답변은 반드시 자연스러운 '한국어(Korean)'로 작성해야 합니다.**"

        base_prompt = f"""
        [역할] 월가 수석 애널리스트
        {metadata_instruction}
        [모드] {mode}
        [중점 분석] {focus}
        [투자 관점] {viewpoint}
        [분석 레벨] {analysis_depth}
        
        {growth_value_logic}
        {scenario_instruction}
        {portfolio_instruction}
        
        [데이터 요약]
        {data_summary}
        
        [재무 지표]
        {fin_str}
        
        [뉴스]
        {news_text}
        
        위 데이터를 바탕으로 전문적인 보고서를 작성하십시오. 뉴스 내용도 반영하십시오.
        결론에는 [매수 / 매도 / 관망] 의견을 명확히 하십시오.
        {korean_enforcement}
        """

        # 공시 모드 프롬프트는 간소화하여 처리 (지면 관계상 핵심만 전달)
        if mode in ["10K", "10Q", "8K"]:
            prompt = f"""
            [역할] 전문 공시 분석가
            [대상] {ticker} ({stock_name})
            [자료] SEC {mode} 보고서
            {metadata_instruction}
            
            위 기업의 {mode} 보고서 내용을 바탕으로 핵심을 분석하십시오.
            **'Unknown' 정보는 AI 지식으로 채우십시오.**
            {korean_enforcement}
            """
        else:
            prompt = base_prompt

        st.session_state['temp_data'] = {
            'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': []
        }
        add_log(f"✅ [STEP 1] 완료. Prompt 준비됨.")
        return True

    except Exception as e:
        add_log(f"❌ [FATAL] Step 1 Error: {e}")
        return False

# ---------------------------------------------------------
# 5. 사이드바 UI
# ---------------------------------------------------------
st.sidebar.subheader("🎯 분석 옵션")

viewpoint_mapping = {"단기": "3mo", "스윙": "6mo", "중기": "2y", "장기": "5y"}
selected_viewpoint = st.sidebar.select_slider("", options=list(viewpoint_mapping.keys()), value="중기", label_visibility="collapsed")
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

# API Key
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.sidebar.error("Secrets에 'GEMINI_API_KEY' 필요")

# 탭
tab_search, tab_fav = st.sidebar.tabs(["⚡ 검색", "⭐ 포트폴리오"])
prompt_mode_search = False
prompt_mode_port = False

with tab_search:
    st.markdown("<br>", unsafe_allow_html=True) 
    st.text_input("티커 (예: SOUN, 005930.KS)", key="s_input")
    c1, c2 = st.columns(2)
    with c1: prompt_mode_search = st.checkbox("프롬프트만", key="chk_p_s", value=True)
    with c2: 
        if st.button("🔍 시작", key="btn_s"):
            handle_search_click("MAIN", prompt_mode_search)
    
    st.markdown("##### 📑 공시")
    c1, c2, c3 = st.columns(3)
    with c1: st.button("10-K", key="b_10k", on_click=handle_search_click, args=("10K", prompt_mode_search))
    with c2: st.button("10-Q", key="b_10q", on_click=handle_search_click, args=("10Q", prompt_mode_search))
    with c3: st.button("8-K", key="b_8k", on_click=handle_search_click, args=("8K", prompt_mode_search))

with tab_fav:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.75, 0.25])
    with c1: st.text_input("추가 (AAPL, TSLA)", key="new_ticker_input", label_visibility="collapsed")
    with c2: st.button("➕", on_click=add_ticker_logic)

    fav_df = st.session_state.get('portfolio_df', pd.DataFrame())
    selected_tickers = []
    if 'selected' in st.query_params and st.query_params['selected']:
        selected_tickers = [t.strip() for t in st.query_params['selected'].split(',')]

    if not fav_df.empty:
        # JSON 변환 및 JS Grid (간소화됨)
        import json
        t_data = [{'ticker': r['ticker'], 'name': str(r['name'])} for i, r in fav_df.iterrows()]
        t_json = json.dumps(t_data); s_json = json.dumps(selected_tickers)
        
        # Grid HTML/JS (기존 로직 유지하되 간소화하여 삽입)
        grid_html = f"""
        <style>.pf-item {{padding:5px; border:1px solid #ddd; margin:2px; cursor:pointer; background:#f9f9f9; display:flex; align-items:center;}} .pf-item.sel {{background:#e0f2fe; border-color:#3b82f6;}}</style>
        <div id="grid"></div>
        <script>
        const data={t_json}; let sel={s_json};
        const grid=document.getElementById('grid');
        data.forEach(d=>{{
            const div=document.createElement('div');
            div.className='pf-item'+(sel.includes(d.ticker)?' sel':'');
            div.innerHTML=`<div style="flex:1"><b>${{d.ticker}}</b><br><small>${{d.name}}</small></div><button onclick="del('${{d.ticker}}')">×</button>`;
            div.onclick=(e)=>{{ if(e.target.tagName!='BUTTON') toggle(d.ticker); }};
            grid.appendChild(div);
        }});
        function toggle(t){{
            if(sel.includes(t)) sel=sel.filter(x=>x!==t); else sel.push(t);
            const url=new URL(window.parent.location.href);
            if(sel.length) url.searchParams.set('selected',sel.join(',')); else url.searchParams.delete('selected');
            window.parent.history.replaceState(null,'',url.toString());
            window.parent.location.reload();
        }}
        function del(t){{
            const url=new URL(window.parent.location.href);
            url.searchParams.set('del_ticker',t);
            window.parent.location.href=url.toString();
        }}
        </script>
        """
        st.components.v1.html(grid_html, height=300, scrolling=True)

    prompt_mode_port = st.checkbox("프롬프트만", key="chk_p_p", value=True)
    if st.button("🚀 종합 분석", type="primary"):
        if selected_tickers: start_analysis_process(selected_tickers, "MAIN", prompt_mode_port)
        else: st.warning("선택된 종목 없음")

# 모델 선택
st.sidebar.markdown('<hr>', unsafe_allow_html=True)
model_opts = ["gemini-1.5-pro", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash"]
sel_model = st.sidebar.selectbox("모델", model_opts)
st.session_state['selected_model'] = sel_model

# 로그창
with st.sidebar.expander("📜 로그", expanded=False):
    st.text_area("", value="\n".join(st.session_state['log_buffer']), height=200)

# ---------------------------------------------------------
# 6. 실행 컨트롤러
# ---------------------------------------------------------
st.title("📈 AI Hyper-Analyst V86 (Robust)")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    idx = st.session_state['proc_index']
    stage = st.session_state['proc_stage']
    
    if idx >= len(targets):
        st.success("완료!")
        st.session_state['is_analyzing'] = False
        st.stop()

    curr_ticker = targets[idx]
    st.progress((idx * 2 + (1 if stage > 1 else 0)) / (len(targets) * 2), text=f"분석 중: {curr_ticker}")

    if stage == 1:
        collapse_sidebar()
        with st.spinner(f"📥 {curr_ticker} 데이터 수집 중..."):
            if step_fetch_data(curr_ticker, st.session_state['current_mode']):
                st.session_state['proc_stage'] = 2
            else:
                st.session_state['analysis_results'][curr_ticker] = {'status': 'error', 'report': '데이터 실패'}
                st.session_state['proc_index'] = idx + 1
            st.rerun()

    elif stage == 2:
        temp = st.session_state['temp_data']
        if st.session_state['prompt_mode']:
            st.session_state['analysis_results'][curr_ticker] = {
                'name': temp['name'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr'],
                'df': temp['df'], 'report': "프롬프트 생성됨", 'status': 'manual', 'prompt': temp['prompt']
            }
        else:
            with st.spinner("🧠 AI 분석 중..."):
                try:
                    rep, model = generate_with_fallback(temp['prompt'], api_key, st.session_state['selected_model'])
                    st.session_state['analysis_results'][curr_ticker] = {
                        'name': temp['name'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr'],
                        'df': temp['df'], 'report': sanitize_text(rep), 'status': 'success', 'model': model
                    }
                except Exception as e:
                    st.session_state['analysis_results'][curr_ticker] = {'status': 'error', 'report': str(e)}
        
        st.session_state['proc_index'] = idx + 1
        st.session_state['proc_stage'] = 1
        st.rerun()

# ---------------------------------------------------------
# 7. 결과 출력
# ---------------------------------------------------------
if not st.session_state['is_analyzing'] and st.session_state['analysis_results']:
    st.write("---")
    for ticker, data in st.session_state['analysis_results'].items():
        if data.get('status') == 'manual':
            with st.expander(f"📋 {ticker} 프롬프트", expanded=True):
                st.link_button("Gemini 열기", "https://gemini.google.com/")
                st.code(data['prompt'], language='text')
        elif data.get('status') == 'success':
            with st.expander(f"📊 {data['name']} ({ticker}) 분석 결과", expanded=True):
                st.caption(f"Model: {data.get('model')}")
                if not data['df'].empty:
                    if data['is_kr']:
                        fig = go.Figure(data=[go.Candlestick(x=data['df'].index, open=data['df']['Open'], high=data['df']['High'], low=data['df']['Low'], close=data['df']['Close'])])
                        fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10), xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.components.v1.html(f"""<div id="c_{ticker}" style="height:350px"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{autosize:true, symbol:"{data['tv_symbol']}", interval:"D", timezone:"Asia/Seoul", theme:"light", style:"1", container_id:"c_{ticker}"}});</script>""", height=360)
                st.markdown(data['report'])
                st.markdown("---")
                st.code(data['report'])
        else:
            st.error(f"{ticker}: {data.get('report')}")
