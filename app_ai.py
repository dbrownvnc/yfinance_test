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
    page_title="AI Hyper-Analyst V87", 
    page_icon="📈",
    initial_sidebar_state=st.session_state['sidebar_state']
)

# [로그 시스템]
if 'log_buffer' not in st.session_state:
    st.session_state['log_buffer'] = []

def add_log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state['log_buffer'].append(log_entry)
    if len(st.session_state['log_buffer']) > 500:
        st.session_state['log_buffer'].pop(0)

# [분석 항목]
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
# 2. 데이터 관리
# ---------------------------------------------------------
def load_data_to_state():
    if 'portfolio_df' not in st.session_state:
        if os.path.exists(CSV_FILE):
            try:
                df = pd.read_csv(CSV_FILE)
                st.session_state['portfolio_df'] = df.reset_index(drop=True) if not df.empty else pd.DataFrame(columns=['ticker', 'name'])
            except:
                st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
        else:
            st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])

def save_state_to_csv():
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df'].reset_index(drop=True)
        st.session_state['portfolio_df'] = df 
        try:
            df.to_csv(CSV_FILE, index=False)
        except: pass

def add_ticker_logic():
    raw_input = st.session_state.get('new_ticker_input', '')
    if raw_input:
        tickers = [t.strip().upper() for t in raw_input.split(',')]
        df = st.session_state['portfolio_df']
        new_rows = []
        for ticker in tickers:
            if ticker and ticker not in df['ticker'].values:
                try: 
                    t_info = yf.Ticker(ticker).info
                    name = t_info.get('shortName') or t_info.get('longName') or ticker
                except: 
                    name = ticker
                new_rows.append({'ticker': ticker, 'name': name})
        
        if new_rows:
            st.session_state['portfolio_df'] = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_state_to_csv()
    st.session_state['new_ticker_input'] = ""

load_data_to_state()

if 'del_ticker' in st.query_params:
    del_ticker = st.query_params['del_ticker']
    if 'portfolio_df' in st.session_state:
        st.session_state['portfolio_df'] = st.session_state['portfolio_df'][st.session_state['portfolio_df']['ticker'] != del_ticker]
        save_state_to_csv()
        if f"chk_{del_ticker}" in st.session_state: del st.session_state[f"chk_{del_ticker}"]
    st.query_params.clear()
    st.rerun()

# ---------------------------------------------------------
# 3. 유틸리티 및 데이터 수집
# ---------------------------------------------------------
def get_robust_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session

def run_with_timeout(func, args=(), timeout=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        try: return executor.submit(func, *args).result(timeout=timeout)
        except: return None

def _fetch_history(ticker, period): return yf.Ticker(ticker).history(period=period)
def _fetch_info(ticker): return yf.Ticker(ticker).info

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
            desc = ""
            if item.find('description') is not None: desc = clean_html_text(item.find('description').text)
            try: date_str = parser.parse(pubDate).strftime("%m-%d %H:%M")
            except: date_str = "최신"
            items.append({'title': title, 'link': link, 'date_str': date_str, 'summary': desc})
        return items
    except: return []

def get_realtime_news(ticker, name):
    """
    [강력 보정] 티커별 정식 기업명 강제 및 검색어 최적화
    """
    add_log(f"📰 [NEWS] 뉴스 검색 시작: {ticker}")
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    # 1. 혼동 티커 강제 매핑 (사용자 요청 반영: MS -> Morgan Stanley)
    explicit_names = {
        'MS': 'Morgan Stanley',
        'T': 'AT&T',
        'O': 'Realty Income',
        'C': 'Citigroup',
        'F': 'Ford Motor',
        'V': 'Visa',
        'M': 'Macy\'s',
        'K': 'Kellanova',
        'GM': 'General Motors'
    }
    
    # 2. 제외 키워드 설정 (MS 검색 시 Microsoft 제외)
    exclude_keywords = {
        'MS': ['microsoft', 'windows', 'azure', 'xbox', 'office 365', 'copilot', 'surface'],
        'T': [], 'O': [], 'C': [], 'F': [], 'V': []
    }
    
    clean_ticker = ticker.split('.')[0].upper()
    
    # [핵심] 검색어 결정 로직
    if clean_ticker in explicit_names:
        search_name = explicit_names[clean_ticker]
        add_log(f"   🚨 혼동 티커 감지! 검색어를 '{search_name}'로 강제 고정합니다.")
    else:
        # 이름이 너무 짧거나 티커와 같으면 가능한 긴 이름 사용
        if len(name) <= 3 or name.upper() == ticker.upper():
             # yfinance 정보가 부실하면 그냥 티커+Stock 보단 안전하게 처리 필요하지만
             # 일단 explicit_names에 없는 건 name을 신뢰하되, (주) 등 제거
             search_name = name
        else:
             search_name = name

    # 검색어 전처리 (Inc, Corp 제거)
    search_name_clean = re.sub(r' Inc\.?| Corp\.?| Ltd\.?| Co\.?| PLC', '', search_name, flags=re.IGNORECASE).strip()

    # 뉴스 검증 내부 함수
    def validate_news(n):
        title = n['title'].lower()
        summary = n.get('summary', '').lower()
        full_text = f"{title} {summary}"
        
        # 제외 키워드 체크
        if clean_ticker in exclude_keywords:
            for bad in exclude_keywords[clean_ticker]:
                if bad in full_text: return False
                
        # 관련성 체크 (제목에 이름이나 티커가 포함되어야 함)
        # 1. 기업명 포함 여부
        if search_name_clean.lower() in full_text: return True
        # 2. 티커 단독 포함 여부 (예: $MS)
        if re.search(rf'\b{clean_ticker}\b', title): return True
        
        return False

    # 1. Yahoo Finance (미국장 우선)
    if not is_kr:
        try:
            rss = fetch_rss_realtime(f"https://finance.yahoo.com/rss/headline?s={ticker}", limit=10)
            valid = [item for item in rss if validate_news(item)]
            if valid:
                for v in valid: v['source'] = 'Yahoo Finance'
                return valid[:7]
        except: pass

    # 2. Google News (검색어 정밀 조작)
    try:
        if is_kr:
            q = f'"{search_name_clean}"'
        else:
            # [핵심] intitle: 명령어 사용 + 제외어 추가
            q = f'intitle:"{search_name_clean}"'
            if clean_ticker in exclude_keywords:
                for bad in exclude_keywords[clean_ticker]:
                    q += f' -{bad}'
        
        add_log(f"   🔍 Google 쿼리: {q}")
        rss = fetch_rss_realtime(f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=ko&gl=KR&ceid=KR:ko", limit=10)
        
        valid = []
        for item in rss:
            if validate_news(item):
                item['source'] = 'Google News'
                valid.append(item)
        
        if valid: return valid[:7]
    except: pass
    
    return news_items

def get_company_info(ticker):
    info = run_with_timeout(_fetch_info, args=(ticker,), timeout=8)
    if not info: return {'name': ticker, 'long_name': ticker, 'sector': 'N/A', 'industry': 'N/A', 'market_cap': 'N/A'}
    
    mcap = info.get('marketCap')
    if mcap:
        if mcap >= 1e12: mcap_str = f"${mcap/1e12:.2f}T"
        elif mcap >= 1e9: mcap_str = f"${mcap/1e9:.2f}B"
        else: mcap_str = f"${mcap:,.0f}"
    else: mcap_str = "N/A"

    return {
        'name': info.get('shortName', ticker),
        'long_name': info.get('longName', ticker), # 정식 명칭 우선
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': mcap_str,
        'employees': info.get('fullTimeEmployees', 'N/A'),
        'country': info.get('country', 'N/A')
    }

def get_financial_metrics(ticker):
    info = run_with_timeout(_fetch_info, args=(ticker,), timeout=5)
    if not info: return {}
    def fmt(k, is_pct=False):
        v = info.get(k)
        if isinstance(v, (int, float)): return f"{v*100:.2f}%" if is_pct else f"{v:,.2f}"
        return "N/A"
    return {
        "FCF": fmt('freeCashflow'), "유동비율": fmt('currentRatio'), "부채비율": fmt('debtToEquity'),
        "ROE": fmt('returnOnEquity', True), "매출": fmt('totalRevenue'), "순이익": fmt('netIncome'),
        "PER(TTM)": fmt('trailingPE'), "PBR": fmt('priceToBook'), "배당수익률": fmt('dividendYield', True),
        "52주 최고": fmt('fiftyTwoWeekHigh'), "52주 최저": fmt('fiftyTwoWeekLow')
    }

def sanitize_text(text):
    return re.sub(r'\n\s*\n+', '\n\n', text.replace('$', '\$')).strip()

def collapse_sidebar():
    st.components.v1.html("""<script>var c=window.parent.document.querySelector('[data-testid="stSidebarExpandedControl"]');if(c)c.click();</script>""", height=0)

def start_analysis_process(targets, mode, is_prompt_only):
    st.session_state.update({'is_analyzing': True, 'targets_to_run': targets, 'current_mode': mode, 
                             'prompt_mode': is_prompt_only, 'analysis_results': {}, 'proc_index': 0, 'proc_stage': 1})

def generate_with_fallback(prompt, api_key, start_model):
    genai.configure(api_key=api_key)
    models = [start_model] + [m for m in ["gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-pro"] if m != start_model]
    for m in models:
        try:
            return genai.GenerativeModel(m).generate_content(prompt).text, m
        except: continue
    raise Exception("All models failed")

def handle_search_click(mode, is_prompt):
    inp = st.session_state.get("s_input", "")
    if inp: start_analysis_process([t.strip() for t in inp.split(',')], mode, is_prompt)
    else: st.warning("티커 입력 필요")

def step_fetch_data(ticker, mode):
    add_log(f"📦 데이터 수집: {ticker}")
    clean_ticker = re.sub(r'[^0-9a-zA-Z.]', '', ticker).upper()
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    tv_symbol = f"KRX:{re.sub(r'[^0-9]', '', ticker)}" if is_kr else ticker

    try:
        # 1. 기업 정보 조회
        c_info = get_company_info(ticker)
        # [핵심] 공식 기업명 설정 (Morgan Stanley 등)
        stock_name = c_info['long_name']
        
        # 포트폴리오 이름이 더 정확하다면 사용 (단, 짧은 이름은 제외)
        if 'portfolio_df' in st.session_state:
            p_df = st.session_state['portfolio_df']
            row = p_df[p_df['ticker'] == ticker]
            if not row.empty:
                saved = row.iloc[0]['name']
                if len(saved) > len(stock_name): stock_name = saved

        # 2. 주가 데이터 (180일로 확대 요청 반영)
        period = st.session_state.get('selected_period_str', '1y')
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=10)
        if df is None: df = pd.DataFrame()
        
        data_summary = "No Data"
        if not df.empty:
            curr = df['Close'].iloc[-1]
            # [수정] 60일 -> 180일 데이터 제공
            display_df = df.tail(180) 
            data_summary = f"[가격] 현재: {curr:.2f}, 52주 고/저: {c_info.get('fiftyTwoWeekHigh','N/A')}/{c_info.get('fiftyTwoWeekLow','N/A')}\n[최근 180일 주가 추세]\n{display_df.to_string()}"

        # 3. 뉴스 및 재무
        fin_str = "N/A"; news_text = "N/A"
        if mode not in ["10K", "10Q", "8K"]:
            fm = get_financial_metrics(ticker)
            if fm: fin_str = str(fm)
            
            if st.session_state.get('use_news', True):
                # get_realtime_news 내부에서 MS -> Morgan Stanley 강제 변환 수행
                news = get_realtime_news(ticker, stock_name)
                if news:
                    news_lines = []
                    for n in news:
                        summ = n['summary'][:100] + "..." if n['summary'] else ""
                        news_lines.append(f"- [{n['source']}] {n['title']} ({n['date_str']}) {summ}")
                    news_text = "\n".join(news_lines)
                else: news_text = "관련 뉴스 없음 (필터링됨)"

        # 4. 프롬프트 생성 (요청하신 포맷 적용)
        focus_list = [opt for opt in opt_targets if st.session_state.get(f"focus_{opt}", True)]
        focus_str = ", ".join(focus_list)
        viewpoint = st.session_state.get('selected_viewpoint', '중기')
        depth = st.session_state.get('analysis_depth', '2.표준')
        
        # 시나리오 모드 확인
        scenario_instruction = ""
        if "5." in depth or "시나리오" in depth:
            scenario_instruction = "가장 낙관적인/비관적인 시나리오와 구체적인 미래 주가 예측(Target Price Range)을 포함하여 심층적으로 분석하십시오."

        # [요청하신 프롬프트 양식 적용]
        prompt = f"""
[역할] 월스트리트 수석 애널리스트
[대상] {ticker} (공식 기업명: {stock_name})
[모드] {mode}
[중점 분석] {focus_str}
[투자 관점] {viewpoint}
[분석 레벨] {depth}

**주의: '{ticker}'는 '{stock_name}'입니다. 다른 기업과 혼동하지 마십시오.**

{scenario_instruction}

[데이터 요약]
{data_summary}

[재무 지표]
{fin_str}

[관련 뉴스]
{news_text}

[지시사항]
위 데이터를 바탕으로 전문적이고 종합적인 투자 보고서를 작성하십시오.
**뉴스 분석 시, 제목뿐만 아니라 제공된 '내용요약'을 참고하여 구체적인 원인과 영향을 파악하십시오.**
보고서는 가독성 있게 마크다운 형식으로 작성하고, 불필요한 서론 없이 본론부터 명확히 서술하십시오.

결론 부분에는 반드시 [매수 / 매도 / 관망] 중 하나의 명확한 투자 의견을 제시하십시오.
"""
        st.session_state['temp_data'] = {
            'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': [], 'company_info': c_info
        }
        return True
    except Exception as e:
        add_log(f"Error: {e}")
        return False

# ---------------------------------------------------------
# 5. UI 구성
# ---------------------------------------------------------
st.sidebar.subheader("🎯 분석 옵션")
viewpoint_mapping = {"단기": "3mo", "스윙": "6mo", "중기": "2y", "장기": "5y"}
sel_vp = st.sidebar.select_slider("", list(viewpoint_mapping.keys()), value="중기", label_visibility="collapsed")
st.session_state['selected_period_str'] = viewpoint_mapping[sel_vp]
st.session_state['selected_viewpoint'] = sel_vp

depth = st.sidebar.select_slider("", ["1.요약", "2.표준", "3.심층", "4.전문가", "5.시나리오"], value="5.시나리오", label_visibility="collapsed")
st.session_state['analysis_depth'] = depth
st.session_state['use_news'] = st.sidebar.toggle("뉴스 반영", value=True)

with st.sidebar.expander("☑️ 분석 항목", expanded=False):
    if st.checkbox("전체 선택", key="focus_all"):
        for opt in opt_targets: st.session_state[f"focus_{opt}"] = True
    for opt in opt_targets: st.checkbox(opt, key=f"focus_{opt}")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.sidebar.error("API Key 필요")

tab1, tab2 = st.sidebar.tabs(["⚡ 검색", "⭐ 포트폴리오"])
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    inp = st.text_input("티커", key="s_input")
    c1, c2 = st.columns(2)
    chk = c1.checkbox("프롬프트만", value=True)
    if c2.button("분석 시작", type="primary"): 
        if inp: start_analysis_process([t.strip() for t in inp.split(',')], "MAIN", chk)

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.7, 0.3])
    new_t = c1.text_input("추가", key="new_ticker_input", label_visibility="collapsed")
    c2.button("➕", on_click=add_ticker_logic)
    
    fav_df = st.session_state.get('portfolio_df', pd.DataFrame())
    if not fav_df.empty:
        selected = []
        for _, r in fav_df.iterrows():
            if st.checkbox(f"{r['ticker']} ({r['name']})", key=f"chk_{r['ticker']}"): selected.append(r['ticker'])
        
        c1, c2 = st.columns(2)
        chk_p = c1.checkbox("프롬프트만", key="chk_p", value=True)
        if c2.button("종합 분석"): start_analysis_process(selected, "MAIN", chk_p)

st.sidebar.markdown('---')
sel_model = st.sidebar.selectbox("모델", ["gemini-1.5-pro", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash"], index=0)

# ---------------------------------------------------------
# 6. 실행
# ---------------------------------------------------------
st.title("📈 AI Hyper-Analyst V87")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    idx = st.session_state['proc_index']
    stage = st.session_state['proc_stage']
    
    if idx >= len(targets):
        st.success("완료!")
        st.session_state['is_analyzing'] = False
        st.rerun()
        
    ticker = targets[idx]
    st.progress((idx * 2 + stage) / (len(targets) * 2), text=f"분석 중: {ticker}")

    if stage == 1:
        if idx == 0: collapse_sidebar()
        with st.spinner("데이터 수집 중..."):
            if step_fetch_data(ticker, st.session_state['current_mode']):
                st.session_state['proc_stage'] = 2
            else:
                st.session_state['analysis_results'][ticker] = {'status': 'error', 'report': '실패'}
                st.session_state['proc_index'] += 1
            st.rerun()
            
    elif stage == 2:
        temp = st.session_state['temp_data']
        if st.session_state['prompt_mode']:
            res = {'status': 'manual', 'prompt': temp['prompt'], 'name': temp['name'], 'tv_symbol': temp['tv_symbol'], 'df': temp['df'], 'company_info': temp['company_info']}
        else:
            with st.spinner("AI 분석 중..."):
                try:
                    txt, m = generate_with_fallback(temp['prompt'], api_key, sel_model)
                    res = {'status': 'success', 'report': txt, 'model': m, 'name': temp['name'], 'tv_symbol': temp['tv_symbol'], 'df': temp['df'], 'company_info': temp['company_info']}
                except Exception as e:
                    res = {'status': 'error', 'report': str(e)}
        
        st.session_state['analysis_results'][ticker] = res
        st.session_state['proc_index'] += 1
        st.session_state['proc_stage'] = 1
        st.rerun()

# ---------------------------------------------------------
# 7. 결과
# ---------------------------------------------------------
if not st.session_state['is_analyzing'] and st.session_state['analysis_results']:
    for t, d in st.session_state['analysis_results'].items():
        with st.expander(f"📊 {d.get('name', t)} ({t})", expanded=True):
            if not d.get('df', pd.DataFrame()).empty:
                # 차트 출력
                html_code = f"""<div id="chart_{t}" style="height:350px"></div>
                <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                <script type="text/javascript">
                new TradingView.widget({{
                    "autosize": true, "symbol": "{d['tv_symbol']}", "interval": "D", "timezone": "Asia/Seoul",
                    "theme": "light", "style": "1", "locale": "ko", "toolbar_bg": "#f1f3f6", "enable_publishing": false,
                    "container_id": "chart_{t}"
                }});
                </script>"""
                st.components.v1.html(html_code, height=360)
            
            if d['status'] == 'manual':
                st.code(d['prompt'])
                st.link_button("Gemini 열기", "https://gemini.google.com")
            elif d['status'] == 'success':
                st.markdown(d['report'])
            else:
                st.error(d.get('report'))
