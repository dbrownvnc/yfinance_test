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
    page_title="AI Hyper-Analyst V86", 
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

# [변수 정의] 최상단 배치
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
# 2. 데이터 관리 함수 (Session State Master 방식)
# ---------------------------------------------------------
def load_data_to_state():
    """CSV 파일을 읽어 Session State에 로드 (앱 실행 시 1회 수행)"""
    if 'portfolio_df' not in st.session_state:
        add_log("📥 [INIT] 포트폴리오 데이터 로드 시도...")
        if os.path.exists(CSV_FILE):
            try:
                df = pd.read_csv(CSV_FILE)
                if df.empty:
                    st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
                    add_log("ℹ️ [INIT] 파일은 존재하나 데이터가 비어있음.")
                else:
                    st.session_state['portfolio_df'] = df.reset_index(drop=True)
                    add_log(f"✅ [INIT] 데이터 로드 완료: {len(df)}개 항목 로드됨.")
            except Exception as e:
                st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
                add_log(f"❌ [INIT] 데이터 로드 중 에러 발생: {str(e)}")
        else:
            st.session_state['portfolio_df'] = pd.DataFrame(columns=['ticker', 'name'])
            add_log("ℹ️ [INIT] 기존 파일 없음. 새 포트폴리오 데이터프레임 생성.")

def save_state_to_csv():
    """현재 Session State의 데이터를 CSV로 저장하고 인덱스 재정렬"""
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        df = df.reset_index(drop=True)
        st.session_state['portfolio_df'] = df 
        
        try:
            with open(CSV_FILE, 'w', encoding='utf-8', newline='') as f:
                df.to_csv(f, index=False)
                f.flush()
                os.fsync(f.fileno()) 
            add_log(f"💾 [SAVE] 파일 저장 완료. 총 {len(df)}개 항목 동기화됨.")
        except Exception as e:
            add_log(f"❌ [SAVE] 파일 저장 실패: {str(e)}")

def add_ticker_logic():
    """티커 추가 로직 (Callback)"""
    raw_input = st.session_state.get('new_ticker_input', '')
    if raw_input:
        add_log(f"➕ [ADD] 티커 추가 요청 감지: '{raw_input}'")
        tickers = [t.strip().upper() for t in raw_input.split(',')]
        df = st.session_state['portfolio_df']
        existing_tickers = df['ticker'].values
        
        new_rows = []
        for ticker in tickers:
            if ticker and ticker not in existing_tickers:
                try: 
                    add_log(f"🔍 [ADD] {ticker} 정보 조회 중 (yfinance)...")
                    t_info = yf.Ticker(ticker).info
                    name = t_info.get('shortName') or t_info.get('longName') or ticker
                    add_log(f"   -> 이름 식별 성공: {name}")
                except Exception as e: 
                    name = ticker
                    add_log(f"   ⚠️ [ADD] {ticker} 정보 조회 실패, 티커명 사용. Error: {e}")
                
                new_rows.append({'ticker': ticker, 'name': name})
                add_log(f"   -> 추가 목록에 등록: {ticker}")
            else:
                add_log(f"   -> 중복 스킵: {ticker}")
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            st.session_state['portfolio_df'] = df
            save_state_to_csv()
            add_log("✅ [ADD] 신규 티커 저장 완료 및 UI 갱신.")
            
    st.session_state['new_ticker_input'] = ""

# 앱 시작 시 데이터 로드
load_data_to_state()

# ---------------------------------------------------------
# [최우선 처리] 삭제 요청 핸들링 (새로고침 로직)
# ---------------------------------------------------------
if 'del_ticker' in st.query_params:
    del_ticker = st.query_params['del_ticker']
    add_log(f"🗑️ [DELETE] 삭제 요청 수신: {del_ticker}")
    
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        prev_len = len(df)
        df = df[df['ticker'] != del_ticker]
        new_len = len(df)
        st.session_state['portfolio_df'] = df
        add_log(f"   -> 메모리 삭제 완료 ({prev_len} -> {new_len})")
        
        save_state_to_csv()
        
        if f"chk_{del_ticker}" in st.session_state:
            del st.session_state[f"chk_{del_ticker}"]
            
    st.query_params.clear()
    add_log("🔄 [DELETE] 변경 사항 반영을 위해 Rerun 수행.")
    st.rerun()

# ---------------------------------------------------------
# 3. 기타 유틸리티 함수
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

def get_stock_name(ticker):
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        row = df[df['ticker'] == ticker]
        if not row.empty:
            return row.iloc[0]['name']
            
    try:
        info = run_with_timeout(_fetch_info, args=(ticker,), timeout=5)
        if info: return info.get('shortName') or info.get('longName') or ticker
        return ticker
    except: return ticker

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
    add_log(f"   🌐 [RSS] Fetching URL: {url}")
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
        add_log(f"   ✅ [RSS] Parsed {len(items)} items.")
        return items
    except Exception as e:
        add_log(f"   ❌ [RSS] Error: {e}")
        return []

def get_realtime_news(ticker, name):
    add_log(f"📰 [NEWS] 뉴스 검색 시작: {ticker} ({name})")
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    if not is_kr:
        try:
            add_log(f"   Trying Yahoo Finance RSS for {ticker}...")
            rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            yahoo_rss_items = fetch_rss_realtime(rss_url, limit=7)
            if yahoo_rss_items:
                add_log(f"   -> Yahoo RSS에서 {len(yahoo_rss_items)}건 발견")
                for item in yahoo_rss_items:
                    item['source'] = "Yahoo Finance"
                    news_items.append(item)
                return news_items
        except Exception as e:
            add_log(f"   ⚠️ Yahoo RSS Fail: {e}")

    if not is_kr and not news_items:
        try:
            add_log(f"   Trying yfinance library for {ticker}...")
            yf_obj = yf.Ticker(ticker)
            yf_news = yf_obj.news
            if yf_news:
                add_log(f"   -> yfinance에서 {len(yf_news)}건 발견")
                for item in yf_news:
                    title = item.get('title'); link = item.get('link')
                    summary = item.get('summary', '') 
                    if not title or not link: continue
                    pub_time = item.get('providerPublishTime', 0)
                    try: date_str = datetime.datetime.fromtimestamp(pub_time).strftime("%m-%d %H:%M")
                    except: date_str = "최신"
                    news_items.append({'title': title, 'link': link, 'date_str': date_str, 'source': "Yahoo Finance", 'summary': summary})
                if news_items: return news_items[:7]
        except Exception as e:
            add_log(f"   ⚠️ yfinance Fail: {e}")

    if is_kr: search_query = f'"{name}"'
    else: search_query = f'{ticker} stock'
    
    add_log(f"   Trying Google News RSS with query: {search_query}")
    q_encoded = urllib.parse.quote(search_query)
    url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ko&gl=KR&ceid=KR:ko"
    google_news = fetch_rss_realtime(url, limit=7)
    for n in google_news: n['source'] = "Google News"
    return google_news

def get_company_info(ticker):
    """기업 기본 정보 (이름, 섹터, 산업) 조회"""
    add_log(f"🏢 [INFO] 기업 정보 조회: {ticker}")
    info = run_with_timeout(_fetch_info, args=(ticker,), timeout=8)
    if not info: 
        add_log("   ❌ [INFO] 기업 정보 가져오기 실패")
        return {
            'name': ticker,
            'long_name': ticker,
            'sector': '정보 없음',
            'industry': '정보 없음',
            'country': '정보 없음',
            'website': '정보 없음',
            'market_cap': 'N/A',
            'employees': 'N/A'
        }
    try:
        def safe_get(key, default='정보 없음'):
            val = info.get(key)
            return val if val else default
        
        market_cap = info.get('marketCap')
        if market_cap:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"
        
        employees = info.get('fullTimeEmployees')
        employees_str = f"{employees:,}명" if employees else "N/A"
        
        company_info = {
            'name': safe_get('shortName', ticker),
            'long_name': safe_get('longName', ticker),
            'sector': safe_get('sector'),
            'industry': safe_get('industry'),
            'country': safe_get('country'),
            'website': safe_get('website'),
            'market_cap': market_cap_str,
            'employees': employees_str
        }
        add_log(f"   ✅ [INFO] 기업 정보 확보: {company_info['name']} | {company_info['sector']} | {company_info['industry']}")
        return company_info
    except Exception as e:
        add_log(f"   ⚠️ [INFO] 데이터 파싱 에러: {e}")
        return {
            'name': ticker,
            'long_name': ticker,
            'sector': '정보 없음',
            'industry': '정보 없음',
            'country': '정보 없음',
            'website': '정보 없음',
            'market_cap': 'N/A',
            'employees': 'N/A'
        }

def get_financial_metrics(ticker):
    add_log(f"📊 [FIN] 재무 지표 조회: {ticker}")
    info = run_with_timeout(_fetch_info, args=(ticker,), timeout=5)
    if not info: 
        add_log("   ❌ [FIN] 정보 가져오기 실패 (Timeout/Empty)")
        return {}
    try:
        def get_fmt(key): 
            val = info.get(key)
            return f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A"
        
        def get_pct(key):
            val = info.get(key)
            if isinstance(val, (int, float)):
                return f"{val*100:.2f}%"
            return "N/A"
        
        metrics = {
            "Free Cash Flow": get_fmt('freeCashflow'), 
            "Current Ratio": get_fmt('currentRatio'),
            "Quick Ratio": get_fmt('quickRatio'), 
            "Debt to Equity": get_fmt('debtToEquity'),
            "Return on Equity (ROE)": get_pct('returnOnEquity'), 
            "Total Revenue": get_fmt('totalRevenue'),
            "Net Income": get_fmt('netIncome'),
            "Revenue Growth": get_pct('revenueGrowth'),
            "EPS (TTM)": get_fmt('trailingEps'),
            "Forward EPS": get_fmt('forwardEps'),
            "Profit Margin": get_pct('profitMargins'),
            "Operating Margin": get_pct('operatingMargins'),
            "Gross Margin": get_pct('grossMargins'),
            "Dividend Yield": get_pct('dividendYield'),
            "Payout Ratio": get_pct('payoutRatio'),
            "Beta": get_fmt('beta'),
            "PE Ratio (TTM)": get_fmt('trailingPE'),
            "Forward PE": get_fmt('forwardPE'),
            "PEG Ratio": get_fmt('pegRatio'),
            "Price to Book": get_fmt('priceToBook'),
            "Price to Sales": get_fmt('priceToSalesTrailing12Months'),
            "52주 최고가": get_fmt('fiftyTwoWeekHigh'),
            "52주 최저가": get_fmt('fiftyTwoWeekLow'),
            "50일 이평선": get_fmt('fiftyDayAverage'),
            "200일 이평선": get_fmt('twoHundredDayAverage'),
        }
        add_log(f"   ✅ [FIN] 재무 지표 확보 완료")
        return metrics
    except Exception as e: 
        add_log(f"   ⚠️ [FIN] 데이터 파싱 에러: {e}")
        return {}

def sanitize_text(text):
    text = text.replace('$', '\$')
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    return text

def collapse_sidebar():
    js = """<script>var closeBtn = window.parent.document.querySelector('[data-testid="stSidebarExpandedControl"]');if (closeBtn) {closeBtn.click();}</script>"""
    st.components.v1.html(js, height=0, width=0)

def start_analysis_process(targets, mode, is_prompt_only):
    add_log(f"▶️ [PROCESS] 분석 프로세스 트리거: Targets={len(targets)}개, Mode={mode}")
    st.session_state['is_analyzing'] = True
    st.session_state['targets_to_run'] = targets
    st.session_state['current_mode'] = mode
    st.session_state['prompt_mode'] = is_prompt_only
    st.session_state['analysis_results'] = {} 
    st.session_state['proc_index'] = 0
    st.session_state['proc_stage'] = 1 

def generate_with_fallback(prompt, api_key, start_model):
    genai.configure(api_key=api_key)
    fallback_chain = [start_model]
    backups = ["gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.0-pro", "gemini-flash-latest"]
    for b in backups:
        if b != start_model: fallback_chain.append(b)
    
    last_error = None
    add_log(f"🧠 [AI] 모델 체인 시작: {fallback_chain}")
    
    for model_name in fallback_chain:
        try:
            start_time = time.time()
            add_log(f"   Attempting: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            duration = time.time() - start_time
            
            add_log(f"   ✅ [AI] 성공! ({model_name}, {duration:.2f}s)")
            return response.text, model_name 
        except Exception as e:
            add_log(f"   ⚠️ [AI] 실패 ({model_name}): {str(e)}")
            last_error = e; time.sleep(0.5); continue
            
    add_log("❌ [AI] 모든 모델 시도 실패.")
    raise Exception(f"All models failed. Last Error: {last_error}")

def handle_search_click(mode, is_prompt):
    raw_input = st.session_state.get("s_input", "")
    if raw_input:
        targets = [t.strip() for t in raw_input.split(',') if t.strip()]
        add_log(f"🔎 [SEARCH] 단일 검색 요청: {targets}")
        start_analysis_process(targets, mode, is_prompt)
    else: st.warning("티커를 입력해주세요.")

def step_fetch_data(ticker, mode):
    add_log(f"==========================================")
    add_log(f"📦 [STEP 1] 데이터 수집 시작: {ticker} ({mode})")
    
    stock_name = ticker 
    clean_code = re.sub(r'[^0-9]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker

    try:
        # =====================================================
        # [핵심 수정] 기업 정보 (이름, 섹터, 산업) 먼저 조회
        # =====================================================
        company_info = get_company_info(ticker)
        stock_name = company_info['long_name'] if company_info['long_name'] != ticker else company_info['name']
        
        # 포트폴리오에서 이름 확인 (우선순위)
        if 'portfolio_df' in st.session_state:
            p_df = st.session_state['portfolio_df']
            row = p_df[p_df['ticker'] == ticker]
            if not row.empty:
                portfolio_name = row.iloc[0]['name']
                if portfolio_name and portfolio_name != ticker:
                    stock_name = portfolio_name
                    add_log(f"   - 이름(포트폴리오 우선): {stock_name}")
            
        period = st.session_state.get('selected_period_str', '1y')
        add_log(f"   - 주가 데이터 요청 (기간: {period})")
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=10)
        
        if df is None: 
            df = pd.DataFrame()
            add_log("   ⚠️ 주가 데이터 타임아웃/실패")
        else:
            add_log(f"   ✅ 주가 데이터 수신: {len(df)} rows")

        data_summary = "No Data"
        if not df.empty:
            curr = df['Close'].iloc[-1]; high_val = df['High'].max(); low_val = df['Low'].min()
            stats_str = f"High: {high_val:.2f}, Low: {low_val:.2f}, Current: {curr:.2f}"
            display_df = df.tail(60); recent_days = df.tail(5)
            data_summary = f"[Stats] {stats_str}\n[Trend]\n{display_df.to_string()}\n[Recent]\n{recent_days.to_string()}"
        else: curr = 0

        fin_str = "N/A"; news_text = "N/A"
        
        if mode not in ["10K", "10Q", "8K"]:
            try: 
                fm = get_financial_metrics(ticker)
                fin_str = str(fm) if fm else "N/A"
            except: pass
            
            if st.session_state.get('use_news', True):
                try:
                    news = get_realtime_news(ticker, stock_name)
                    if news: 
                        formatted_news = []
                        for n in news:
                            title = n['title']
                            summary = n.get('summary', '')
                            if is_similar(title, summary): summary = ""
                            elif len(summary) > 200: summary = summary[:200] + "..."
                            item_str = f"- [{n.get('source', 'News')}] {title} ({n['date_str']})"
                            if summary: item_str += f"\n  > 내용요약: {summary}"
                            formatted_news.append(item_str)
                        news_text = "\n".join(formatted_news)
                        add_log(f"   ✅ 뉴스 텍스트 생성 완료 ({len(news)}건)")
                    else: news_text = "관련된 최신 뉴스가 없습니다."
                except Exception as e: 
                    news_text = f"뉴스 가져오기 실패: {str(e)}"
                    add_log(f"   ❌ 뉴스 처리 중 치명적 오류: {e}")

        selected_focus_list = []
        for opt in opt_targets:
            if st.session_state.get(f"focus_{opt}", True): selected_focus_list.append(opt)
        focus = ", ".join(selected_focus_list)
        viewpoint = st.session_state.get('selected_viewpoint', 'General')
        analysis_depth = st.session_state.get('analysis_depth', "2. 표준 브리핑 (Standard)")
        
        # =====================================================
        # [핵심 수정] 시나리오 모드 강화
        # =====================================================
        level_instruction = ""
        scenario_section = ""
        
        if "5." in analysis_depth:
            level_instruction = """
⚠️ **[시나리오 분석 모드 활성화]**
이 분석은 '시나리오 모드'입니다. 아래 시나리오 분석 섹션을 **반드시 상세하게 작성**하십시오.
절대로 생략하지 마십시오.
"""
            scenario_section = """
---
## 🎭 시나리오 분석 (SCENARIO ANALYSIS) - 필수 작성

⚠️ **이 섹션은 시나리오 모드의 핵심입니다. 반드시 3가지 시나리오를 모두 상세히 작성하십시오.**

### 📈 시나리오 1: 낙관적 시나리오 (Bull Case)
**발생 확률**: [구체적인 %를 제시하시오, 예: 25%]

**시나리오 전제 조건** (이 시나리오가 실현되려면):
1. [첫 번째 필요 조건 - 구체적으로]
2. [두 번째 필요 조건 - 구체적으로]
3. [세 번째 필요 조건 - 구체적으로]

**예상 주가 흐름**:
- 목표 주가: [구체적인 금액]
- 현재가 대비 상승률: [%]
- 예상 도달 시점: [기간]

**이 시나리오를 지지하는 근거**:
1. [근거 1 - 데이터/사실 기반]
2. [근거 2 - 데이터/사실 기반]
3. [근거 3 - 데이터/사실 기반]

---

### ➡️ 시나리오 2: 기본 시나리오 (Base Case)
**발생 확률**: [구체적인 %를 제시하시오, 예: 50%]

**시나리오 전제 조건** (현재 상황이 유지된다면):
1. [첫 번째 전제 - 구체적으로]
2. [두 번째 전제 - 구체적으로]
3. [세 번째 전제 - 구체적으로]

**예상 주가 흐름**:
- 목표 주가 범위: [구체적인 금액 범위]
- 현재가 대비 등락률: [%]
- 예상 횡보/변동 기간: [기간]

**이 시나리오가 가장 가능성 높은 이유**:
1. [이유 1 - 논리적 설명]
2. [이유 2 - 논리적 설명]
3. [이유 3 - 논리적 설명]

---

### 📉 시나리오 3: 비관적 시나리오 (Bear Case)
**발생 확률**: [구체적인 %를 제시하시오, 예: 25%]

**시나리오 전제 조건** (이 시나리오가 실현되려면):
1. [첫 번째 위험 요소 - 구체적으로]
2. [두 번째 위험 요소 - 구체적으로]
3. [세 번째 위험 요소 - 구체적으로]

**예상 주가 흐름**:
- 하방 목표가: [구체적인 금액]
- 현재가 대비 하락률: [%]
- 손절 권장 가격: [구체적인 금액]

**이 시나리오의 위험 신호 (모니터링 포인트)**:
1. [위험 신호 1 - 구체적인 지표/이벤트]
2. [위험 신호 2 - 구체적인 지표/이벤트]
3. [위험 신호 3 - 구체적인 지표/이벤트]

---

### 🎯 시나리오별 대응 전략 요약

**낙관적 시나리오 (Bull)**: 확률 [X]%, 목표가 [금액], 핵심 트리거는 [트리거], 권장 액션은 [매수/홀드/추가매수]

**기본 시나리오 (Base)**: 확률 [X]%, 목표가 [금액], 핵심 트리거는 [트리거], 권장 액션은 [홀드/부분매도]

**비관적 시나리오 (Bear)**: 확률 [X]%, 목표가 [금액], 핵심 트리거는 [트리거], 권장 액션은 [손절/비중축소]

**⚠️ 확률 합계 검증**: 세 시나리오의 확률 합이 100%가 되도록 조정하십시오.
"""
        
        # =====================================================
        # [핵심 수정] 각 분석 항목별 상세 지시사항 생성 함수 (강화)
        # =====================================================
        def build_detailed_analysis_instructions(focus_list):
            """선택된 분석 항목에 대한 상세 지시사항 생성"""
            instructions = []
            
            if "현금건전성 지표 (FCF, 유동비율, 부채비율)" in focus_list:
                instructions.append("""
---
### 📊 현금건전성 지표 분석 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오. 생략 시 분석 품질이 크게 저하됩니다.**

- **Free Cash Flow (FCF)**: 
  - 현재 값: [구체적 금액]
  - 전년 대비 증감률: [%]
  - FCF 마진율: [%]
  - 해석: [양호/주의/위험]

- **유동비율 (Current Ratio)**: 
  - 현재 값: [숫자]
  - 업종 평균 대비: [상회/하회/유사]
  - 해석: [단기 지급능력 평가]

- **부채비율 (Debt to Equity)**: 
  - 현재 값: [%]
  - 추세: [증가/감소/안정]
  - 해석: [재무 건전성 평가]

- **Quick Ratio**: 
  - 현재 값: [숫자]
  - 해석: [즉시 지급능력 평가]

**💡 현금건전성 종합 의견**: [양호/보통/주의 필요 중 하나 선택 및 근거 설명]""")
            
            if "핵심 재무제표 분석 (손익, 대차대조, 현금흐름)" in focus_list:
                instructions.append("""
---
### 📈 핵심 재무제표 분석 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오.**

**1. 손익계산서 분석**
- **매출액**: 최근 실적 [금액], YoY 성장률 [%], 평가 [양호/보통/부진]
- **영업이익**: 최근 실적 [금액], YoY 성장률 [%], 평가 [양호/보통/부진]
- **순이익**: 최근 실적 [금액], YoY 성장률 [%], 평가 [양호/보통/부진]
- **영업이익률 변화**: 전년 [%] → 금년 [%] ([개선/악화])
- **순이익률 변화**: 전년 [%] → 금년 [%] ([개선/악화])
- **비용 구조 특이사항**: [있다면 기술]

**2. 대차대조표 분석**
- **자산 총계**: [금액] (전년 대비 [%] 변화)
- **부채 총계**: [금액] (부채 비율 [%])
- **자기자본**: [금액] (ROE [%])

**3. 현금흐름표 분석**
- **영업활동 현금흐름**: [금액], 전년 대비 [증가/감소], 해석: [플러스면 양호, 마이너스면 주의]
- **투자활동 현금흐름**: [금액], 전년 대비 [증가/감소], 해석: [CAPEX 투자 현황]
- **재무활동 현금흐름**: [금액], 전년 대비 [증가/감소], 해석: [배당/자사주/차입 상황]

**💡 재무제표 종합 평가**: [건전/보통/취약 중 선택 및 근거]""")
            
            if "투자기관 목표주가 및 컨센서스" in focus_list:
                instructions.append("""
---
### 🎯 투자기관 목표주가 및 컨센서스 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오. 데이터가 부족하면 '정보 제한적'이라고 명시하되, 가용한 정보는 모두 제공하십시오.**

**목표주가 분석**
- **최저 목표가**: [금액] (현재가 대비 [%])
- **평균 목표가**: [금액] (현재가 대비 [%])
- **최고 목표가**: [금액] (현재가 대비 [%])

**투자의견 분포**
- 강력 매수: [개] ([%])
- 매수: [개] ([%])
- 보유: [개] ([%])
- 매도: [개] ([%])

**최근 컨센서스 변화**
- 추세: [상향 조정 / 하향 조정 / 유지]
- 주요 변경 사유: [있다면 기술]

**주요 증권사 최근 의견** (2-3개)
1. [증권사명]: [의견] / 목표가 [금액] / [핵심 논거]
2. [증권사명]: [의견] / 목표가 [금액] / [핵심 논거]

**💡 컨센서스 종합 평가**: [긍정적/중립/부정적 중 선택 및 근거]""")
            
            if "호재/악재 뉴스 판단" in focus_list:
                instructions.append("""
---
### 📰 호재/악재 뉴스 판단 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오.**

**[호재 뉴스 🟢]** 
1. **[뉴스 제목 1]**: 영향도 [상/중/하], 주가 영향 분석 - [구체적 영향 설명]
2. **[뉴스 제목 2]**: 영향도 [상/중/하], 주가 영향 분석 - [구체적 영향 설명]

**[악재 뉴스 🔴]**
1. **[뉴스 제목 1]**: 리스크 수준 [높음/중간/낮음], 대응 전략 - [대응 방안]
2. **[뉴스 제목 2]**: 리스크 수준 [높음/중간/낮음], 대응 전략 - [대응 방안]

**[중립 뉴스 ⚪]** (있다면)
- [뉴스 제목 및 해석]

**💡 뉴스 환경 종합 판단**: 
- 현재 뉴스 톤: [긍정적 / 부정적 / 혼재]
- 투자 시사점: [매수 기회 / 리스크 관리 필요 / 관망]""")
            
            if "기술적 지표 (RSI/이평선)" in focus_list:
                instructions.append("""
---
### 📉 기술적 지표 분석 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오.**

**1. RSI (14일 기준)**
- 현재 RSI 값: [숫자]
- 해석: [과매수(>70) / 과매도(<30) / 중립]
- 최근 추세: [상승 / 하락 / 횡보]
- 다이버전스: [발생 여부 및 의미]

**2. 이동평균선 분석**
- **5일선**: 현재가 [가격], 괴리율 [%], [지지선/저항선/돌파] 역할
- **20일선**: 현재가 [가격], 괴리율 [%], [지지선/저항선/돌파] 역할
- **60일선**: 현재가 [가격], 괴리율 [%], [지지선/저항선/돌파] 역할
- **120일선**: 현재가 [가격], 괴리율 [%], [지지선/저항선/돌파] 역할
- **골든크로스/데드크로스**: [발생 여부 및 시점]
- **배열 상태**: [정배열/역배열]

**3. 추가 기술적 지표**
- MACD: [현재 상태 및 신호]
- 볼린저 밴드: [상단/중단/하단 위치]
- 거래량 추세: [증가/감소/평균 대비]

**💡 기술적 분석 결론**:
- 단기 방향성: [상승 / 하락 / 횡보]
- 매수 적정가: [가격대]
- 손절가: [가격]""")
            
            if "외국인/기관 수급 분석" in focus_list:
                instructions.append("""
---
### 🏦 외국인/기관 수급 분석 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오.**

**1. 외국인 동향**
- **최근 5일**: [순매수/순매도], 금액 [금액], 지분율 변화 [%p]
- **최근 20일**: [순매수/순매도], 금액 [금액], 지분율 변화 [%p]
- **최근 60일**: [순매수/순매도], 금액 [금액], 지분율 변화 [%p]
- **현재 외국인 지분율**: [%]
- **추세 해석**: [매집 / 이탈 / 중립]

**2. 기관 동향**
- **투신**: 최근 5일 [금액], 최근 20일 [금액], 해석 [매집/이탈]
- **연기금**: 최근 5일 [금액], 최근 20일 [금액], 해석 [매집/이탈]
- **보험**: 최근 5일 [금액], 최근 20일 [금액], 해석 [매집/이탈]

**3. 수급 종합 판단**
- 수급 모멘텀: [긍정적 / 부정적 / 중립]
- 스마트머니 흐름: [유입 / 이탈 / 대기]
- 수급 기반 단기 전망: [상승 / 하락 / 횡보]""")
            
            if "경쟁사 비교 및 업황" in focus_list:
                instructions.append("""
---
### 🏭 경쟁사 비교 및 업황 분석 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오.**

**1. 업종 현황**
- 산업 사이클 위치: [도입기 / 성장기 / 성숙기 / 쇠퇴기]
- 업종 전망: [긍정적 / 중립 / 부정적]
- 주요 트렌드: [2-3가지 핵심 트렌드]
- 규제 환경: [우호적 / 중립 / 부정적]

**2. 주요 경쟁사 비교**
- **해당 기업**: 시가총액 [금액], PER [배수], PBR [배수], 매출 성장률 [%], 영업이익률 [%], ROE [%]
- **경쟁사 A ([기업명])**: 시가총액 [금액], PER [배수], PBR [배수], 매출 성장률 [%], 영업이익률 [%], ROE [%]
- **경쟁사 B ([기업명])**: 시가총액 [금액], PER [배수], PBR [배수], 매출 성장률 [%], 영업이익률 [%], ROE [%]
- **업종 평균**: PER [배수], PBR [배수], 매출 성장률 [%], 영업이익률 [%], ROE [%]

**3. 경쟁 우위 분석**
- **강점 (Strengths)**: [2-3가지]
- **약점 (Weaknesses)**: [2-3가지]
- **기회 (Opportunities)**: [2-3가지]
- **위협 (Threats)**: [2-3가지]

**💡 경쟁력 종합 평가**: [업종 내 상위 / 중위 / 하위 및 근거]""")
            
            if "단기/중기 매매 전략" in focus_list:
                instructions.append("""
---
### 💰 단기/중기 매매 전략 (반드시 작성 - 생략 금지)
**⚠️ 이 섹션을 반드시 작성하십시오.**

**[단기 전략 (1주~1개월)]**
- **추천 포지션**: [매수/매도/관망], 근거: [근거 설명]
- **1차 진입가**: [가격], 근거: [지지선 기반]
- **2차 진입가**: [가격], 근거: [강한 지지선]
- **1차 목표가**: [가격], 근거: [저항선 기반]
- **2차 목표가**: [가격], 근거: [강한 저항선]
- **손절가**: [가격], 근거: [근거]

**[중기 전략 (1~6개월)]**
- **추천 포지션**: [매수/홀드/매도], 근거: [근거 설명]
- **분할 매수 전략**: 1차 [%], 2차 [%], 3차 [%], 근거: [근거]
- **목표 수익률**: [%], 근거: [근거]
- **포트폴리오 권장 비중**: [%], 근거: [투자 성향 고려]

**[리스크 관리]**
- 손절 기준: [조건 명시]
- 익절 기준: [조건 명시]
- 모니터링 포인트: [3가지]
- 포지션 조정 트리거: [구체적 조건]""")
            
            if "투자성향별 포트폴리오 적정보유비중" in focus_list:
                instructions.append("""
---
## 🎯 투자성향별 포트폴리오 적정보유비중 (필수 - 절대 생략 금지)

**⚠️⚠️⚠️ 이 섹션은 매우 중요합니다. 반드시 모든 항목을 상세히 작성하십시오. ⚠️⚠️⚠️**

---

### 📌 STEP 1: 성장주 vs 가치주 판단

**이 종목의 분류**: [성장주 / 가치주 / 혼합형] 중 하나 선택

**판단 근거** (각 기준별로 평가):
- **PER**: 해당 종목 [숫자]배 → 성장주 특성(>20배) vs 가치주 특성(<15배) → **[성장/가치] 판정**
- **PBR**: 해당 종목 [숫자]배 → 성장주 특성(>3배) vs 가치주 특성(<1.5배) → **[성장/가치] 판정**
- **매출성장률**: 해당 종목 [%] → 성장주 특성(>15%) vs 가치주 특성(<5%) → **[성장/가치] 판정**
- **배당수익률**: 해당 종목 [%] → 성장주 특성(<1%) vs 가치주 특성(>3%) → **[성장/가치] 판정**
- **이익재투자 성향**: [설명] → 성장주 특성(높음) vs 가치주 특성(낮음) → **[성장/가치] 판정**

**최종 판정**: 이 종목은 **[성장주/가치주]**입니다.

---

### 📌 STEP 2: 성장주/가치주별 핵심 지표 심층 분석

#### 🌱 [성장주로 판단된 경우 - 아래 5가지를 반드시 모두 분석]

**1️⃣ 매출 성장률 분석 (CAGR)**
- 5년 전 매출액: [금액] (기준점)
- 4년 전 매출액: [금액], YoY 성장률 [%]
- 3년 전 매출액: [금액], YoY 성장률 [%]
- 2년 전 매출액: [금액], YoY 성장률 [%]
- 1년 전 매출액: [금액], YoY 성장률 [%]
- 최근 매출액: [금액], YoY 성장률 [%]
- **5년 CAGR**: [%]
- **성장 추세**: [가속 / 안정 / 둔화]
- **평가**: [우수 / 양호 / 주의]

**2️⃣ Cash Flow 추이 분석**
- 3년 전: 영업CF [금액], 투자CF [금액], 잉여CF [금액], FCF마진 [%]
- 2년 전: 영업CF [금액], 투자CF [금액], 잉여CF [금액], FCF마진 [%]
- 최근: 영업CF [금액], 투자CF [금액], 잉여CF [금액], FCF마진 [%]
- **현금흐름 추세**: [개선 / 안정 / 악화]
- **현금창출력 평가**: [강함 / 보통 / 약함]

**3️⃣ ROI (투자수익률) 분석**
- **ROE**: 3년 전 [%] → 2년 전 [%] → 최근 [%], 추세 [개선/악화]
- **ROA**: 3년 전 [%] → 2년 전 [%] → 최근 [%], 추세 [개선/악화]
- **ROIC**: 3년 전 [%] → 2년 전 [%] → 최근 [%], 추세 [개선/악화]
- **투자효율성 평가**: [높음 / 보통 / 낮음]

**4️⃣ Profit Margin 추이**
- 3년 전: 매출총이익률 [%], 영업이익률 [%], 순이익률 [%]
- 2년 전: 매출총이익률 [%], 영업이익률 [%], 순이익률 [%]
- 최근: 매출총이익률 [%], 영업이익률 [%], 순이익률 [%]
- **수익성 전환**: [흑자전환 완료 / 흑자전환 진행중 / 적자 지속]
- **마진 추세**: [확대 / 유지 / 축소]

**5️⃣ 성장 지속성 평가**
- **변동성 (표준편차)**: [숫자] - [낮음/보통/높음]
- **분기별 실적 일관성**: [일관적 / 변동적]
- **가이던스 달성률**: [%]
- **성장 지속 가능성**: [높음 / 보통 / 낮음]

---

#### 💎 [가치주로 판단된 경우 - 아래 5가지를 반드시 모두 분석]

**1️⃣ 시장 점유율 분석**
- 3년 전: 시장점유율 [%], 순위 [위] (기준점)
- 2년 전: 시장점유율 [%], 변화 [+/-]%p, 순위 [위]
- 최근: 시장점유율 [%], 변화 [+/-]%p, 순위 [위]
- **점유율 추세**: [확대 / 유지 / 축소]
- **⚠️ 점유율 감소 시 경고**: [배당 축소 가능성 평가]

**2️⃣ 배당금 안정성 분석**
- 5년 전: 주당배당금 [원/달러], 배당수익률 [%], 배당성향 [%]
- 3년 전: 주당배당금 [원/달러], 배당수익률 [%], 배당성향 [%]
- 최근: 주당배당금 [원/달러], 배당수익률 [%], 배당성향 [%]
- **배당 연속 지급**: [X년 연속]
- **배당 증가 추세**: [증가 / 유지 / 감소]
- **배당 안정성 등급**: [AAA / AA / A / BBB / 주의]

**3️⃣ 주가 안정성 (변동성) 분석**
- **베타(β)**: 해당 종목 [숫자], 업종 평균 [숫자], 평가 [낮음/보통/높음]
- **52주 변동폭**: 해당 종목 [%], 업종 평균 [%], 평가 [안정/보통/변동]
- **최대 낙폭(MDD)**: 해당 종목 [%], 업종 평균 [%], 평가 [양호/주의]
- **변동성 등급**: [매우 안정 / 안정 / 보통 / 변동적]

**4️⃣ 이익률 변화 분석**
- 3년 전: 영업이익률 [%], 순이익률 [%] (기준점)
- 2년 전: 영업이익률 [%], 순이익률 [%], 변화 [개선/악화]
- 최근: 영업이익률 [%], 순이익률 [%], 변화 [개선/악화]
- **마진 추세**: [상승 = 경쟁력 강화 / 하락 = 경쟁력 약화]
- **업종 대비**: [상위 / 중위 / 하위]

**5️⃣ EPS 변화 분석**
- 3년 전: EPS [원/달러] (기준점)
- 2년 전: EPS [원/달러], YoY 변화 [%], 컨센서스 [상회/하회]
- 최근: EPS [원/달러], YoY 변화 [%], 컨센서스 [상회/하회]
- 예상(다음해): EPS [원/달러], YoY 변화 [%]
- **EPS 성장 추세**: [안정 성장 / 변동 / 하락]
- **어닝 서프라이즈 빈도**: [자주 / 가끔 / 드뭄]

---

### 📌 STEP 3: 투자 성향별 권장 보유 비중

⚠️ **아래 세 가지 투자 성향 모두에 대해 반드시 상세히 작성하십시오.**

---

#### 🦁 1. 공격적 투자자 (Aggressive Investor)

**투자자 특성**:
- 높은 변동성 감내 가능
- 고수익 추구형 (연 20% 이상 목표)
- 투자 기간: 단기~중기 (6개월~2년)
- 손실 허용 범위: -30% 이상

**권장 보유 비중**: **[X]%** (전체 주식 포트폴리오 대비)

**비중 산정 근거**:
1. [첫 번째 근거 - 성장성/수익성 관점에서 구체적으로]
2. [두 번째 근거 - 리스크/변동성 관점에서 구체적으로]  
3. [세 번째 근거 - 업종/시장 상황 관점에서 구체적으로]

**주의사항 및 리스크**:
- ⚠️ [핵심 리스크 1]
- ⚠️ [핵심 리스크 2]

**추천 진입 전략**:
- 진입 시점: [조건]
- 분할 매수: [1차 X%, 2차 X%, 3차 X%]

---

#### ⚖️ 2. 중립적 투자자 (Moderate Investor)

**투자자 특성**:
- 성장과 안정의 균형 중시
- 적정 수익 추구형 (연 10-15% 목표)
- 투자 기간: 중기 (1-3년)
- 손실 허용 범위: -15% 내외

**권장 보유 비중**: **[X]%** (전체 주식 포트폴리오 대비)

**비중 산정 근거**:
1. [첫 번째 근거 - 균형 잡힌 관점에서 구체적으로]
2. [두 번째 근거 - 리스크 대비 수익 관점에서 구체적으로]
3. [세 번째 근거 - 분산 투자 관점에서 구체적으로]

**리밸런싱 제안**:
- 비중 확대 조건: [구체적 조건 - 예: 주가 X% 하락 시]
- 비중 축소 조건: [구체적 조건 - 예: PER X배 초과 시]
- 리밸런싱 주기: [월간/분기/반기]

**포트폴리오 내 역할**:
- [핵심 / 위성 / 분산] 종목으로 편입 권장

---

#### 🛡️ 3. 보수적 투자자 (Conservative Investor)

**투자자 특성**:
- 원금 보존 최우선
- 안정적 수익 추구형 (연 5-8% + 배당)
- 투자 기간: 장기 (3년 이상)
- 손실 허용 범위: -10% 미만

**권장 보유 비중**: **[X]%** (전체 주식 포트폴리오 대비)

**비중 산정 근거**:
1. [첫 번째 근거 - 안전성 관점에서 구체적으로]
2. [두 번째 근거 - 배당/현금흐름 관점에서 구체적으로]
3. [세 번째 근거 - 방어적 특성 관점에서 구체적으로]

**대안 제시** (비중이 낮은 경우):
- 대신 추천하는 자산: [구체적 대안 - 예: 배당 ETF, 채권, 우선주 등]
- 이유: [대안이 더 적합한 이유]

**안전 마진 확보 전략**:
- 매수 적정가: [현재가 대비 X% 하락 시]
- 필수 체크 포인트: [배당 지속성, 부채비율 등]

---

### 📌 투자 성향별 비중 요약

**🦁 공격적 투자자**: 권장 비중 **[X]%**, 핵심 근거 - [한줄 요약], 주의사항 - [핵심 리스크]

**⚖️ 중립적 투자자**: 권장 비중 **[X]%**, 핵심 근거 - [한줄 요약], 주의사항 - [핵심 리스크]

**🛡️ 보수적 투자자**: 권장 비중 **[X]%**, 핵심 근거 - [한줄 요약], 주의사항 - [핵심 리스크]

**💡 최종 권고**: [이 종목의 전반적인 투자 매력도와 누구에게 가장 적합한지 1-2문장으로 요약]
""")
            
            return "\n".join(instructions)
        
        # 상세 지시사항 생성
        detailed_instructions = build_detailed_analysis_instructions(selected_focus_list)
        
        # =====================================================
        # [핵심 수정] 기업 기본정보 섹션 생성
        # =====================================================
        company_info_section = f"""
## 🏢 기업 기본 정보 (Company Overview)

- **정식 기업명**: **{company_info['long_name']}**
- **티커(심볼)**: {ticker}
- **섹터 (Sector)**: **{company_info['sector']}**
- **산업 (Industry)**: **{company_info['industry']}**
- **국가**: {company_info['country']}
- **시가총액**: {company_info['market_cap']}
- **직원 수**: {company_info['employees']}

⚠️ **확인**: 이 분석은 **{company_info['long_name']}** ({ticker})에 대한 것입니다. 
이 기업은 **{company_info['sector']}** 섹터의 **{company_info['industry']}** 산업에 속합니다.
다른 기업과 혼동하지 마십시오.

---
"""

        add_log(f"📝 프롬프트 조립 시작 (Mode: {mode})")
        if mode == "10K":
            prompt = f"""
[역할] 월가 수석 애널리스트 (펀더멘털 & 장기 투자 전문가)

⚠️ **중요: 모든 응답은 반드시 한글(Korean)로 작성하십시오.**

{company_info_section}

[자료] 최신 SEC 10-K 보고서 (Annual Report)

[지시사항]
당신은 월가 최고의 주식 애널리스트입니다.
위 종목의 **최신 SEC 10-K 보고서**를 기반으로 기업의 기초 체력과 장기 비전을 심층 분석해 주세요.
**주의: '{ticker}'는 '{company_info['long_name']}'입니다. 다른 기업과 혼동하지 마십시오.**
필요하다면 Google Search 도구를 활용하여 최신 데이터를 교차 검증하세요.

**[출력 형식]**
- 마크다운(Markdown) 형식을 사용하여 깔끔하게 작성하세요.
- 섹션 헤더, 불렛 포인트, 볼드체를 적절히 활용하세요.

**[필수 분석 항목]**
1. **비즈니스 개요 (Overview)**: 
   - 산업 내 위치, 비즈니스 모델의 강점, Fiscal Year End 날짜.

2. **MD&A 및 미래 전망 (Outlook)**: (중요)
   - 경영진이 제시하는 내년도 시장 전망 및 전략.
   - 매출 및 수익성 성장에 대한 경영진의 자신감 톤(Tone) 분석.

3. **핵심 리스크 및 법적 이슈 (Risk & Legal)**:
   - 사업에 치명적일 수 있는 Risk Factors.
   - 진행 중인 중요한 소송(Legal Proceedings)이나 규제 이슈 여부.

4. **재무제표 정밀 분석 (Financials)**:
   - 대차대조표, 손익계산서, 현금흐름표의 주요 변동 사항.
   - **부채 만기 구조(Debt Maturity)** 및 유동성 위기 가능성 점검.

5. **주요 이벤트 (Key Events)**:
   - 자사주 매입, M&A, 경영진 변동, 대규모 구조조정 등.

[결론]
기업의 장기적인 투자가치와 해자(Moat)에 대한 종합 평가.
"""
        elif mode == "10Q":
            prompt = f"""
[역할] 실적 모멘텀 및 트렌드 분석가

⚠️ **중요: 모든 응답은 반드시 한글(Korean)로 작성하십시오.**

{company_info_section}

[자료] 최신 SEC 10-Q 보고서 (Quarterly Report)

[지시사항]
위 종목의 **최신 SEC 10-Q 보고서**를 기반으로 **직전 분기 대비 변화(Trend)**에 집중하여 분석 보고서를 작성하세요.
**주의: '{ticker}'는 '{company_info['long_name']}'입니다.**
단기적인 실적 흐름과 경영진의 가이던스 변화를 포착하는 것이 핵심입니다.

**[출력 형식]**
- 마크다운(Markdown) 형식 사용.

**[필수 분석 항목]**
1. **실적 요약 (Earnings Summary)**:
   - 매출 및 EPS의 전년 동기(YoY) 및 전 분기(QoQ) 대비 성장률.
   - 시장 예상치(Consensus) 상회/하회 여부 및 그 원인.

2. **가이던스 변화 (Guidance Update)**: (매우 중요)
   - 경영진이 제시한 향후 실적 전망치가 상향되었는가, 하향되었는가?
   - 전망 변경의 구체적인 근거 (수요 증가, 비용 절감 등).

3. **부문별 성과 (Segment Performance)**:
   - 주요 사업 부문별 매출 및 이익 증감 추이.
   - 가장 빠르게 성장하는 부문과 둔화되는 부문 식별.

4. **현금흐름 및 비용 (Cash & Costs)**:
   - 영업활동 현금흐름의 변화.
   - R&D 및 마케팅 비용 지출 추이 (효율성 분석).

[결론]
이번 분기 실적이 일시적인지 구조적인 추세인지 판단하고, 단기/중기 투자 매력도 제시.
"""
        elif mode == "8K":
            prompt = f"""
[역할] 속보 뉴스 데스크 / 이벤트 드리븐 트레이더

⚠️ **중요: 모든 응답은 반드시 한글(Korean)로 작성하십시오.**

{company_info_section}

[자료] 최신 SEC 8-K 보고서 (Current Report)

[지시사항]
위 종목의 **최신 SEC 8-K 보고서**를 분석하여, 발생한 **특정 사건(Event)**의 내용과 주가에 미칠 영향을 즉각적으로 분석하세요.
**주의: '{ticker}'는 '{company_info['long_name']}'입니다.**
가장 최근에 공시된 중요한 사건 하나에 집중하십시오.

**[출력 형식]**
- 마크다운(Markdown) 형식 사용.
- 핵심 위주로 간결하고 명확하게 작성.

**[필수 분석 항목]**
1. **공시 사유 (Triggering Event)**:
   - 8-K가 제출된 핵심 이유 (Item 번호 및 제목 확인).
   - 예: 실적 발표, 주요 계약 체결, 경영진 사퇴, M&A, 유상증자 등.

2. **세부 내용 (Details)**:
   - 계약 금액, 거래 조건, 변경된 인물의 프로필 등 구체적 팩트 정리.
   - 재무적으로 즉각적인 영향이 있는가?

3. **호재/악재 판별 (Impact Analysis)**:
   - 이 뉴스가 주가에 단기적으로 긍정적인지(Bullish) 부정적인지(Bearish) 명확한 판단.
   - 시장의 예상 범위를 벗어난 서프라이즈 요소가 있는지.

[결론]
이 뉴스에 대해 투자자가 취해야 할 즉각적인 대응 전략 (매수 기회 vs 리스크 관리).
"""
        else:
            # =====================================================
            # [핵심 수정] MAIN 모드 프롬프트 - 완전 개편
            # =====================================================
            prompt = f"""
[역할] 월스트리트 수석 애널리스트 / 투자 전략가

⚠️⚠️⚠️ **[최우선 지시사항 - 반드시 준수]** ⚠️⚠️⚠️
1. **모든 응답은 반드시 한글(Korean)로 작성하십시오. 영어 사용 금지.**
2. **아래 모든 섹션을 빠짐없이 상세하게 작성하십시오.**
3. **어떤 항목도 "생략", "축약", "간략화" 하지 마십시오.**
4. **표(Table)가 있는 섹션은 반드시 표를 작성하십시오.**
5. **각 섹션의 "💡 종합 평가/의견" 부분을 반드시 작성하십시오.**

---

{company_info_section}

---

## ⚙️ 분석 설정

- **투자 관점**: {viewpoint}
- **분석 레벨**: {analysis_depth}
- **중점 분석 항목**: {focus}

{level_instruction}

---

## 📊 제공된 데이터

### 주가 데이터
{data_summary}

### 재무 지표
{fin_str}

### 관련 뉴스
{news_text}

---

# 📋 필수 분석 항목

⚠️ **아래 모든 섹션을 빠짐없이 상세하게 작성하십시오. 생략 시 분석 품질이 크게 저하됩니다.**
⚠️ **표(Table) 형식을 사용하지 말고, 불릿 포인트(-)와 서술형으로 작성하십시오.**

{detailed_instructions}

{scenario_section}

---

## 🔮 종합 결론 및 투자 의견

### 최종 투자 의견
- **투자 의견**: [매수 / 매도 / 관망] 중 명확히 선택
- **확신도**: [매우 높음 / 높음 / 보통 / 낮음]
- **투자 기간**: [단기 / 중기 / 장기]

### 핵심 근거 (Top 3)
1. **[근거 1 제목]**: [구체적 설명]
2. **[근거 2 제목]**: [구체적 설명]
3. **[근거 3 제목]**: [구체적 설명]

### 목표 주가
- **하단 목표가**: [금액] (현재가 대비 [%])
- **기본 목표가**: [금액] (현재가 대비 [%])
- **상단 목표가**: [금액] (현재가 대비 [%])

### 주요 리스크 요인
1. ⚠️ [리스크 1]: 영향도 [상/중/하]
2. ⚠️ [리스크 2]: 영향도 [상/중/하]

---

⚠️ **[최종 점검]**: 
- 기업 기본 정보(섹터, 산업)가 상단에 포함되어 있습니까? ✓
- 모든 분석 항목이 빠짐없이 작성되었습니까? ✓
- 투자성향별 비중이 3가지 모두 상세히 작성되었습니까? ✓
- 시나리오 분석이 요청된 경우, 3가지 시나리오와 확률이 모두 작성되었습니까? ✓

누락된 항목이 있다면 지금 즉시 추가하십시오.
"""
        
        st.session_state['temp_data'] = {
            'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': [],
            'company_info': company_info  # 기업 정보 저장
        }
        add_log(f"✅ [STEP 1] 데이터 준비 완료 (Prompt Length: {len(prompt)})")
        return True

    except Exception as e:
        add_log(f"❌ [FATAL] Step 1 Error: {str(e)}")
        st.error(f"Data Step Error: {e}")
        return False

# ---------------------------------------------------------
# 5. 사이드바 UI (Compact Version)
# ---------------------------------------------------------
st.sidebar.subheader("🎯 분석 옵션")

viewpoint_mapping = {"단기 (1주~1개월)": "3mo", "스윙 (1~3개월)": "6mo", "중기 (6개월~1년)": "2y", "장기 (1~3년)": "5y"}
selected_viewpoint = st.sidebar.select_slider("", options=list(viewpoint_mapping.keys()), value="중기 (6개월~1년)", label_visibility="collapsed")
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

api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.sidebar.error("⚠️ Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다.")

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
        else:
            st.button("🔍 분석 시작", disabled=True, key="btn_s_main_disabled", help="API Key가 필요합니다.")
    
    st.markdown("##### 📑 공시 분석")
    c1, c2, c3 = st.columns(3)
    with c1: st.button("10-K", key="btn_s_10k", on_click=handle_search_click, args=("10K", prompt_mode_search))
    with c2: st.button("10-Q", key="btn_s_10q", on_click=handle_search_click, args=("10Q", prompt_mode_search))
    with c3: st.button("8-K", key="btn_s_8k", on_click=handle_search_click, args=("8K", prompt_mode_search))

selected_tickers = []
if 'selected' in st.query_params:
    selected_str = st.query_params['selected']
    if selected_str:
        selected_tickers = [t.strip() for t in selected_str.split(',') if t.strip()]
        for t in selected_tickers:
            st.session_state[f"chk_{t}"] = True

with tab_fav:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.75, 0.25])
    with c1: st.text_input("종목 추가 (콤마 구분)", placeholder="AAPL, TSLA", label_visibility="collapsed", key="new_ticker_input")
    with c2: st.button("➕", on_click=add_ticker_logic)

    fav_df = st.session_state.get('portfolio_df', pd.DataFrame(columns=['ticker', 'name']))
    
    if not fav_df.empty:
        for t in fav_df['ticker']:
            if st.session_state.get(f"chk_{t}", False):
                if t not in selected_tickers: selected_tickers.append(t)
    
    if not fav_df.empty:
        import json
        tickers_data = []
        for idx, row in fav_df.iterrows():
            is_checked = row['ticker'] in selected_tickers or st.session_state.get(f"chk_{row['ticker']}", False)
            tickers_data.append({'ticker': row['ticker'], 'name': str(row['name']), 'checked': is_checked})
        tickers_json = json.dumps(tickers_data)
        initial_selected = json.dumps(selected_tickers)
        
        count_selected = len(selected_tickers)
        header_label = f"📂 포트폴리오 ({count_selected}개 선택)" if count_selected > 0 else "📂 포트폴리오 (미선택)"
        
        with st.expander(header_label, expanded=True):
            row_count = (len(fav_df) + 1) // 2
            grid_height = min(row_count * 60, 240)
            dynamic_height = 160 + grid_height 
            
            grid_html = f"""<style>* {{margin: 0; padding: 0; box-sizing: border-box;}} .pf-wrapper {{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}} .pf-content {{background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 0px;}} .selected-display {{background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 8px 10px; margin-bottom: 10px; min-height: 32px;}} .selected-label {{font-size: 11px; color: #15803d; font-weight: 600; margin-bottom: 4px;}} .selected-tickers {{font-size: 12px; color: #166534; font-weight: 500; word-break: break-word;}} .selected-empty {{font-size: 11px; color: #9ca3af; font-style: italic;}} .select-all {{display: flex; align-items: center; gap: 8px; padding: 8px 0 10px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 10px;}} .select-all input[type="checkbox"] {{width: 16px; height: 16px; accent-color: #3b82f6; cursor: pointer;}} .select-all label {{font-size: 13px; color: #475569; cursor: pointer; user-select: none;}} .pf-grid-wrapper {{max-height: 240px; overflow-y: auto; overflow-x: hidden; padding-right: 4px;}} .pf-grid-wrapper::-webkit-scrollbar {{width: 6px;}} .pf-grid-wrapper::-webkit-scrollbar-track {{background: #f1f5f9; border-radius: 3px;}} .pf-grid-wrapper::-webkit-scrollbar-thumb {{background: #cbd5e1; border-radius: 3px;}} .pf-grid-wrapper::-webkit-scrollbar-thumb:hover {{background: #94a3b8;}} .pf-grid {{display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; width: 100%;}} .pf-item {{display: flex; align-items: center; gap: 5px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 5px 6px; height: 48px; transition: all 0.15s; cursor: pointer; min-width: 0; width: 100%; box-sizing: border-box; overflow: hidden;}} .pf-item:hover {{background: #f1f5f9; border-color: #cbd5e1;}} .pf-item.selected {{background: #eff6ff; border-color: #3b82f6;}} .pf-item input[type="checkbox"] {{width: 14px; height: 14px; accent-color: #3b82f6; cursor: pointer; flex-shrink: 0; margin: 0;}} .pf-info {{flex: 1; min-width: 0; overflow: hidden;}} .pf-ticker {{font-size: 11px; font-weight: 600; color: #1e293b; line-height: 1.2;}} .pf-name {{font-size: 9px; color: #9ca3af; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; line-height: 1.3; margin-top: 1px;}} .pf-delete {{background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 14px; padding: 2px 4px; border-radius: 4px; transition: all 0.15s; flex-shrink: 0; line-height: 1;}} .pf-delete:hover {{color: #ef4444; background: #fef2f2;}}</style>
            <div class="pf-wrapper"><div class="pf-content" id="pfContent"><div class="selected-display"><div class="selected-label">✅ 선택된 종목</div><div class="selected-tickers" id="selectedDisplay"><span class="selected-empty">선택된 종목이 없습니다</span></div></div><div class="select-all"><input type="checkbox" id="selectAllCb"><label for="selectAllCb">전체 선택</label></div><div class="pf-grid-wrapper"><div class="pf-grid" id="pfGrid"></div></div></div></div>
            <script>const tickersData={tickers_json};let selectedTickers={initial_selected};function renderGrid(){{const grid=document.getElementById('pfGrid');grid.innerHTML='';tickersData.forEach((item,idx)=>{{const isSelected=selectedTickers.includes(item.ticker);const div=document.createElement('div');div.className='pf-item'+(isSelected?' selected':'');div.innerHTML=`<input type="checkbox" ${{isSelected?'checked':''}} data-ticker="${{item.ticker}}"><div class="pf-info"><div class="pf-ticker">${{item.ticker}}</div><div class="pf-name" title="${{item.name}}">${{item.name}}</div></div><button class="pf-delete" data-ticker="${{item.ticker}}">×</button>`;grid.appendChild(div);}});bindEvents();updateSelectAllState();updateAllDisplays();}}function updateAllDisplays(){{const display=document.getElementById('selectedDisplay');if(selectedTickers.length>0){{display.innerHTML=selectedTickers.join(', ');}}else{{display.innerHTML='<span class="selected-empty">선택된 종목이 없습니다</span>';}}}}function bindEvents(){{document.querySelectorAll('.pf-item input[type="checkbox"]').forEach(cb=>{{cb.addEventListener('change',function(e){{e.stopPropagation();const ticker=this.dataset.ticker;if(this.checked){{if(!selectedTickers.includes(ticker)){{selectedTickers.push(ticker);}}}}else{{selectedTickers=selectedTickers.filter(t=>t!==ticker);}}this.closest('.pf-item').classList.toggle('selected',this.checked);updateSelectAllState();updateAllDisplays();syncToStreamlit();}});}});document.querySelectorAll('.pf-delete').forEach(btn=>{{btn.addEventListener('click',function(e){{e.stopPropagation();const ticker=this.dataset.ticker;const item=this.closest('.pf-item');item.style.transform='scale(0.9)';item.style.opacity='0';setTimeout(()=>{{const url=new URL(window.parent.location.href);url.searchParams.set('del_ticker',ticker);window.parent.location.href=url.toString();}},150);}});}});document.querySelectorAll('.pf-item').forEach(item=>{{item.addEventListener('click',function(e){{if(e.target.tagName==='INPUT'||e.target.tagName==='BUTTON')return;const cb=this.querySelector('input[type="checkbox"]');cb.checked=!cb.checked;cb.dispatchEvent(new Event('change'));}});}});}}document.getElementById('selectAllCb').addEventListener('change',function(){{const isChecked=this.checked;document.querySelectorAll('.pf-item input[type="checkbox"]').forEach(cb=>{{cb.checked=isChecked;cb.closest('.pf-item').classList.toggle('selected',isChecked);}});if(isChecked){{selectedTickers=tickersData.map(t=>t.ticker);}}else{{selectedTickers=[];}}updateAllDisplays();syncToStreamlit();}});function updateSelectAllState(){{const allCheckboxes=document.querySelectorAll('.pf-item input[type="checkbox"]');const checkedCount=document.querySelectorAll('.pf-item input[type="checkbox"]:checked').length;const selectAllCb=document.getElementById('selectAllCb');selectAllCb.checked=checkedCount===allCheckboxes.length&&allCheckboxes.length>0;selectAllCb.indeterminate=checkedCount>0&&checkedCount<allCheckboxes.length;}}function syncToStreamlit(){{const url=new URL(window.parent.location.href);if(selectedTickers.length>0){{url.searchParams.set('selected',selectedTickers.join(','));}}else{{url.searchParams.delete('selected');}}window.parent.history.replaceState(null,'',url.toString());}}renderGrid();</script>"""
            
            st.components.v1.html(grid_html, height=dynamic_height, scrolling=False)
            
    else:
        st.markdown("""<div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;"><span style="font-size: 14px; font-weight: 600; color: #1e293b;">📂 포트폴리오</span><span style="font-size: 11px; color: #9ca3af; font-style: italic;">비어있음</span></div>""", unsafe_allow_html=True)
    st.markdown('<div style="height: 10px"></div>', unsafe_allow_html=True)
    
    c_chk_p, c_btn_p = st.columns([0.5, 0.5])
    with c_chk_p: prompt_mode_port = st.checkbox("☑️ 프롬프트만", key="chk_prompt_port", value=True)
    with c_btn_p: 
        if st.button("🚀 종합 분석 시작", type="primary", key="btn_run_main"):
            if 'selected' in st.query_params:
                selected_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]
            start_analysis_process(selected_tickers, "MAIN", prompt_mode_port)
    
    st.markdown("##### 📑 포트폴리오 공시 분석")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("연간 실적 (10-K)", key="btn_p_10k"):
            if 'selected' in st.query_params:
                selected_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]
            start_analysis_process(selected_tickers, "10K", prompt_mode_port)
        if st.button("수시 공시 (8-K)", key="btn_p_8k"):
            if 'selected' in st.query_params:
                selected_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]
            start_analysis_process(selected_tickers, "8K", prompt_mode_port)
    with c2:
        if st.button("분기 실적 (10-Q)", key="btn_p_10q"):
            if 'selected' in st.query_params:
                selected_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]
            start_analysis_process(selected_tickers, "10Q", prompt_mode_port)

st.sidebar.markdown('<hr>', unsafe_allow_html=True)
st.sidebar.subheader("🤖 AI 모델 선택")
model_options = [
    "gemini-1.5-pro",          
    "gemini-2.0-flash-lite-preview-02-05", 
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",     
    "gemini-1.0-pro",          
    "gemini-flash-latest"
]
selected_model = st.sidebar.selectbox("기본 분석 모델", model_options, index=0, label_visibility="collapsed")
st.session_state['selected_model'] = selected_model

st.sidebar.markdown('<hr>', unsafe_allow_html=True)
with st.sidebar.expander("📜 시스템 실행 로그 (System Logs)", expanded=False):
    log_text = "\n".join(st.session_state['log_buffer'])
    st.text_area("Log Output", value=log_text, height=200, label_visibility="collapsed")
    if st.button("🧹 로그 초기화"):
        st.session_state['log_buffer'] = []
        st.rerun()

# ---------------------------------------------------------
# 6. 실행 컨트롤러 (오토 드라이브)
# ---------------------------------------------------------
st.title(f"📈 AI Hyper-Analyst V86")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    current_idx = st.session_state['proc_index']
    current_stage = st.session_state['proc_stage']
    
    if not targets:
        st.warning("⚠️ 분석할 종목을 선택해주세요.")
        st.session_state['is_analyzing'] = False
        st.stop()

    if current_idx >= len(targets):
        st.success("🎉 모든 분석이 완료되었습니다!")
        st.session_state['is_analyzing'] = False
        st.rerun() 
        st.stop()

    curr_ticker = targets[current_idx]
    
    total_steps = len(targets) * 2
    current_progress = (current_idx * 2 + (1 if current_stage > 1 else 0)) / total_steps
    st.progress(current_progress, text=f"🚀 [{current_idx+1}/{len(targets)}] {curr_ticker} 분석 진행 중...")

    if current_stage == 1:
        if current_idx == 0:
            collapse_sidebar()
            time.sleep(0.3)

        with st.spinner(f"📥 {curr_ticker}: 데이터 수집 및 프롬프트 생성 중..."):
            time.sleep(0.1) 
            success = step_fetch_data(curr_ticker, st.session_state['current_mode'])
            
            if success:
                st.session_state['proc_stage'] = 2 
            else:
                st.session_state['analysis_results'][curr_ticker] = {
                    'name': curr_ticker, 'df': pd.DataFrame(), 'report': "데이터 수집 실패", 'status': 'error', 'mode': st.session_state['current_mode']
                }
                st.session_state['proc_index'] = current_idx + 1
                st.session_state['proc_stage'] = 1
            
            st.rerun() 

    elif current_stage == 2:
        temp = st.session_state['temp_data']
        
        if st.session_state['prompt_mode']:
            st.session_state['analysis_results'][curr_ticker] = {
                'name': temp['name'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr'],
                'df': temp['df'], 'report': "프롬프트 생성 완료", 'news': [], 
                'model': "Manual", 'mode': st.session_state['current_mode'],
                'prompt': temp['prompt'], 'status': 'manual',
                'company_info': temp.get('company_info', {})
            }
        else:
            with st.spinner(f"🧠 {curr_ticker}: AI 분석 보고서 작성 중 (자동 재시도 포함)..."):
                time.sleep(0.1)
                try:
                    report, used_model = generate_with_fallback(temp['prompt'], api_key, st.session_state['selected_model'])
                    status = 'success'
                except Exception as e:
                    report = f"AI Error: {e}"
                    used_model = "Error"
                    status = 'error'
                
                st.session_state['analysis_results'][curr_ticker] = {
                    'name': temp['name'], 'tv_symbol': temp['tv_symbol'], 'is_kr': temp['is_kr'],
                    'df': temp['df'], 'report': sanitize_text(report), 'news': [], 
                    'model': used_model, 'mode': st.session_state['current_mode'],
                    'prompt': temp['prompt'], 'status': status,
                    'company_info': temp.get('company_info', {})
                }

        st.session_state['proc_index'] = current_idx + 1
        st.session_state['proc_stage'] = 1 
        st.rerun() 

# ---------------------------------------------------------
# 7. 결과 출력
# ---------------------------------------------------------
if not st.session_state['is_analyzing'] and st.session_state['analysis_results']:
    st.write("---")
    for ticker, data in st.session_state['analysis_results'].items():
        header_prefix = "📊"
        if data.get('status') == 'error': 
            header_prefix = "❌ (Error)"
            status_color = "red"
        elif data.get('status') == 'manual': 
            header_prefix = "📋 (Prompt)"
            status_color = "blue"
        else: 
            status_color = "green"

        # 기업 정보 표시
        company_info = data.get('company_info', {})
        sector_info = f" | 🏭 {company_info.get('sector', 'N/A')} > {company_info.get('industry', 'N/A')}" if company_info else ""

        with st.expander(f"{header_prefix} {data.get('name', ticker)} ({ticker}){sector_info}", expanded=True):
            st.caption(f"Mode: **{data.get('mode')}** | 🤖 Model: **{data.get('model')}** | Status: :{status_color}[{data.get('status', 'success')}]")
            
            # 기업 기본 정보 카드
            if company_info and company_info.get('sector') != '정보 없음':
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px 16px; border-radius: 8px; margin-bottom: 15px;">
                    <div style="color: white; font-size: 11px; opacity: 0.9;">📍 섹터/산업</div>
                    <div style="color: white; font-size: 14px; font-weight: 600;">{company_info.get('sector', 'N/A')} → {company_info.get('industry', 'N/A')}</div>
                    <div style="color: white; font-size: 11px; margin-top: 4px; opacity: 0.8;">시가총액: {company_info.get('market_cap', 'N/A')} | 직원: {company_info.get('employees', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if not data['df'].empty:
                if data.get('is_kr', False):
                    fig = go.Figure(data=[go.Candlestick(x=data['df'].index, open=data['df']['Open'], high=data['df']['High'], low=data['df']['Low'], close=data['df']['Close'], increasing_line_color='#ef5350', decreasing_line_color='#26a69a')])
                    fig.update_layout(height=350 if mobile_mode else 500, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    html = f"""<div id="chart_{ticker}" class="tv-chart-container" style="height:{chart_height}"></div><script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script><script type="text/javascript">new TradingView.widget({{"autosize": true, "symbol": "{data['tv_symbol']}", "interval": "D", "timezone": "Asia/Seoul", "theme": "light", "style": "1", "locale": "ko", "toolbar_bg": "#f1f3f6", "enable_publishing": false, "container_id": "chart_{ticker}"}});</script>"""
                    st.components.v1.html(html, height=int(chart_height.replace('px',''))+10)

            if data.get('status') == 'manual':
                st.markdown("<div style='text-align: right;'><b>아래 프롬프트를 복사하여 사용하세요. 👇</b></div>", unsafe_allow_html=True)
                st.link_button("🚀 Google Gemini 열기", "https://gemini.google.com/")
                st.code(data.get('prompt', '프롬프트 없음'), language='text')
            else:
                if data.get('status') == 'error':
                    st.error(data['report'])
                else:
                    st.markdown(f"{data['report']}")
            
            st.markdown("---")
            st.link_button("🚀 Google Gemini 열기", "https://gemini.google.com/")
            if data.get('status') == 'success':
                with st.expander("📄 분석 결과 텍스트 복사", expanded=False):
                    st.code(data['report'], language="text")

elif not st.session_state['is_analyzing']:
    st.info("👈 왼쪽 사이드바에서 종목을 선택하고 분석 버튼을 눌러주세요.")
