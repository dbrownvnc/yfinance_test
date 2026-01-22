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
            add_log(f"💾 [SAVE] 저장 완료. {len(df)}개.")
        except Exception as e:
            add_log(f"❌ [SAVE] 저장 실패: {str(e)}")

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
                    t_info = yf.Ticker(ticker).info
                    name = t_info.get('shortName') or t_info.get('longName') or ticker
                except Exception as e: 
                    name = ticker
                new_rows.append({'ticker': ticker, 'name': name})
            
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            st.session_state['portfolio_df'] = df
            save_state_to_csv()
            add_log("✅ [ADD] 추가 완료.")
    st.session_state['new_ticker_input'] = ""

load_data_to_state()

# ---------------------------------------------------------
# [최우선 처리] 삭제 요청
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
# 3. 유틸리티 함수
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
            items.append({'title': title, 'date_str': date_str, 'summary': desc, 'source': 'RSS'})
        return items
    except: return []

def get_realtime_news(ticker, name):
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    if not is_kr:
        try:
            rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            items = fetch_rss_realtime(rss_url, limit=5)
            for i in items: i['source'] = "Yahoo"
            news_items.extend(items)
        except: pass

        if not news_items:
            try:
                yf_news = yf.Ticker(ticker).news
                for item in yf_news[:5]:
                    title = item.get('title'); link = item.get('link')
                    if not title: continue
                    try: date_str = datetime.datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime("%m-%d %H:%M")
                    except: date_str = "최신"
                    news_items.append({'title': title, 'date_str': date_str, 'source': "Yahoo", 'summary': item.get('summary','')})
            except: pass

    if is_kr: search_query = f'"{name}"'
    else: search_query = f'{ticker} stock'
    
    q_encoded = urllib.parse.quote(search_query)
    url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ko&gl=KR&ceid=KR:ko"
    g_items = fetch_rss_realtime(url, limit=5)
    for n in g_items: n['source'] = "Google"
    news_items.extend(g_items)
    
    return news_items[:7]

def get_financial_metrics(info):
    try:
        def get_fmt(key): val = info.get(key); return f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A"
        return {
            "Free Cash Flow": get_fmt('freeCashflow'), "Current Ratio": get_fmt('currentRatio'),
            "Debt to Equity": get_fmt('debtToEquity'), "ROE": get_fmt('returnOnEquity'), 
            "Net Income": get_fmt('netIncome')
        }
    except: return {}

def sanitize_text(text):
    text = text.replace('$', '\$')
    text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
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
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name 
        except Exception as e:
            time.sleep(0.5); continue
    raise Exception("All models failed.")

def handle_search_click(mode, is_prompt):
    raw = st.session_state.get("s_input", "")
    if raw: start_analysis_process([t.strip() for t in raw.split(',') if t.strip()], mode, is_prompt)
    else: st.warning("티커 입력 필요")

def step_fetch_data(ticker, mode):
    add_log(f"📦 [STEP 1] 데이터 수집: {ticker}")
    
    # 기본값 초기화
    stock_name = ticker
    sector = "N/A"
    industry = "N/A"
    
    clean_code = re.sub(r'[^0-9]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker

    try:
        stock = yf.Ticker(ticker)
        
        # 1. Info 가져오기 (이름, 섹터, 산업)
        try:
            info = run_with_timeout(_fetch_info, args=(ticker,), timeout=6)
            if info:
                # 이름 우선순위: 포트폴리오 저장값 > Info ShortName > Info LongName
                fetched_name = info.get('shortName') or info.get('longName')
                sector = info.get('sector', 'Unknown Sector')
                industry = info.get('industry', 'Unknown Industry')
                
                if 'portfolio_df' in st.session_state:
                    p_df = st.session_state['portfolio_df']
                    row = p_df[p_df['ticker'] == ticker]
                    if not row.empty: stock_name = row.iloc[0]['name']
                    elif fetched_name: stock_name = fetched_name
                elif fetched_name: stock_name = fetched_name
        except:
            info = {}
            
        add_log(f"   -> 식별: {stock_name} | Sec: {sector} | Ind: {industry}")

        # 2. 주가 데이터
        period = st.session_state.get('selected_period_str', '1y')
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=10)
        
        if df is None or df.empty: 
            df = pd.DataFrame()
            data_summary = "No Price Data"
        else:
            curr = df['Close'].iloc[-1]
            data_summary = f"Current: {curr:.2f}, High: {df['High'].max():.2f}, Low: {df['Low'].min():.2f}\n{df.tail(30).to_string()}"

        fin_str = "N/A"; news_text = "N/A"
        
        if mode not in ["10K", "10Q", "8K"]:
            if info: fin_str = str(get_financial_metrics(info))
            if st.session_state.get('use_news', True):
                news = get_realtime_news(ticker, stock_name)
                if news: 
                    news_text = "\n".join([f"- {n['title']} ({n['date_str']})" for n in news])
                else: news_text = "최신 뉴스 없음"

        # 프롬프트 조립
        selected_focus_list = []
        for opt in opt_targets:
            if st.session_state.get(f"focus_{opt}", True): selected_focus_list.append(opt)
        focus = ", ".join(selected_focus_list)
        viewpoint = st.session_state.get('selected_viewpoint', 'General')
        analysis_depth = st.session_state.get('analysis_depth', "2. 표준 브리핑")
        
        # [수정됨] 시나리오 분석 지침 강화 (확률 및 이유 필수)
        level_instruction = ""
        if "5." in analysis_depth:
            level_instruction = """
            \n[시나리오 분석 필수 포함 사항]
            1. 낙관적(Bull) / 중립적(Base) / 비관적(Bear) 시나리오 3가지를 반드시 제시하십시오.
            2. 각 시나리오별 **'실현 확률(%)'**을 명시하고, 세 확률의 합은 정확히 100%가 되어야 합니다.
            3. 각 확률을 산정한 구체적인 **'근거(Rationale)'**를 설명하십시오.
            4. 각 시나리오별 예상 주가 범위(Target Price Range)를 제시하십시오.
            """
        
        if "투자성향별 포트폴리오 적정보유비중" in focus:
            level_instruction += """
            \n[특별 지시: 투자성향별 비중 제안]
            보고서 결론에 다음 3가지 성향별 권장 비중(%)과 논리를 서술하십시오:
            1. 🦁 공격적 (Aggressive)
            2. ⚖️ 중립적 (Moderate)
            3. 🛡️ 보수적 (Conservative)
            """

        growth_value_logic = """
        [핵심 지시사항: 성장주 vs 가치주 판단]
        1. 이 기업이 '성장주'인지 '가치주'인지 규정하고 이유를 설명하십시오.
        2. 성장주라면: 매출 성장률, Cash Flow, ROI, Profit Margin 전환, 지속성 중점 분석.
        3. 가치주라면: 시장 점유율, 배당 안정성, 주가 변동성, 이익률, EPS 트렌드 중점 분석.
        """
        level_instruction += growth_value_logic
        korean_enforcement = "\n\n**[중요] 답변은 반드시 자연스러운 '한국어'로 작성하십시오.**"

        # [수정됨] 프롬프트에 Sector/Industry 정보 추가
        base_info = f"[대상] {ticker}\n- 기업명: {stock_name}\n- 섹터(Sector): {sector}\n- 산업(Industry): {industry}"

        if mode == "10K":
            prompt = f"""[역할] 월가 애널리스트 (10-K 분석)\n{base_info}\n[자료] SEC 10-K 보고서 기반 분석.\n[분석] 비즈니스, MD&A, 리스크, 재무제표, 주요이벤트.\n{korean_enforcement}"""
        elif mode == "10Q":
            prompt = f"""[역할] 실적 트렌드 분석가 (10-Q 분석)\n{base_info}\n[자료] SEC 10-Q 보고서 기반 분석.\n[분석] 실적요약, 가이던스 변화, 부문별 성과.\n{korean_enforcement}"""
        elif mode == "8K":
            prompt = f"""[역할] 속보 뉴스 분석가 (8-K 분석)\n{base_info}\n[자료] SEC 8-K 보고서 기반 분석.\n[분석] 공시 사유, 세부 내용, 호재/악재 판별.\n{korean_enforcement}"""
        else:
            prompt = f"""
            [역할] 수석 애널리스트
            {base_info}
            [모드] {mode} / [관점] {viewpoint} / [레벨] {analysis_depth}
            [중점] {focus}
            
            {level_instruction}
            
            [데이터]
            {data_summary}
            [재무] {fin_str}
            [뉴스] {news_text}
            
            [지시] 위 데이터를 바탕으로 투자 보고서를 작성하십시오. 뉴스 내용을 상세히 반영하십시오.
            결론에 [매수/매도/관망] 의견을 제시하십시오.
            {korean_enforcement}
            """
        
        st.session_state['temp_data'] = {
            'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': []
        }
        return True

    except Exception as e:
        add_log(f"❌ Error: {str(e)}")
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

st.session_state['use_news'] = st.sidebar.toggle("뉴스 반영", value=True)

def toggle_focus_all():
    new_state = st.session_state['focus_all']
    for opt in opt_targets: st.session_state[f"focus_{opt}"] = new_state

with st.sidebar.expander("☑️ 중점 분석 항목", expanded=False):
    st.checkbox("전체 선택", key="focus_all", on_change=toggle_focus_all)
    for opt in opt_targets: st.checkbox(opt, key=f"focus_{opt}")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.sidebar.error("Secrets Key 설정 필요")

tab_search, tab_fav = st.sidebar.tabs(["⚡ 검색", "⭐ 포트폴리오"])
prompt_mode_search = False
prompt_mode_port = False

with tab_search:
    st.markdown("<br>", unsafe_allow_html=True) 
    single_input = st.text_input("티커 (예: 005930.KS)", key="s_input")
    c1, c2 = st.columns(2)
    with c1: prompt_mode_search = st.checkbox("프롬프트만", key="chk_prompt_single", value=True)
    with c2: st.button("🔍 시작", type="primary", on_click=handle_search_click, args=("MAIN", prompt_mode_search))
    
    st.markdown("##### 📑 공시")
    c1, c2, c3 = st.columns(3)
    with c1: st.button("10-K", on_click=handle_search_click, args=("10K", prompt_mode_search))
    with c2: st.button("10-Q", on_click=handle_search_click, args=("10Q", prompt_mode_search))
    with c3: st.button("8-K", on_click=handle_search_click, args=("8K", prompt_mode_search))

# [포트폴리오]
selected_tickers = []
if 'selected' in st.query_params:
    selected_str = st.query_params['selected']
    if selected_str: selected_tickers = [t.strip() for t in selected_str.split(',') if t.strip()]

with tab_fav:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.75, 0.25])
    with c1: st.text_input("종목 추가", placeholder="AAPL", label_visibility="collapsed", key="new_ticker_input")
    with c2: st.button("➕", on_click=add_ticker_logic)

    fav_df = st.session_state.get('portfolio_df', pd.DataFrame())
    
    if not fav_df.empty:
        import json
        tickers_data = []
        for idx, row in fav_df.iterrows():
            is_checked = row['ticker'] in selected_tickers
            tickers_data.append({'ticker': row['ticker'], 'name': str(row['name']), 'checked': is_checked})
        
        # HTML/JS Grid Code (Compact Version)
        grid_html = f"""<style>
        .pf-item {{display: flex; align-items: center; gap: 5px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 5px; margin-bottom: 5px; cursor: pointer;}}
        .pf-item.selected {{background: #eff6ff; border-color: #3b82f6;}}
        .pf-info {{flex: 1; overflow: hidden;}}
        .pf-ticker {{font-weight: bold; font-size: 12px;}}
        .pf-name {{font-size: 10px; color: #666;}}
        </style>
        <div id="pfGrid"></div>
        <script>
        const data={json.dumps(tickers_data)};
        const grid=document.getElementById('pfGrid');
        let selected={json.dumps(selected_tickers)};
        
        function update(){{
            const url=new URL(window.parent.location.href);
            if(selected.length>0) url.searchParams.set('selected',selected.join(','));
            else url.searchParams.delete('selected');
            window.parent.history.replaceState(null,'',url.toString());
        }}

        data.forEach(item=>{{
            const div=document.createElement('div');
            div.className='pf-item'+(item.checked?' selected':'');
            div.innerHTML=`<input type="checkbox" ${{item.checked?'checked':''}}> <div class="pf-info"><div class="pf-ticker">${{item.ticker}}</div><div class="pf-name">${{item.name}}</div></div> <button onclick="del('${{item.ticker}}')">×</button>`;
            div.onclick=(e)=>{{
                if(e.target.tagName==='BUTTON') return;
                const cb=div.querySelector('input');
                cb.checked=!cb.checked;
                div.classList.toggle('selected',cb.checked);
                if(cb.checked) selected.push(item.ticker);
                else selected=selected.filter(t=>t!==item.ticker);
                update();
            }};
            grid.appendChild(div);
        }});
        function del(t){{
            const url=new URL(window.parent.location.href);
            url.searchParams.set('del_ticker',t);
            window.parent.location.href=url.toString();
        }}
        </script>"""
        st.components.v1.html(grid_html, height=300, scrolling=True)

    c1, c2 = st.columns(2)
    with c1: prompt_mode_port = st.checkbox("프롬프트만", key="chk_p", value=True)
    with c2: 
        if st.button("🚀 실행"):
            if 'selected' in st.query_params:
                selected_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]
            start_analysis_process(selected_tickers, "MAIN", prompt_mode_port)

    c1, c2 = st.columns(2)
    with c1: 
        if st.button("10-K 분석"): start_analysis_process(selected_tickers, "10K", prompt_mode_port)
        if st.button("8-K 분석"): start_analysis_process(selected_tickers, "8K", prompt_mode_port)
    with c2:
        if st.button("10-Q 분석"): start_analysis_process(selected_tickers, "10Q", prompt_mode_port)

# 모델 선택 및 로그
st.sidebar.markdown('<hr>', unsafe_allow_html=True)
model_opts = ["gemini-1.5-pro", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.0-pro"]
st.session_state['selected_model'] = st.sidebar.selectbox("모델", model_opts)

with st.sidebar.expander("로그"):
    st.text_area("", value="\n".join(st.session_state['log_buffer']), height=150)

# ---------------------------------------------------------
# 6. 실행 로직
# ---------------------------------------------------------
st.title(f"📈 AI Hyper-Analyst V86")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    idx = st.session_state['proc_index']
    stage = st.session_state['proc_stage']
    
    if idx >= len(targets):
        st.success("완료!"); st.session_state['is_analyzing']=False; st.stop()

    curr = targets[idx]
    st.progress((idx*2 + (1 if stage>1 else 0))/(len(targets)*2), text=f"분석 중: {curr}")

    if stage == 1:
        collapse_sidebar(); time.sleep(0.1)
        if step_fetch_data(curr, st.session_state['current_mode']): st.session_state['proc_stage'] = 2
        else: 
            st.session_state['analysis_results'][curr] = {'status':'error', 'report':'데이터 실패', 'df':pd.DataFrame()}
            st.session_state['proc_index'] += 1
        st.rerun()

    elif stage == 2:
        temp = st.session_state['temp_data']
        if st.session_state['prompt_mode']:
            res = {'status':'manual', 'report':'프롬프트 생성됨', 'prompt':temp['prompt'], 'df':temp['df'], 'name':temp['name'], 'mode':st.session_state['current_mode']}
        else:
            try:
                txt, model = generate_with_fallback(temp['prompt'], api_key, st.session_state['selected_model'])
                res = {'status':'success', 'report':sanitize_text(txt), 'df':temp['df'], 'name':temp['name'], 'model':model, 'mode':st.session_state['current_mode'], 'tv_symbol':temp['tv_symbol'], 'is_kr':temp['is_kr']}
            except:
                res = {'status':'error', 'report':'AI 오류', 'df':temp['df'], 'name':temp['name']}
        
        st.session_state['analysis_results'][curr] = res
        st.session_state['proc_index'] += 1
        st.session_state['proc_stage'] = 1
        st.rerun()

# ---------------------------------------------------------
# 7. 결과 표시
# ---------------------------------------------------------
if st.session_state['analysis_results']:
    for t, d in st.session_state['analysis_results'].items():
        with st.expander(f"📊 {d.get('name', t)} 결과", expanded=True):
            if not d['df'].empty:
                if d.get('is_kr', False):
                    fig = go.Figure(data=[go.Candlestick(x=d['df'].index, open=d['df']['Open'], high=d['df']['High'], low=d['df']['Low'], close=d['df']['Close'])])
                    fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    html = f"""<div id="c_{t}" style="height:350px"></div><script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{"autosize":true,"symbol":"{d['tv_symbol']}","interval":"D","timezone":"Asia/Seoul","theme":"light","style":"1","locale":"ko","toolbar_bg":"#f1f3f6","enable_publishing":false,"container_id":"c_{t}"}});</script>"""
                    st.components.v1.html(html, height=360)
            
            if d['status'] == 'manual':
                st.code(d['prompt'])
                st.link_button("Gemini 열기", "https://gemini.google.com/")
            else:
                st.markdown(d['report'])
