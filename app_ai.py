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
if 'new_ticker_input' not in st.session_state: st.session_state['new_ticker_input'] = ""

# 체크박스 초기화
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
            add_log(f"💾 [SAVE] 저장 완료.")
        except Exception as e:
            add_log(f"❌ [SAVE] 실패: {e}")

def add_ticker_logic():
    raw_input = st.session_state.get('new_ticker_input', '')
    if raw_input:
        add_log(f"➕ [ADD] 티커 추가: '{raw_input}'")
        tickers = [t.strip().upper() for t in raw_input.split(',')]
        df = st.session_state['portfolio_df']
        existing_tickers = df['ticker'].values
        
        new_rows = []
        for ticker in tickers:
            if ticker and ticker not in existing_tickers:
                # 간단한 정보 확인 (이름만)
                try: 
                    t_info = yf.Ticker(ticker).info
                    name = t_info.get('shortName') or t_info.get('longName') or ticker
                except: 
                    name = ticker
                new_rows.append({'ticker': ticker, 'name': name})
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            st.session_state['portfolio_df'] = df
            save_state_to_csv()
    st.session_state['new_ticker_input'] = ""

load_data_to_state()

# 삭제 로직
if 'del_ticker' in st.query_params:
    del_ticker = st.query_params['del_ticker']
    if 'portfolio_df' in st.session_state:
        df = st.session_state['portfolio_df']
        df = df[df['ticker'] != del_ticker]
        st.session_state['portfolio_df'] = df
        save_state_to_csv()
        if f"chk_{del_ticker}" in st.session_state: del st.session_state[f"chk_{del_ticker}"]
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

def get_realtime_news(ticker, name):
    add_log(f"📰 [NEWS] 뉴스 검색: {ticker}")
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    # 1. Yahoo Finance RSS
    if not is_kr:
        try:
            url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            session = get_robust_session()
            resp = session.get(url, timeout=5)
            root = ET.fromstring(resp.content)
            for item in root.findall('./channel/item')[:5]:
                title = item.find('title').text
                link = item.find('link').text
                pub = item.find('pubDate').text
                try: dt = parser.parse(pub).strftime("%m-%d %H:%M")
                except: dt = "최신"
                news_items.append(f"- [Yahoo] {title} ({dt})")
        except: pass

    # 2. Google News RSS
    try:
        q = f'"{name}"' if is_kr else f'{ticker} stock'
        q_enc = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={q_enc}&hl=ko&gl=KR&ceid=KR:ko"
        session = get_robust_session()
        resp = session.get(url, timeout=5)
        root = ET.fromstring(resp.content)
        for item in root.findall('./channel/item')[:5]:
            title = item.find('title').text
            try: dt = parser.parse(item.find('pubDate').text).strftime("%m-%d %H:%M")
            except: dt = "최신"
            news_items.append(f"- [Google] {title} ({dt})")
    except: pass
    
    return "\n".join(news_items) if news_items else "관련 뉴스 없음"

def get_financial_metrics(info):
    """info 객체를 직접 받아서 처리"""
    if not info: return {}
    try:
        def get_fmt(key): val = info.get(key); return f"{val:,.2f}" if isinstance(val, (int, float)) else "N/A"
        return {
            "Free Cash Flow": get_fmt('freeCashflow'), "Current Ratio": get_fmt('currentRatio'),
            "Debt/Equity": get_fmt('debtToEquity'), "ROE": get_fmt('returnOnEquity'),
            "Revenue": get_fmt('totalRevenue'), "Net Income": get_fmt('netIncome')
        }
    except: return {}

def sanitize_text(text):
    text = text.replace('$', '\$'); text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
    return text

def collapse_sidebar():
    js = """<script>var closeBtn = window.parent.document.querySelector('[data-testid="stSidebarExpandedControl"]');if (closeBtn) {closeBtn.click();}</script>"""
    st.components.v1.html(js, height=0, width=0)

def start_analysis_process(targets, mode, is_prompt_only):
    add_log(f"▶️ [START] 분석 시작: {targets}")
    st.session_state['is_analyzing'] = True
    st.session_state['targets_to_run'] = targets
    st.session_state['current_mode'] = mode
    st.session_state['prompt_mode'] = is_prompt_only
    st.session_state['analysis_results'] = {} 
    st.session_state['proc_index'] = 0
    st.session_state['proc_stage'] = 1 

def generate_with_fallback(prompt, api_key, start_model):
    genai.configure(api_key=api_key)
    chain = [start_model, "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.0-pro"]
    
    for m in chain:
        try:
            model = genai.GenerativeModel(m)
            resp = model.generate_content(prompt)
            return resp.text, m
        except: continue
    return "분석 실패 (API Error)", "Error"

def handle_search_click(mode, is_prompt):
    raw = st.session_state.get("s_input", "")
    if raw: start_analysis_process([t.strip() for t in raw.split(',')], mode, is_prompt)
    else: st.warning("티커 입력 필요")

def step_fetch_data(ticker, mode):
    add_log(f"📦 [STEP 1] 데이터 수집: {ticker}")
    
    # 기본값 설정
    stock_name = ticker
    sector = "Unknown (AI가 문맥으로 판단할 것)"
    industry = "Unknown (AI가 문맥으로 판단할 것)"
    
    clean_code = re.sub(r'[^0-9]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker

    try:
        # 1. 정보 가져오기 (타임아웃 10초로 증가)
        info = run_with_timeout(_fetch_info, args=(ticker,), timeout=10)
        
        if info:
            # 이름/섹터/산업 추출 (Safety Get)
            stock_name = info.get('shortName') or info.get('longName') or ticker
            sector = info.get('sector', "정보 없음 (AI 추론 필요)")
            industry = info.get('industry', "정보 없음 (AI 추론 필요)")
            add_log(f"   -> 정보 획득: {stock_name} / {sector} / {industry}")
        else:
            add_log("   ⚠️ yfinance 정보 획득 실패 (Timeout/Null). AI 추론으로 대체.")

        # 2. 주가 데이터
        period = st.session_state.get('selected_period_str', '1y')
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=10)
        
        data_summary = "No Data"
        if df is not None and not df.empty:
            curr = df['Close'].iloc[-1]
            data_summary = f"Current: {curr:.2f}\n[Trend]\n{df.tail(60).to_string()}"
        else:
            df = pd.DataFrame()

        # 3. 재무 및 뉴스
        fin_str = str(get_financial_metrics(info)) if info else "N/A"
        news_text = "N/A"
        if st.session_state.get('use_news', True):
            news_text = get_realtime_news(ticker, stock_name)

        # 4. 프롬프트 구성
        focus_list = [opt for opt in opt_targets if st.session_state.get(f"focus_{opt}", True)]
        focus = ", ".join(focus_list)
        viewpoint = st.session_state.get('selected_viewpoint', 'General')
        analysis_depth = st.session_state.get('analysis_depth', "2. 표준 브리핑")
        
        # [핵심] 시나리오 분석 지시사항 (확률 및 근거 필수)
        level_instruction = ""
        if "5." in analysis_depth:
            level_instruction = """
            \n[매우 중요: 시나리오 분석 필수 양식]
            사용자가 '시나리오 분석'을 요청했습니다. 결론 부근에 반드시 아래 3가지 시나리오를 작성하고, **각 시나리오의 실현 확률(%)과 그 이유**를 명시하십시오.
            
            1. 🚀 **Best Case (낙관적 시나리오)**: 
               - 예상 확률: O%
               - 논리적 근거: (구체적 호재 및 성장 동력)
               - 목표 주가 범위: $00 ~ $00
               
            2. ⚖️ **Base Case (기본 시나리오)**: 
               - 예상 확률: O%
               - 논리적 근거: (시장 컨센서스 및 현황 유지)
               - 목표 주가 범위: $00 ~ $00
               
            3. 🌧️ **Worst Case (비관적 시나리오)**: 
               - 예상 확률: O%
               - 논리적 근거: (리스크 요인 현실화 시)
               - 목표 주가 범위: $00 ~ $00
            
            * 세 시나리오 확률의 합은 반드시 100%가 되도록 조정하십시오.
            """

        # [핵심] 투자 성향 지시사항
        if "투자성향별 포트폴리오 적정보유비중" in focus:
            level_instruction += """
            \n[특별 지시: 투자성향별 비중 제안]
            보고서 마지막에 다음 3가지 투자 성향별 권장 비중(%)을 제시하십시오:
            1. 🦁 공격적 투자자 (Aggressive)
            2. ⚖️ 중립적 투자자 (Moderate)
            3. 🛡️ 보수적 투자자 (Conservative)
            """

        # [핵심] 성장주 vs 가치주 판단 로직
        growth_value_logic = """
        [Step 0: 기업 성향 판단]
        먼저 이 기업이 '성장주'인지 '가치주'인지 명확히 규정하고, 그 성향에 맞춰 분석을 전개하십시오.
        - 성장주라면: 매출 성장률, Cash Flow, 흑자 전환 가능성 집중.
        - 가치주라면: 시장 점유율, 배당 안정성, 이익률 집중.
        """
        
        # 공통 프롬프트 헤더
        base_info = f"""
        [대상 정보]
        - 티커: {ticker}
        - 기업명: {stock_name}
        - 섹터: {sector}
        - 산업: {industry}
        * 만약 위 섹터/산업 정보가 'Unknown'이거나 부정확하다면, 당신의 지식 베이스를 활용하여 올바른 정보를 채워서 분석하십시오.
        """

        korean_enforcement = "\n**[중요] 모든 답변은 반드시 전문적이고 자연스러운 '한국어'로 작성하십시오.**"

        # 모드별 프롬프트 분기
        if mode == "10K":
            prompt = f"""
            [역할] 월가 수석 애널리스트 (10-K 분석)
            {base_info}
            [자료] 최신 SEC 10-K (Annual Report) 기반
            
            [지시사항]
            기업의 연간 보고서를 바탕으로 펀더멘털과 장기 비전을 분석하십시오.
            1. 비즈니스 모델 및 산업 내 위치
            2. 경영진의 미래 전망 (Outlook) 및 자신감 톤(Tone)
            3. 핵심 리스크 요인 (Risk Factors)
            4. 재무 상태 건전성 (부채, 현금흐름)
            
            {korean_enforcement}
            """
        elif mode == "10Q":
            prompt = f"""
            [역할] 실적 트렌드 분석가 (10-Q 분석)
            {base_info}
            [자료] 최신 SEC 10-Q (Quarterly Report) 기반
            
            [지시사항]
            직전 분기 대비 변화(Trend)와 모멘텀에 집중하십시오.
            1. 매출/EPS의 YoY, QoQ 성장률 및 컨센서스 비교
            2. 가이던스(Guidance) 상향/하향 여부 및 원인
            3. 부문별 성과 및 특이사항
            
            {korean_enforcement}
            """
        elif mode == "8K":
            prompt = f"""
            [역할] 속보 및 이벤트 분석가 (8-K 분석)
            {base_info}
            [자료] 최신 SEC 8-K (Current Report) 기반
            
            [지시사항]
            최근 발생한 공시 이벤트의 핵심 내용과 주가 영향을 분석하십시오.
            1. 공시 사유 (실적, 계약, 인사 등)
            2. 호재/악재 판단 및 단기 주가 영향
            3. 투자자 대응 전략
            
            {korean_enforcement}
            """
        else: # MAIN
            prompt = f"""
            [역할] AI Hyper-Analyst (종합 분석)
            {base_info}
            [분석 모드] {mode}
            [중점 항목] {focus}
            [투자 관점] {viewpoint}
            [분석 깊이] {analysis_depth}
            
            {growth_value_logic}
            
            {level_instruction}
            
            [데이터 요약]
            {data_summary}
            
            [재무 지표]
            {fin_str}
            
            [뉴스 헤드라인]
            {news_text}
            
            [최종 지시]
            위 데이터를 종합하여 논리적이고 통찰력 있는 투자 보고서를 작성하십시오.
            결론에는 [매수 / 매도 / 관망] 중 하나의 의견을 명확히 제시하십시오.
            {korean_enforcement}
            """
        
        st.session_state['temp_data'] = {
            'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': []
        }
        add_log(f"✅ [STEP 1] 완료. Prompt 길이: {len(prompt)}")
        return True

    except Exception as e:
        add_log(f"❌ [ERROR] Step 1 실패: {e}")
        st.error(f"데이터 수집 오류: {e}")
        return False

# ---------------------------------------------------------
# 5. 사이드바 UI
# ---------------------------------------------------------
st.sidebar.subheader("🎯 분석 옵션")

viewpoint_mapping = {"단기": "3mo", "스윙": "6mo", "중기": "2y", "장기": "5y"}
sel_vp = st.sidebar.select_slider("", options=list(viewpoint_mapping.keys()), value="중기", label_visibility="collapsed")
st.session_state['selected_period_str'] = viewpoint_mapping[sel_vp]
st.session_state['selected_viewpoint'] = sel_vp

# 레벨 설정 (시나리오 포함)
analysis_levels = ["1.요약", "2.표준", "3.심층", "4.전문가", "5.시나리오(확률포함)"]
analysis_depth = st.sidebar.select_slider("", options=analysis_levels, value=analysis_levels[-1], label_visibility="collapsed")
st.session_state['analysis_depth'] = analysis_depth

st.session_state['use_news'] = st.sidebar.toggle("뉴스 반영", value=True)

def toggle_focus_all():
    val = st.session_state['focus_all']
    for opt in opt_targets: st.session_state[f"focus_{opt}"] = val

with st.sidebar.expander("☑️ 중점 분석 항목", expanded=False):
    st.checkbox("전체 선택", key="focus_all", on_change=toggle_focus_all)
    for opt in opt_targets: st.checkbox(opt, key=f"focus_{opt}")

# API KEY
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.sidebar.error("Secrets에 API Key 설정 필요")

# 탭 UI
tab1, tab2 = st.sidebar.tabs(["⚡ 검색", "⭐ 포트폴리오"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.text_input("티커 (예: SOUN, 005930.KS)", key="s_input")
    c1, c2 = st.columns(2)
    chk_p = c1.checkbox("프롬프트만", key="chk_p_s", value=True)
    if c2.button("🔍 분석", type="primary"): handle_search_click("MAIN", chk_p)
    
    st.markdown("##### 📑 공시")
    b1, b2, b3 = st.columns(3)
    if b1.button("10-K"): handle_search_click("10K", chk_p)
    if b2.button("10-Q"): handle_search_click("10Q", chk_p)
    if b3.button("8-K"): handle_search_click("8K", chk_p)

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.7, 0.3])
    c1.text_input("추가", key="new_ticker_input", label_visibility="collapsed")
    c2.button("➕", on_click=add_ticker_logic)

    fav_df = st.session_state.get('portfolio_df', pd.DataFrame())
    selected_tickers = []
    
    if not fav_df.empty:
        # 쿼리 파라미터 연동
        if 'selected' in st.query_params:
            selected_tickers = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]

        # 그리드 HTML 생성 (간소화)
        import json
        t_data = [{'ticker': r['ticker'], 'name': str(r['name'])} for i, r in fav_df.iterrows()]
        init_sel = json.dumps(selected_tickers)
        t_json = json.dumps(t_data)
        
        # HTML/JS (기존 로직 유지하되 간략화)
        html_code = f"""
        <style>
        .pf-item {{padding:5px; border:1px solid #ddd; margin-bottom:5px; border-radius:5px; cursor:pointer; display:flex; align-items:center;}}
        .pf-item.sel {{background:#e6f3ff; border-color:#2196F3;}}
        .pf-info {{flex:1;}} .pf-del {{color:#999; border:none; background:none; cursor:pointer;}}
        </style>
        <div id="grid"></div>
        <script>
        const data={t_json}; let sel={init_sel};
        function render(){{
            const g=document.getElementById('grid'); g.innerHTML='';
            data.forEach(d=>{{
                const isSel=sel.includes(d.ticker);
                const el=document.createElement('div'); el.className='pf-item'+(isSel?' sel':'');
                el.innerHTML=`<div class="pf-info"><b>${{d.ticker}}</b><br><small>${{d.name}}</small></div><button class="pf-del">×</button>`;
                el.onclick=(e)=>{{ 
                    if(e.target.className==='pf-del'){{ 
                        window.parent.location.href='?del_ticker='+d.ticker; return; 
                    }}
                    if(isSel) sel=sel.filter(x=>x!==d.ticker); else sel.push(d.ticker);
                    const p = new URLSearchParams(window.parent.location.search);
                    if(sel.length) p.set('selected', sel.join(',')); else p.delete('selected');
                    window.parent.history.replaceState(null,'','?'+p.toString());
                    render();
                }};
                g.appendChild(el);
            }});
        }}
        render();
        </script>
        """
        st.components.v1.html(html_code, height=300, scrolling=True)
        
    chk_p_port = st.checkbox("프롬프트만", key="chk_p_p", value=True)
    if st.button("🚀 종합 분석 시작", type="primary"):
        if 'selected' in st.query_params:
            targets = [t.strip() for t in st.query_params['selected'].split(',') if t.strip()]
            start_analysis_process(targets, "MAIN", chk_p_port)

# 모델 선택
st.sidebar.markdown('---')
sel_model = st.sidebar.selectbox("모델", ["gemini-1.5-pro", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash"], label_visibility="collapsed")
st.session_state['selected_model'] = sel_model

# 로그창
with st.sidebar.expander("📜 로그", expanded=False):
    st.text_area("", "\n".join(st.session_state['log_buffer']), height=200)

# ---------------------------------------------------------
# 6. 실행 컨트롤러
# ---------------------------------------------------------
st.title("📈 AI Hyper-Analyst V90")

if st.session_state['is_analyzing']:
    targets = st.session_state['targets_to_run']
    idx = st.session_state['proc_index']
    stage = st.session_state['proc_stage']
    
    if idx >= len(targets):
        st.success("분석 완료!"); st.session_state['is_analyzing'] = False; st.rerun()

    curr = targets[idx]
    st.progress((idx * 2 + (1 if stage > 1 else 0)) / (len(targets) * 2), text=f"분석 중: {curr}")

    if stage == 1: # 데이터 수집
        collapse_sidebar()
        with st.spinner(f"{curr} 데이터 수집 중..."):
            if step_fetch_data(curr, st.session_state['current_mode']):
                st.session_state['proc_stage'] = 2
            else:
                st.session_state['analysis_results'][curr] = {'status': 'error', 'report': '데이터 실패'}
                st.session_state['proc_index'] += 1
            st.rerun()

    elif stage == 2: # AI 생성
        temp = st.session_state['temp_data']
        if st.session_state['prompt_mode']:
            res = {'status': 'manual', 'prompt': temp['prompt'], 'name': temp['name'], 'df': temp['df'], 'mode': "Manual"}
        else:
            with st.spinner("보고서 작성 중..."):
                txt, m = generate_with_fallback(temp['prompt'], api_key, sel_model)
                res = {'status': 'success', 'report': sanitize_text(txt), 'name': temp['name'], 'df': temp['df'], 'model': m, 'mode': st.session_state['current_mode']}
        
        st.session_state['analysis_results'][curr] = res
        st.session_state['proc_index'] += 1
        st.session_state['proc_stage'] = 1
        st.rerun()

# 결과 출력
if st.session_state['analysis_results']:
    st.write("---")
    for t, d in st.session_state['analysis_results'].items():
        with st.expander(f"📊 {d.get('name', t)} 결과", expanded=True):
            if d.get('status') == 'manual':
                st.code(d['prompt'])
                st.link_button("Gemini로 이동", "https://gemini.google.com")
            elif d.get('status') == 'success':
                st.markdown(d['report'])
                if not d['df'].empty: st.line_chart(d['df']['Close'])
