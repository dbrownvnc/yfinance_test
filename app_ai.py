def step_fetch_data(ticker, mode):
    add_log(f"==========================================")
    add_log(f"📦 [STEP 1] 데이터 수집 시작: {ticker} ({mode})")
    
    stock_name = ticker 
    clean_code = re.sub(r'[^0-9]', '', ticker)
    is_kr = (".KS" in ticker or ".KQ" in ticker or (ticker.isdigit() and len(ticker)==6))
    tv_symbol = f"KRX:{clean_code}" if is_kr else ticker

    try:
        stock = yf.Ticker(ticker)
        
        # [NEW] 기업 기본 정보(Overview) 수집
        info = run_with_timeout(_fetch_info, args=(ticker,), timeout=5)
        if not info: info = {}
        
        # Session State 이름 우선 사용
        if 'portfolio_df' in st.session_state:
            p_df = st.session_state['portfolio_df']
            row = p_df[p_df['ticker'] == ticker]
            if not row.empty:
                stock_name = row.iloc[0]['name']
            else:
                fetched_name = info.get('shortName') or info.get('longName')
                if fetched_name: stock_name = fetched_name
        else:
            fetched_name = info.get('shortName') or info.get('longName')
            if fetched_name: stock_name = fetched_name
            
        # Overview 변수 추출
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        country = info.get('country', 'N/A')
        employees = info.get('fullTimeEmployees', 'N/A')
        if employees != 'N/A': employees = f"{employees:,}명"
        
        mkt_cap_raw = info.get('marketCap', 'N/A')
        cap_category = "N/A"
        if isinstance(mkt_cap_raw, (int, float)):
             mkt_cap = f"${mkt_cap_raw / 1_000_000_000:,.2f}B"
             if mkt_cap_raw >= 10_000_000_000: cap_category = "Large Cap"
             elif mkt_cap_raw >= 2_000_000_000: cap_category = "Mid Cap"
             else: cap_category = "Small Cap"
        else: mkt_cap = "N/A"

        add_log(f"   - 기본 정보 확보: {stock_name} / {sector} / {industry} / {cap_category}")

        # ------------------------------------------------------------------
        # [수정] 주가 데이터 로직 강화 (24시간 반영)
        # ------------------------------------------------------------------
        period = st.session_state.get('selected_period_str', '1y')
        
        # 1. 차트용 기본 데이터 (긴 기간)
        df = run_with_timeout(_fetch_history, args=(ticker, period), timeout=10)
        
        if df is None: df = pd.DataFrame(); add_log("   ⚠️ 차트 데이터 타임아웃/실패")
        else: add_log(f"   ✅ 차트 데이터 수신: {len(df)} rows")

        # 2. [핵심] "Why" 모드용 실시간/시간외(Pre/Post) 데이터 정밀 조회
        realtime_note = ""
        current_price = 0
        
        if mode == "WHY" and not is_kr:
            # 미국 주식의 경우 1분 단위 Pre/Post 데이터 조회
            add_log("   🕒 [Real-time] 미국 주식 Pre/Post Market 데이터 조회 시도...")
            try:
                # 최근 5일치 중 1분봉, prepost=True로 장외거래 포함
                df_realtime = stock.history(period="5d", interval="1m", prepost=True)
                if not df_realtime.empty:
                    last_row = df_realtime.iloc[-1]
                    current_price = last_row['Close']
                    last_time = df_realtime.index[-1]
                    
                    # 정규장 종가(previousClose)와 비교
                    prev_close = info.get('previousClose', current_price)
                    delta = current_price - prev_close
                    delta_pct = (delta / prev_close) * 100
                    
                    realtime_note = f"""
                    [실시간/시간외 시세 정보]
                    - 기준 시간: {last_time} (현지시간 추정)
                    - 현재가(Pre/Post 포함): {current_price:.2f}
                    - 전일 종가 대비: {delta_pct:.2f}% ({delta:+.2f})
                    - 참고: 이 데이터는 정규장 마감 후(After-hours) 또는 개장 전(Pre-market) 거래가 포함된 최신 가격입니다.
                    """
                    add_log(f"   ✅ [Real-time] 시간외 가격 확보: {current_price} ({delta_pct:.2f}%)")
                else:
                    realtime_note = "(실시간 데이터 조회 실패, 정규장 데이터만 사용됨)"
            except Exception as e:
                add_log(f"   ⚠️ [Real-time] Error: {e}")
        elif not df.empty:
            # 한국 주식 혹은 일반 모드
            current_price = df['Close'].iloc[-1]
            realtime_note = f"현재가(종가): {current_price:,.0f}" if is_kr else f"Current: {current_price:.2f}"

        # ------------------------------------------------------------------
        # 데이터 요약 텍스트 생성
        # ------------------------------------------------------------------
        if not df.empty:
            high_val = df['High'].max(); low_val = df['Low'].min()
            stats_str = f"Range(Period): {low_val:.2f} ~ {high_val:.2f}"
            display_df = df.tail(60); recent_days = df.tail(5)
            # realtime_note를 요약 맨 위에 붙여줌
            data_summary = f"{realtime_note}\n\n[Period Trend]\n{display_df.to_string()}\n[Recent 5 Days]\n{recent_days.to_string()}"
        else:
            data_summary = f"Chart Data Not Available.\n{realtime_note}"

        fin_str = "N/A"; news_text = "N/A"
        
        if mode not in ["10K", "10Q", "8K"]:
            try: 
                fm = get_financial_metrics(ticker); fin_str = str(fm) if fm else "N/A"
            except: pass
            
            if st.session_state.get('use_news', True):
                try:
                    # Why 모드일 때는 검색 쿼리를 좀 더 구체적으로 변경 가능
                    # 여기서는 기존 로직을 타되, 프롬프트에서 강력하게 해석하도록 유도
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
        
        level_instruction = ""
        scenario_block = ""
        if "5." in analysis_depth:
            level_instruction = "이 분석은 '시나리오 모드'입니다. 미래 불확실성을 고려하여 확률적 접근이 필수적입니다."
            scenario_block = """
            4. **[시나리오별 확률 및 근거 (Scenario Analysis)]**
               - **Bull (낙관) / Base (기본) / Bear (비관)** 3가지 시나리오를 설정하십시오.
               - 각 시나리오별 **예상 주가 밴드**와 **실현 확률(%)**을 명시적으로 제시하십시오.
               - 왜 그러한 확률이 배정되었는지에 대한 **논리적/정량적 근거**를 상세히 설명하십시오.
            """

        add_log(f"📝 프롬프트 조립 시작 (Mode: {mode})")
        
        # [프롬프트 정의 구간 - 기존과 동일하되 Why 모드만 수정된 data_summary 반영]
        if mode == "10K":
            prompt = f"""
            [역할] 월가 수석 애널리스트 (펀더멘털 & 장기 투자 전문가)
            [대상] {ticker} (공식 기업명: {stock_name})
            [자료] 최신 SEC 10-K 보고서 (Annual Report)
            [지시사항] 위 종목의 최신 SEC 10-K 보고서를 기반으로 기업의 기초 체력과 장기 비전을 심층 분석해 주세요.
            (중략 - 기존 10K 프롬프트 내용 유지)
            **모든 답변은 반드시 한글로 작성해 주십시오.**
            """
        elif mode == "10Q":
            prompt = f"""
            [역할] 실적 모멘텀 및 트렌드 분석가
            [대상] {ticker} (공식 기업명: {stock_name})
            [자료] 최신 SEC 10-Q 보고서 (Quarterly Report)
            [지시사항] 위 종목의 최신 SEC 10-Q 보고서를 기반으로 직전 분기 대비 변화(Trend)에 집중하여 분석하세요.
            (중략 - 기존 10Q 프롬프트 내용 유지)
            **모든 답변은 반드시 한글로 작성해 주십시오.**
            """
        elif mode == "8K":
            prompt = f"""
            [역할] 속보 뉴스 데스크 / 이벤트 드리븐 트레이더
            [대상] {ticker} (공식 기업명: {stock_name})
            [자료] 최신 SEC 8-K 보고서 (Current Report)
            [지시사항] 위 종목의 최신 SEC 8-K 보고서를 분석하여, 발생한 특정 사건(Event)의 내용과 주가 영향을 분석하세요.
            (중략 - 기존 8K 프롬프트 내용 유지)
            **모든 답변은 반드시 한글로 작성해 주십시오.**
            """
        elif mode == "WHY":
            # [수정] Why 프롬프트 강화: 시간외 정보를 반영하도록 지시
            prompt = f"""
            [역할] 주식 시황 및 급등락 원인 분석가 (24시간 시장 모니터링)
            [대상] {ticker} (공식 기업명: {stock_name})
            [자료] 실시간 뉴스(Earnings 포함) 및 24시간 주가 데이터(Pre/Post Market)
            
            [지시사항]
            사용자는 **"이 주식이 지금(장중 혹은 시간외) 왜 움직이는지"** 알고 싶어 합니다.
            특히 정규장 종료 후 실적 발표(Earnings)나 중요 공시로 인한 **시간외 급등락** 여부를 면밀히 살피십시오.
            
            [데이터 요약 - 시간외 시세 포함]
            {data_summary}
            
            [수집된 최신 뉴스]
            {news_text}
            
            [분석 요구사항]
            1. **현재 상황 팩트 체크**: 
               - 현재 주가 움직임이 '정규장' 흐름인지 '시간외(After-hours/Pre-market)' 흐름인지 구분하여 명시하십시오.
               - 예: "정규장은 1% 상승 마감했으나, 실적 발표 후 시간외 거래에서 10% 급락 중입니다."
               
            2. **핵심 원인 (3줄 요약)**: 
               - 뉴스(특히 실적, 가이던스, 공시)를 근거로 변동의 핵심 원인을 3가지로 요약하십시오.
               - 실적 발표 직후라면 **매출/EPS가 예상치(Consensus)를 상회했는지 하회했는지** 뉴스에서 찾아 언급하십시오.
               
            3. **투자자 팁**: 
               - 이 뉴스나 변동이 내일 정규장 시초가에 어떤 영향을 미칠지 짧게 전망하십시오.
            
            **[출력 형식]**
            - 서론 없이 바로 분석 내용을 마크다운으로 작성하십시오.
            - **반드시 한글로 작성하십시오.**
            """
        else:
            # [기존 MAIN 모드 프롬프트]
            prompt = f"""
            [역할] 월스트리트 수석 애널리스트
            [대상] {ticker} (공식 기업명: {stock_name})
            [모드] {mode}
            [중점 분석] {focus}
            [투자 관점] {viewpoint}
            [분석 레벨] {analysis_depth}
            {level_instruction}
            
            [데이터 요약]
            {data_summary}
            
            [재무 지표]
            {fin_str}
            
            [관련 뉴스]
            {news_text}
            
            [분석 지침]
            (중략 - 기존 MAIN 프롬프트 내용 유지)
            0. **[기업 기본 정보 (Company Overview)]**
               - 보고서의 **가장 첫 부분**에 다음 데이터를 사용하여 **마크다운 표**를 작성하십시오.
               - 표 헤더: | 항목 | 내용 |
               - 표 데이터:
                 | 정식 기업명 | {stock_name} |
                 | 티커(심볼) | {ticker} |
                 | 섹터 | {sector} |
                 | 산업 | {industry} |
                 | 시가총액 | {mkt_cap} |
                 | 기업 규모 | {cap_category} |

            1. **[성장주/가치주 정의 및 핵심 지표 분석]** ...
            2. **[사용자 선택 중점 분석 항목 상세]** ...
            3. **[투자성향별 포트폴리오 비중 분석]** ...
            {scenario_block}
            
            [결론]
            반드시 [매수 / 매도 / 관망] 중 하나의 명확한 투자 의견을 제시하십시오.
            **모든 답변은 반드시 한글로 작성하십시오.**
            """
        
        st.session_state['temp_data'] = {
            'name': stock_name, 'tv_symbol': tv_symbol, 'is_kr': is_kr,
            'df': df, 'prompt': prompt, 'news': []
        }
        add_log(f"✅ [STEP 1] 데이터 준비 완료 (Prompt Length: {len(prompt)})")
        return True

    except Exception as e:
        add_log(f"❌ [FATAL] Step 1 Error: {str(e)}")
        st.error(f"Data Step Error: {e}")
        return False
