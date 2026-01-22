def get_realtime_news(ticker, name):
    """
    뉴스 검색 - 정식 기업명 기반으로 검색하여 혼동 방지
    예: MS 티커 → "Morgan Stanley" 뉴스 검색 (Microsoft 아님)
    """
    add_log(f"📰 [NEWS] 뉴스 검색 시작: {ticker} ({name})")
    news_items = []
    is_kr = bool(re.search(r'\.KS|\.KQ|[0-9]{6}', ticker))
    
    # [핵심 로직 1] 뉴스 관련성 검증 함수
    def is_relevant_news(news_title, news_summary, company_name, ticker_symbol):
        """뉴스가 해당 기업과 관련 있는지 확인"""
        title_lower = news_title.lower() if news_title else ""
        summary_lower = news_summary.lower() if news_summary else ""
        combined_text = f"{title_lower} {summary_lower}"
        
        # 기업명 전처리 (Inc, Corp 등 제거하여 핵심 단어만 추출)
        name_clean = company_name.lower()
        for suffix in [' inc.', ' inc', ' corp.', ' corp', ' ltd.', ' ltd', ' llc', ' co.', ' co', 
                       ' corporation', ' incorporated', ' limited', ' group', ' holdings']:
            name_clean = name_clean.replace(suffix, '')
        name_clean = name_clean.strip()
        
        # 이름이 아주 짧은 경우(3글자 이하)가 아니면 기업명 포함 여부 체크
        if len(name_clean) > 2 and name_clean in combined_text:
            return True
            
        # 티커가 명확하게 단독으로 쓰였는지 체크 (단어 경계 확인)
        # 예: "MS" 단어는 찾되 "MSFT"나 "Systems"의 s는 제외
        ticker_clean = ticker_symbol.replace('.KS', '').replace('.KQ', '').upper()
        if re.search(rf'\b{re.escape(ticker_clean)}\b', news_title or ""):
            return True
            
        return False
    
    # [핵심 로직 2] 혼동되기 쉬운 티커에 대한 제외 키워드 설정
    exclude_keywords = {
        'MS': ['microsoft', 'windows', 'azure', 'xbox', 'office 365', 'satya nadella', 'bill gates'], # MS(모건스탠리) vs Microsoft
        'GM': [], # General Motors
        'F': [],  # Ford
        'T': [],  # AT&T
        'C': [],  # Citigroup
        'O': [],  # Realty Income
        'V': [],  # Visa
    }
    
    def should_exclude(news_title, news_summary, ticker_symbol):
        """혼동될 수 있는 뉴스(제외 키워드 포함) 걸러내기"""
        if ticker_symbol.upper() not in exclude_keywords:
            return False
        
        combined = f"{news_title} {news_summary}".lower()
        for keyword in exclude_keywords.get(ticker_symbol.upper(), []):
            if keyword in combined:
                add_log(f"      ❌ 제외됨 (혼동 키워드 '{keyword}' 발견): {news_title[:40]}...")
                return True
        return False
    
    # 1. Yahoo Finance RSS (티커 기반 - 가장 빠름)
    if not is_kr:
        try:
            add_log(f"   Trying Yahoo Finance RSS for {ticker}...")
            rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            yahoo_rss_items = fetch_rss_realtime(rss_url, limit=10)
            
            filtered_items = []
            for item in yahoo_rss_items:
                # 혼동 키워드 체크 (예: MS 검색했는데 Microsoft 기사면 제외)
                if should_exclude(item['title'], item.get('summary', ''), ticker):
                    continue
                item['source'] = "Yahoo Finance"
                filtered_items.append(item)
            
            if filtered_items:
                add_log(f"   -> Yahoo RSS 필터링 후 {len(filtered_items)}건 확보")
                return filtered_items[:7]
        except Exception as e:
            add_log(f"   ⚠️ Yahoo RSS Fail: {e}")

    # 2. yfinance 라이브러리 (티커 기반)
    if not is_kr and not news_items:
        try:
            add_log(f"   Trying yfinance library for {ticker}...")
            yf_obj = yf.Ticker(ticker)
            yf_news = yf_obj.news
            if yf_news:
                filtered_items = []
                for item in yf_news:
                    title = item.get('title')
                    link = item.get('link')
                    summary = item.get('summary', '') 
                    
                    if not title or not link: continue
                    if should_exclude(title, summary, ticker): continue
                        
                    pub_time = item.get('providerPublishTime', 0)
                    try: date_str = datetime.datetime.fromtimestamp(pub_time).strftime("%m-%d %H:%M")
                    except: date_str = "최신"
                    
                    filtered_items.append({
                        'title': title, 'link': link, 'date_str': date_str, 
                        'source': "Yahoo Finance", 'summary': summary
                    })
                
                if filtered_items:
                    add_log(f"   -> yfinance 필터링 후 {len(filtered_items)}건 확보")
                    return filtered_items[:7]
        except Exception as e:
            add_log(f"   ⚠️ yfinance Fail: {e}")

    # 3. Google News RSS (정식 기업명 검색 - 티커 혼동의 최후 보루)
    # [핵심 변경] 티커 대신 받아온 'stock_name'(정식 기업명)으로 검색합니다.
    if is_kr:
        search_query = f'"{name}"' # 한국 주식은 이름으로 검색
    else:
        # 미국 주식: 이름이 있으면 이름으로 검색, 없으면 티커+stock
        if name and name != ticker and len(name) > 3:
            # 정식 기업명에서 불필요한 접미사 제거 후 검색 (검색 정확도 향상)
            search_name = name
            for suffix in [' Inc.', ' Inc', ' Corp.', ' Corp', ' Ltd.', ' Ltd', ' LLC', ' Co.', ' Co']:
                search_name = search_name.replace(suffix, '')
            search_query = f'"{search_name.strip()}" stock' # 따옴표로 정확히 일치하는 것 검색
            add_log(f"   📌 정식 기업명으로 정밀 검색: '{search_query}' (티커 혼동 방지)")
        else:
            search_query = f'{ticker} stock'
            add_log(f"   ⚠️ 기업명 불분명, 티커로 검색: '{search_query}'")
    
    add_log(f"   Trying Google News RSS with query: {search_query}")
    try:
        q_encoded = urllib.parse.quote(search_query)
        url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ko&gl=KR&ceid=KR:ko"
        google_news = fetch_rss_realtime(url, limit=10)
        
        # Google News 결과 필터링
        filtered_news = []
        if google_news:
            for n in google_news:
                n['source'] = "Google News"
                # 역시 혼동 키워드 체크
                if should_exclude(n['title'], n.get('summary', ''), ticker):
                    continue
                # 관련성 체크 (검색어가 이름이었으면 이름이 포함되어야 함)
                if is_relevant_news(n['title'], n.get('summary', ''), name, ticker):
                    filtered_news.append(n)
                else:
                    # 너무 엄격하게 걸러서 뉴스가 0개가 되는 것을 방지하기 위해
                    # 검색어(이름)가 제목에 없어도 요약에 있거나 하면 통과
                    if name.lower() in (n.get('summary','') or '').lower():
                        filtered_news.append(n)

            add_log(f"   -> Google News 필터링: {len(google_news)}건 → {len(filtered_news)}건")
            
            if filtered_news:
                return filtered_news[:7]
            elif google_news:
                # 필터링 결과가 아예 없으면 원본 중 상위 3건만 반환 (완전 공백 방지)
                add_log(f"   ⚠️ 필터링 결과 0건. 관련성 낮을 수 있으나 원본 반환.")
                return google_news[:3]
    except Exception as e:
        add_log(f"   ⚠️ Google News Fail: {e}")
    
    return news_items
