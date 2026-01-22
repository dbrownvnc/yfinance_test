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

# ... (이전 코드와 동일한 1. 설정 및 초기화, 2. 데이터 관리 함수 부분) ...

# ---------------------------------------------------------
# 3. 기타 유틸리티 함수
# ---------------------------------------------------------
# ... (get_robust_session, run_with_timeout, _fetch_history, _fetch_info, get_stock_name 등 기존 함수 유지) ...

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
    """
    [수정됨] 티커가 아닌 '공식 기업명'을 기준으로 뉴스를 검색합니다.
    기존의 Yahoo Finance(티커 기반) 로직을 건너뛰고 Google News(이름 기반)를 사용합니다.
    """
    add_log(f"📰 [NEWS] 뉴스 검색 시작: {name} (Ticker: {ticker} 무시/참고용)")
    
    # 검색 쿼리 생성: 무조건 기업명(name)을 사용합니다.
    # 정확도를 위해 따옴표("")로 감싸서 구문 검색을 시도합니다.
    search_query = f'"{name}"'
    
    add_log(f"   Trying Google News RSS with query: {search_query}")
    
    try:
        q_encoded = urllib.parse.quote(search_query)
        # hl=ko&gl=KR: 한국어/한국 설정 (필요시 en/US로 변경 가능하나, 앱 설정상 ko 유지)
        # 만약 영문 기업의 영문 뉴스를 원하신다면 hl=en&gl=US 등으로 변경 고려 가능
        url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ko&gl=KR&ceid=KR:ko"
        
        google_news = fetch_rss_realtime(url, limit=7)
        for n in google_news: 
            n['source'] = "Google News"
            
        if google_news:
            return google_news
        else:
            add_log("   ⚠️ Google News 검색 결과 없음.")
            return []
            
    except Exception as e:
        add_log(f"   ❌ 뉴스 검색 중 오류 발생: {str(e)}")
        return []

# ... (이하 get_financial_metrics, sanitize_text 및 나머지 UI/로직 코드 동일) ...
