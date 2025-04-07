import os
import json
import time
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
import google.generativeai as genai

# 設置頁面配置
st.set_page_config(
    page_title="語言學習問題分析工具",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 載入 .env 中的環境變數
load_dotenv()

# 應用標題和介紹
st.title("🌐 語言學習問題分析工具")
st.markdown("""
這個工具可以幫助您分析網絡上關於各種語言學習的常見問題，並將它們分類為不同類型。
您可以自訂想要研究的語言和相關查詢詞。
""")

# 從 .env 檔案中獲取 API 金鑰和搜尋引擎 ID (sidebar)
with st.sidebar:
    st.header("API 設定")
    
    # 獲取環境變量或讓用戶輸入
    google_api_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    google_cx = st.text_input("Google Custom Search ID", value=os.getenv("GOOGLE_CX", ""), type="password")
    gemini_api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    
    st.markdown("---")
    
    # 語言設定部分
    st.header("語言設定")
    
    # 自訂分類項目
    st.subheader("分類項目設定")
    if "custom_categories" not in st.session_state:
        st.session_state.custom_categories = [
            "語法問題",
            "發音問題",
            "聽力問題",
            "詞彙問題",
            "文化理解問題",
            "口說練習",
            "語言學習技巧"
        ]
    
    # 允許用戶編輯分類項目
    categories_text = st.text_area(
        "分類項目 (每行一個)",
        "\n".join(st.session_state.custom_categories),
        height=200
    )
    
    # 更新分類項目列表
    if st.button("更新分類項目"):
        categories = [cat.strip() for cat in categories_text.split("\n") if cat.strip()]
        if categories:
            st.session_state.custom_categories = categories
            st.success(f"已更新為 {len(categories)} 個分類項目")
        else:
            st.error("至少需要一個分類項目")
    
    st.markdown("---")
    st.header("關於")
    st.info("這個應用使用 Google Search API 和 Gemini API 來搜尋和分析各種語言學習問題。")

# 獲取當前分類項目
def get_categories():
    return st.session_state.custom_categories

# 使用 Google Custom Search API 進行搜尋
def fetch_search_results(query, api_key, cx, num_results=50):
    with st.spinner(f"正在搜尋 '{query}' 相關資訊..."):
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": api_key,
            "cx": cx,
            "num": num_results
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            results = response.json()
            if "items" in results:
                return results["items"]
            else:
                st.error("沒有找到相關搜尋結果")
                return []
        else:
            st.error(f"搜尋API錯誤: {response.status_code}")
            return []

# 從搜尋結果中提取標題、描述和鏈接
def parse_search_results(results):
    search_data = []
    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        search_data.append({"title": title, "snippet": snippet, "link": link})
    return search_data

# 嘗試解析 Gemini API 回傳的 JSON 格式結果
def parse_response(response_text, categories):
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    try:
        result = json.loads(cleaned)
        for item in categories:
            if item not in result:
                result[item] = ""
        return result
    except Exception as e:
        st.error(f"解析 JSON 失敗：{e}")
        st.write("無法解析的內容：", cleaned)
        return {item: "" for item in categories}

# 使用 Gemini API 分類批次處理的討論
def process_batch_dialogue(dialogues, target_language, categories, delimiter="-----"):
    with st.spinner(f"正在使用 Gemini AI 分析 {target_language} 學習問題類型..."):
        prompt = (
            f"你是一位 {target_language} 語言學習問題分類專家，請根據以下分類項目對每條學生討論進行分類：\n"
            + "\n".join(categories) +
            "\n\n請根據討論內容將每個問題分類為相應的項目，並標記每個項目：若該項目涉及則標記為 1，否則留空。"
            " 請對每條討論產生 JSON 格式回覆，並在各條結果間用下列分隔線隔開：\n"
            f"{delimiter}\n"
            "例如：\n"
            "```json\n"
            "{\n  \"語法問題\": \"1\",\n  \"發音問題\": \"\",\n  ...\n}\n"
            f"{delimiter}\n"
            "{{...}}\n```"
        )
        batch_text = f"\n{delimiter}\n".join(dialogues)
        content = prompt + "\n\n" + batch_text

        try:
            # 使用正確的方式創建和調用 Gemini API
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(content)
            response_text = response.text
            
            # 顯示API原始回應進行調試
            with st.expander(f"{target_language} API 回應詳情"):
                st.code(response_text)
                
        except Exception as e:
            st.error(f"Gemini API 呼叫失敗：{e}")
            return [{item: "" for item in categories} for _ in dialogues]
        
        parts = response_text.split(delimiter)
        results = []
        for part in parts:
            part = part.strip()
            if part:
                results.append(parse_response(part, categories))
        
        # 如果結果數量與對話數量不匹配，填充空結果
        while len(results) < len(dialogues):
            results.append({item: "" for item in categories})
            
        # 如果結果數量多於對話數量，截斷多餘結果
        if len(results) > len(dialogues):
            results = results[:len(dialogues)]
            
        return results

# 計算分類項目的統計數據
def calculate_category_counts(results, categories):
    counts = {category: 0 for category in categories}
    for result in results:
        for category in categories:
            if result.get(category) == "1":
                counts[category] += 1
    return counts

# 將分類结果從 ["", "1"] 轉換為可讀形式
def format_category_result(result, categories):
    formatted = {}
    for category in categories:
        formatted[category] = "✓" if result.get(category) == "1" else ""
    return formatted

# 主要界面設計
tab1, tab2, tab3 = st.tabs(["搜尋與分析", "結果詳情", "數據儲存"])

with tab1:
    # 語言學習設定區域
    st.subheader("語言學習設定")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 目標語言輸入
        target_language = st.text_input("目標語言", value="", help="輸入您想要分析的語言，例如：英語、日語、西班牙語等")
    
    with col2:
        # 客群描述輸入
        target_audience = st.text_input("學習者描述 (選填)", value="", 
                                       placeholder="例如：初學者、大學生、商務人士等",
                                       help="描述您想研究的語言學習者群體")
    
    # 搜索設定區域
    st.subheader("搜索設定")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 自動生成搜索關鍵詞，但允許用戶修改
        default_query = f"{target_language}學習問題"
        if target_audience:
            default_query = f"{target_audience}{target_language}學習問題"
            
        # 搜索查詢輸入
        query = st.text_input("搜索關鍵詞", value=default_query, 
                             help="輸入您想要搜索的關鍵詞，系統會根據您的語言和學習者設定自動生成，您也可以手動修改")
    
    with col2:
        # 結果數量選擇
        num_results = st.number_input("搜尋結果數量", min_value=1, max_value=50, value=5, help="選擇要分析的搜尋結果數量")
    
    # 執行按鈕
    if st.button("執行分析", type="primary", use_container_width=True):
        # 獲取最新的分類項目
        categories = get_categories()
        
        if not google_api_key or not google_cx or not gemini_api_key:
            st.error("請在側邊欄中填寫所有 API 密鑰")
        else:
            # 儲存API密鑰到session state以便其他函數使用
            st.session_state.gemini_api_key = gemini_api_key
            
            # 使用搜尋代理來抓取與語言學習相關的討論
            search_results = fetch_search_results(query, google_api_key, google_cx, num_results=int(num_results))
            
            if search_results:
                search_data = parse_search_results(search_results)
                
                # 顯示搜尋結果
                st.subheader("搜尋結果")
                for i, item in enumerate(search_data):
                    with st.expander(f"{i+1}. {item['title']}"):
                        st.markdown(f"**描述**: {item['snippet']}")
                        st.markdown(f"**鏈接**: [{item['link']}]({item['link']})")
                
                # 將搜尋結果轉換為討論內容
                dialogues = [item["snippet"] for item in search_data]
                
                # 顯示要分析的內容
                st.subheader("待分析內容")
                for i, snippet in enumerate(dialogues):
                    st.text(f"{i+1}. {snippet[:100]}...")
                
                # 載入 GEMINI API 客戶端
                try:
                    genai.configure(api_key=gemini_api_key)
                except Exception as e:
                    st.error(f"配置 Gemini API 失敗：{e}")
                    st.stop()
                
                # 處理批次並儲存結果
                batch_results = process_batch_dialogue(dialogues, target_language, categories)
                
                # 顯示解析後的結果
                st.subheader("分類結果")
                for i, result in enumerate(batch_results):
                    with st.expander(f"結果 {i+1}"):
                        st.json(result)
                
                # 計算分類項目的統計數據
                category_counts = calculate_category_counts(batch_results, categories)
                
                # 合併搜尋結果和分類結果
                for i, result in enumerate(batch_results):
                    if i < len(search_data):
                        result.update(search_data[i])  # 將標題、描述和鏈接加到分類結果中
                
                # 將結果保存到 session state
                st.session_state.batch_results = batch_results
                st.session_state.category_counts = category_counts
                st.session_state.current_categories = categories.copy()  # 保存當前使用的分類項目
                st.session_state.current_language = target_language  # 保存當前分析的語言
                
                # 顯示分類結果概要
                st.subheader(f"{target_language}學習問題分類結果概要")
                
                # 創建條形圖
                counts_df = pd.DataFrame([category_counts])
                fig = px.bar(
                    x=list(category_counts.keys()),
                    y=list(category_counts.values()),
                    title=f"{target_language}學習問題類型分布",
                    labels={'x': '問題類型', 'y': '數量'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # 自動切換到第二個標籤
                st.write("分析完成！請查看「結果詳情」標籤以了解更多。")

with tab2:
    if 'batch_results' in st.session_state and 'current_categories' in st.session_state:
        st.subheader(f"{st.session_state.get('current_language', '語言')}學習問題詳細分類結果")
        
        # 創建一個乾淨的數據表
        display_data = []
        categories = st.session_state.current_categories
        
        for i, result in enumerate(st.session_state.batch_results):
            formatted = format_category_result(result, categories)
            if "title" in result:
                formatted["標題"] = result["title"]
                formatted["描述"] = result["snippet"]
                formatted["鏈接"] = f"[網站連結]({result['link']})"
            else:
                formatted["標題"] = f"結果 {i+1}"
                formatted["描述"] = "無數據"
                formatted["鏈接"] = "無鏈接"
            display_data.append(formatted)
        
        # 使用 DataFrame 顯示結果
        df = pd.DataFrame(display_data)
        # 重新排列列順序，將標題、描述和鏈接放在前面
        columns = ["標題", "描述", "鏈接"] + categories
        st.dataframe(df[columns], use_container_width=True)
    else:
        st.info("請先在「搜尋與分析」標籤中執行分析。")

with tab3:
    st.subheader("儲存結果")
    
    if 'batch_results' in st.session_state:
        language_name = st.session_state.get('current_language', '語言')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 將結果儲存到 CSV
            results_df = pd.DataFrame(st.session_state.batch_results)
            csv_results = results_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label=f"下載 {language_name} 詳細結果 (CSV)",
                data=csv_results,
                file_name=f"{language_name}_classified_search_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # 儲存分類統計結果
            stats_df = pd.DataFrame([st.session_state.category_counts])
            csv_stats = stats_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label=f"下載 {language_name} 統計摘要 (CSV)",
                data=csv_stats,
                file_name=f"{language_name}_category_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # 顯示 JSON 格式結果
        with st.expander("查看 JSON 格式結果"):
            st.json(st.session_state.batch_results)
    else:
        st.info("請先在「搜尋與分析」標籤中執行分析。")