import os
import json
import time
import requests
import pandas as pd
from datetime import datetime

# ==============================================================================
# 1. 底层获取函数：直接请求东方财富接口 (替代 akshare)
# ==============================================================================

def fetch_eastmoney_news(symbol: str) -> pd.DataFrame:
    """
    底层函数：直接从东方财富接口获取个股新闻
    包含：动态回调名、User-Agent伪装、JSON/JSONP 双重解析机制
    """
    # 动态生成 callback 参数
    callback_name = f"callback_{int(time.time() * 1000)}"
    url = "http://search-api-web.eastmoney.com/search/jsonp"
    
    # 构造请求参数
    params = {
        "cb": callback_name,
        "param": '{"uid":"",'
        + f'"keyword":"{symbol}"'
        + ',"type":["cmsArticle"],"client":"web","clientType":"web","clientVersion":"curr",'
        + '"param":{"cmsArticle":{"searchScope":"default","sort":"default","pageIndex":1,'
        + '"pageSize":100,"preTag":"<em>","postTag":"</em>"}}}',
    }
    
    # 模拟浏览器 Header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # 发送请求
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data_text = r.text
        
        data_json = None
        
        # --- 解析逻辑 ---
        # 尝试方式 A: 纯 JSON
        try:
            data_json = r.json()
        except json.JSONDecodeError:
            # 尝试方式 B: JSONP (剥离 callback(...))
            start_index = data_text.find('(')
            end_index = data_text.rfind(')')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_string = data_text[start_index + 1 : end_index]
                try:
                    data_json = json.loads(json_string)
                except:
                    pass
        
        if not data_json:
            print(f"错误: 无法解析东方财富返回的数据格式 - {symbol}")
            return pd.DataFrame()

        # --- 数据提取与清洗 ---
        if "result" not in data_json or "cmsArticle" not in data_json["result"]:
            return pd.DataFrame()
            
        temp_df = pd.DataFrame(data_json["result"]["cmsArticle"])
        if temp_df.empty:
            return pd.DataFrame()

        # 构造标准字段
        temp_df["url"] = "http://finance.eastmoney.com/a/" + temp_df["code"] + ".html"
        temp_df.rename(columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接"
        }, inplace=True)
        
        temp_df["关键词"] = symbol
        
        # 确保所需列存在
        required_cols = ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]
        for col in required_cols:
            if col not in temp_df.columns:
                temp_df[col] = ""

        final_df = temp_df[required_cols].copy()

        # 清洗 HTML 标签 (<em>...</em>)
        for col in ["新闻标题", "新闻内容"]:
            final_df[col] = (
                final_df[col]
                .astype(str)
                .str.replace(r"\(<em>", "", regex=True)
                .str.replace(r"</em>\)", "", regex=True)
                .str.replace(r"<em>", "", regex=True)
                .str.replace(r"</em>", "", regex=True)
            )
            
        final_df["新闻内容"] = final_df["新闻内容"].str.replace(r"\u3000", "", regex=True).str.replace(r"\r\n", " ", regex=True)
        
        return final_df

    except Exception as e:
        print(f"获取 {symbol} 新闻时发生网络或解析错误: {e}")
        return pd.DataFrame()


# ==============================================================================
# 2. 中间层：处理数据格式 (替代原有的 akshare 调用逻辑)
# ==============================================================================

def get_stock_news_via_akshare(symbol: str, max_news: int = 100) -> list:
    """
    获取股票新闻的中间层处理函数。
    此处不再依赖 akshare 库，而是调用自定义的 fetch_eastmoney_news。
    """
    try:
        # [修改点] 调用自定义函数，而不是 ak.stock_news_em
        news_df = fetch_eastmoney_news(symbol)

        if news_df is None or len(news_df) == 0:
            print(f"未获取到 {symbol} 的新闻数据")
            return []

        # 实际可获取的新闻数量
        available_news_count = len(news_df)
        fetch_count = min(available_news_count, int(max_news * 1.5)) # 多取一点以防过滤

        news_list = []
        
        # 遍历 DataFrame (逻辑与原来保持一致，保证兼容性)
        for _, row in news_df.head(fetch_count).iterrows():
            try:
                # 获取并处理内容
                content = row["新闻内容"] if not pd.isna(row["新闻内容"]) else ""
                if not content:
                    content = row["新闻标题"] # 降级策略

                content = content.strip()
                if len(content) < 10:  # 过滤过短内容
                    continue

                keyword = row["关键词"] if not pd.isna(row["关键词"]) else ""

                news_item = {
                    "title": row["新闻标题"].strip(),
                    "content": content,
                    "publish_time": row["发布时间"],
                    "source": row["文章来源"].strip(),
                    "url": row["新闻链接"].strip(),
                    "keyword": keyword.strip()
                }
                news_list.append(news_item)

            except Exception as e:
                print(f"处理单条新闻数据出错: {e}")
                continue

        # 按时间降序
        news_list.sort(key=lambda x: x["publish_time"], reverse=True)
        
        return news_list[:max_news]

    except Exception as e:
        print(f"获取新闻流程出错: {e}")
        return []


# ==============================================================================
# 3. 上层接口：缓存与文件管理 (保持原有逻辑不变)
# ==============================================================================

def get_stock_news(symbol: str, max_news: int = 10, date: str = None) -> list:
    """
    获取并处理个股新闻 (包含本地缓存机制)
    """
    # 限制最大新闻条数
    max_news = min(max_news, 100)

    # 获取当前日期或使用指定日期
    cache_date = date if date else datetime.now().strftime("%Y-%m-%d")

    # 构建新闻文件路径
    news_dir = os.path.join("data", "stock_news")
    
    # 确保目录存在
    try:
        os.makedirs(news_dir, exist_ok=True)
    except Exception as e:
        print(f"创建目录失败: {e}")
        return []

    # 缓存文件名包含日期信息
    news_file = os.path.join(news_dir, f"{symbol}_news_{cache_date}.json")

    # --- 检查缓存 ---
    cached_news = []
    cache_valid = False

    if os.path.exists(news_file):
        try:
            file_mtime = os.path.getmtime(news_file)
            # 缓存有效期策略
            if date:  # 指定历史日期，缓存永久有效
                cache_valid = True
            else:     # 当天数据，要求文件必须是今天生成的
                cache_date_obj = datetime.fromtimestamp(file_mtime).date()
                today = datetime.now().date()
                cache_valid = cache_date_obj == today

            if cache_valid:
                with open(news_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_news = data.get("news", [])

                    if len(cached_news) >= max_news:
                        # print(f"使用缓存数据: {len(cached_news)}条")
                        return cached_news[:max_news]
            else:
                print(f"缓存已过期，重新获取...")

        except Exception as e:
            print(f"读取缓存失败: {e}")
            cached_news = []

    # --- 获取新数据 ---
    need_more_news = max_news - len(cached_news)
    fetch_count = max(need_more_news, max_news)

    # [调用修改后的中间层函数]
    new_news_list = get_stock_news_via_akshare(symbol, fetch_count)

    # --- 合并与去重 ---
    if cached_news and new_news_list:
        existing_titles = {news['title'] for news in cached_news}
        unique_new_news = [n for n in new_news_list if n['title'] not in existing_titles]
        combined_news = cached_news + unique_new_news
    else:
        combined_news = new_news_list or cached_news

    # 排序
    try:
        combined_news.sort(key=lambda x: x.get("publish_time", ""), reverse=True)
    except:
        pass

    final_news_list = combined_news[:max_news]

    # --- 保存缓存 ---
    if new_news_list or not cache_valid:
        try:
            save_data = {
                "date": cache_date,
                "method": "custom_eastmoney_api", # 标记一下方法变了
                "news": combined_news,
                "last_updated": datetime.now().isoformat()
            }
            with open(news_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存出错: {e}")

    return final_news_list

if __name__ == "__main__":
    TEST_SYMBOL = "600519"  # 贵州茅台
    TEST_MAX_NEWS = 10
    TEST_DATE = None  # None 表示使用当前日期

    print("=" * 40)
    print("      🚀 开始运行股票新闻缓存系统测试 🚀")
    print("=" * 40)
    
    # 测试获取新闻
    print(f"\n测试股票: {TEST_SYMBOL}")
    print(f"最大新闻条数: {TEST_MAX_NEWS}")
    print(f"日期: {TEST_DATE or '当前日期'}")
    
    news = get_stock_news(TEST_SYMBOL, TEST_MAX_NEWS, TEST_DATE)
    
    if news:
        print(f"\n成功获取 {len(news)} 条新闻")
        for item in news[:3]:  # 展示前 3 条
            print(f"\n标题: {item['title']}")
            print(f"来源: {item['source']}")
            print(f"时间: {item['publish_time']}")