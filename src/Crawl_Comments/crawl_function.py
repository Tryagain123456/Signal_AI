# 爬虫实现

import sys
import argparse
import os
import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger



# =========================================================
#                    爬取相关功能函数定义
# =========================================================

# 爬取功能路径
mediacrawler_path = Path(__file__).resolve().parent / "MediaCrawler"


# 1. 爬取基础配置设置
def create_base_config(platform: str, keywords: List[str], crawler_type: str = "search", max_notes: int = 50) -> bool:
    """
    创建MediaCrawler的基础配置

    Args:
        platform: 平台名称
        keywords: 关键词列表
        crawler_type: 爬取类型
        max_notes: 最大爬取数量

    Returns:
        是否配置成功
    """
    try:
        save_data_option = "json"
        base_config_path = mediacrawler_path / "config" / "base_config.py"
        # 将关键词列表转换为逗号分隔的字符串
        keywords_str = ",".join(keywords)
        # 读取原始配置文件
        with open(base_config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修改关键配置项
        lines = content.split('\n')
        new_lines = []

        for line in lines:
            if line.startswith('PLATFORM = '):
                new_lines.append(f'PLATFORM = "{platform}"')  # 平台，xhs | dy | ks | bili | wb | tieba | zhihu
            elif line.startswith('KEYWORDS = '):
                new_lines.append(f'KEYWORDS = "{keywords_str}"')  # 关键词搜索配置，以英文逗号分隔
            elif line.startswith('CRAWLER_TYPE = '):
                new_lines.append(
                    f'CRAWLER_TYPE = "{crawler_type}"')  # 爬取类型，search(关键词搜索) | detail(帖子详情)| creator(创作者主页数据)
            elif line.startswith('SAVE_DATA_OPTION = '):
                new_lines.append(f'SAVE_DATA_OPTION = "{save_data_option}"')
            elif line.startswith('CRAWLER_MAX_NOTES_COUNT = '):
                new_lines.append(f'CRAWLER_MAX_NOTES_COUNT = {max_notes}')
            elif line.startswith('ENABLE_GET_COMMENTS = '):
                new_lines.append('ENABLE_GET_COMMENTS = True')
            elif line.startswith('CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = '):
                new_lines.append('CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 20')
            elif line.startswith('HEADLESS = '):
                new_lines.append('HEADLESS = True')
            else:
                new_lines.append(line)
        # 写入新配置
        with open(base_config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        # logger.info(
        #     f"已配置 {platform} 平台，爬取类型: {crawler_type}，关键词数量: {len(keywords)}，最大爬取数量: {max_notes}")
        return True

    except Exception as e:
        logger.exception(f"创建基础配置失败: {e}")
        return False


# 2. 爬取函数 -> 基于关键词的多平台爬取

# 基于关键词单平台爬虫功能实现
def run_crawler(platform: str, keywords: List[str], login_type: str = "qrcode", max_notes: int = 5) -> Dict:
    """
    Args:
        platform: 平台名称
        keywords: 关键词列表
        login_type: 登录方式
        max_notes: 最大爬取数量
    Returns:
        爬取结果统计
    """

    # 爬取开始日志
    start_message = f"\n开始爬取平台: {platform} | 关键词: {keywords[:5]}{'...' if len(keywords) > 5 else ''} | 最大爬取数量: {max_notes} | 登录方式: {login_type}"
    logger.info(start_message)

    start_time = datetime.now()

    try:
        save_data_option = "json"
        # 创建基础配置
        if not create_base_config(platform, keywords, "search", max_notes):
            return {"success": False, "error": "基础配置创建失败"}

        cmd = [
            sys.executable, "main.py",
            "--platform", platform,
            "--lt", login_type,
            "--type", "search",
            "--save_data_option", save_data_option
        ]

        # logger.info(f"执行命令: {' '.join(cmd)}")

        # 切换到 MediaCrawler 目录执行爬虫
        result = subprocess.run(
            cmd,
            cwd=mediacrawler_path,
            capture_output=True,
            text=True,
            encoding='utf-8',  # 避免编码错误
            timeout=3600  # 60分钟超时
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 爬取统计信息
        crawl_stats = {
            "platform": platform,
            "keywords": keywords,
            "keywords_count": len(keywords),
            "max_notes": max_notes,
            "login_type": login_type,
            "duration_seconds": round(duration, 1),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "return_code": result.returncode,
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

        if result.returncode == 0:
            logger.info(f"✅ {platform}_{keywords} 爬取完成，耗时: {duration:.1f}秒")
        else:
            logger.error(f"❌ {platform} 爬取失败，返回码: {result.returncode}")
            logger.error(f"错误详情: {result.stderr}")

        return crawl_stats

    except subprocess.TimeoutExpired:
        error_msg = f"{platform} 爬取超时（超过60分钟）"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "platform": platform,
            "keywords_count": len(keywords)
        }
    except Exception as e:
        error_msg = f"{platform} 爬取异常: {str(e)}"
        logger.exception(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "platform": platform,
            "keywords_count": len(keywords)
        }


# =========================================================
#                    爬取内容解析
# =========================================================

# 提取爬取内容文件（帖子+评论）
def context_extract(platform: str, keywords: List[str]):
    def get_current_date():
        return datetime.now().strftime("%Y-%m-%d")
    date = get_current_date()
    data_contents_path = mediacrawler_path / "data" / f"{platform}" / "json" / f"search_contents_{date}_{keywords[0]}.json"
    data_comments_path = mediacrawler_path / "data" / f"{platform}" / "json" / f"search_comments_{date}_{keywords[0]}.json"

    # 读取帖子信息
    with open(data_contents_path, "r", encoding="utf-8") as f:
        data_contents = json.load(f)
    content_summary = ""
    for index, item in enumerate(data_contents, 1):
        title = item.get("title", "")
        desc = item.get("desc", "")
        tieba_name = item.get("tieba_name", "")
        content_summary += f"## 帖子{index}\n"
        content_summary += f"### 来源贴吧名：\n{tieba_name}\n"
        content_summary += f"### 题目：\n{title}\n"
        content_summary += f"### 概述：\n{desc}\n\n"

    # 读取评论信息
    with open(data_comments_path, "r", encoding="utf-8") as f:
        data_comments = json.load(f)
    comments_summary = ""
    for index, item in enumerate(data_comments, 1):
        content = item.get("content", "")
        tieba_name = item.get("tieba_name", "")
        comments_summary += f"## 评论{index}\n"
        comments_summary += f"### 来源贴吧名：\n{tieba_name}\n"
        comments_summary += f"### 评论内容：\n{content}\n\n"

    summary = f"# {keywords[0]}相关的帖子合集：\n # 帖子内容：\n {content_summary} \n # 帖子评论：\n {comments_summary}"

    return summary


def all_crawl_process(platform: str, keywords: List[str]):
    # 先检查是否有缓存，若没有则爬取，否则直接读取缓存
    def get_current_date():
        return datetime.now().strftime("%Y-%m-%d")
    date = get_current_date()
    data_contents_path = mediacrawler_path / "data" / f"{platform}" / "json" / f"search_contents_{date}_{keywords[0]}.json"
    data_comments_path = mediacrawler_path / "data" / f"{platform}" / "json" / f"search_comments_{date}_{keywords[0]}.json"
    if not (os.path.exists(data_contents_path) and os.path.exists(data_comments_path)) :
        run_crawler(platform, keywords)
    result = context_extract(platform, keywords)
    return result


if __name__ == "__main__":
    platform = "tieba"
    keywords = ["特斯拉股票"]
    result = all_crawl_process(platform, keywords)


    print(result)