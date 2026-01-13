import json
import time
import csv
import os
from DrissionPage import ChromiumOptions, ChromiumPage
import random

CRAWL_STATUS = {
    "is_running": False,
    "progress": 0,
    "total_comments": 0,
    "current_video": "",
    "error": None
}

def ensure_json(obj):
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8", errors="ignore")
    if isinstance(obj, str):
        obj = obj.strip()
        try:
            return json.loads(obj)
        except Exception:
            return None
    return None

def extract_replies(data: dict):
    if not data or data.get("code") != 0:
        return [], None
    d = data.get("data") or {}
    replies = d.get("replies") or []
    cursor = d.get("cursor") or {}
    pr = (cursor.get("pagination_reply") or {})
    next_offset = pr.get("next_offset") or cursor.get("next")
    return replies, next_offset

def parse_comment_item(it: dict):
    content = it.get("content") or {}
    return {"rpid": it.get("rpid"), "content": content.get("message", "")}

def start_crawl(video_url, profile_dir="./bili_profile", csv_path="bili_comments.csv", headless=True):
    global CRAWL_STATUS
    CRAWL_STATUS["is_running"] = True
    CRAWL_STATUS["progress"] = 0
    CRAWL_STATUS["total_comments"] = 0
    CRAWL_STATUS["current_video"] = video_url
    CRAWL_STATUS["error"] = None
    
    try:
        co = ChromiumOptions()
        co.set_user_data_path(profile_dir)
        if headless:
            co.mute(True)
        else:
            co.mute(False)
            co.headless(False)
        
        page = ChromiumPage(co)
        page.get(video_url)
        time.sleep(2)
        page.listen.start("/x/v2/reply/wbi/main")
        page.scroll.down(1200)
        time.sleep(2)
        
        # 每次爬取都覆盖文件，而不是追加
        f = open(csv_path, "w", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(f, fieldnames=["num", "content"])
        writer.writeheader()
        
        num_counter = 0  # 重置计数器
        
        seen = set()
        total = 0
        print(f"开始爬取评论（从编号 {num_counter + 1} 开始）")
        page.scroll.down(5000)
        
        STUCK_LIMIT = 15
        WAIT_SEC_MIN = 0.5
        WAIT_SEC_MAX = 1.0
        
        stuck = 0
        
        while True:
            wait_s = random.uniform(WAIT_SEC_MIN, WAIT_SEC_MAX) * 0.7
            
            before = page.run_js("return document.documentElement.scrollTop || document.body.scrollTop;")
            page.run_js("window.scrollTo(0, document.body.scrollHeight);")
            
            time.sleep(wait_s)
            after = page.run_js("return document.documentElement.scrollTop || document.body.scrollTop;")
            
            if after == before:
                stuck += 1
               
                if stuck >= STUCK_LIMIT:
                    break
                continue
            
            packet = None
            for _ in range(3):
                try:
                    packet = page.listen.wait(timeout=3)  
                    if packet:
                        break
                except Exception:
                    pass
            
            if not packet:
                stuck += 1
                print(f"暂未获取到新数据 stuck={stuck}/{STUCK_LIMIT}，已获取评论{total}条")
                if stuck >= STUCK_LIMIT:
                    break
                continue
            
            data = ensure_json(getattr(packet.response, "body", None))
            data = ensure_json(data)
            
            replies, _ = extract_replies(data)
            new_cnt = 0
            
            for it in replies:
                item = parse_comment_item(it)
                rpid = item.get("rpid")
                if not rpid or rpid in seen:
                    continue
                seen.add(rpid)
                
                num_counter += 1
                writer.writerow({"num": num_counter, "content": item["content"]})
                new_cnt += 1
                total += 1
            
            if new_cnt:
                f.flush()
                stuck = 0

                CRAWL_STATUS["progress"] = min(95, int((total / max(total + 1, 100)) * 100))
                CRAWL_STATUS["total_comments"] = total
            else:
                stuck += 1
            

            if total % 10 == 0 or new_cnt > 0:
                print(f"新增 {new_cnt} 条；已获取 {total} 条评论；当前编号到 {num_counter}")
            
            if stuck >= STUCK_LIMIT:
                break
        

        CRAWL_STATUS["progress"] = 100
        CRAWL_STATUS["is_running"] = False
        print(f"爬取完成！成功获取 {total} 条评论")
        
        return total
        
    except Exception as e:
        CRAWL_STATUS["error"] = str(e)
        CRAWL_STATUS["is_running"] = False
        CRAWL_STATUS["progress"] = 0
        raise e
    finally:
        
        if 'f' in locals():
            f.close()

def get_crawl_status():
    return CRAWL_STATUS.copy()



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        headless = True
        if len(sys.argv) > 2:
            if sys.argv[2] == '--headed':
                headless = False
            elif sys.argv[2] == '--headless':
                headless = True
        
        start_crawl(video_url, headless=headless)
    else:
        print('请提供视频URL作为参数进行爬取')