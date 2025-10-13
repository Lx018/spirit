import os
from volcenginesdkarkruntime import Ark
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Ark客户端，从环境变量中读取您的API Key
client = Ark(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key="47514fc4-9e54-4846-b93a-3702a3adec2c",
)

# Non-streaming:
import json
import sys

# Argument parsing
arg_start = None
arg_end = None
output_filename = "chatData/all.json"
CHAT_WINDOW = 400
try:
    with open("raw_chat_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print("Error: raw_chat_data.json not found or is invalid.")
    sys.exit(1)

if len(sys.argv) == 3:
    try:
        arg_start = int(sys.argv[1])
        num_data = int(sys.argv[2])
        if num_data <= 0:
            print("Error: Number of data points must be a positive integer.")
            sys.exit(1)
        arg_end = arg_start + num_data - 1
        output_filename = f"chatData{CHAT_WINDOW}/{arg_start}_{arg_end}.json"
    except ValueError:
        print("Invalid arguments. Please provide an integer start index and the number of data points.")
        sys.exit(1)
else:
    arg_start = 0
    arg_end = 10
num_data = arg_end - arg_start + 1
# Resume logic
start_idx = 0
try:
    with open(output_filename, 'r', encoding='utf-8') as f:
        final = json.load(f)
        if final and isinstance(final, list) and final[-1].get('data_idx') is not None:
            last_idx = max(item.get('data_idx', -1) for item in final)
            start_idx = last_idx + 1
            print(f"Resuming from data index {start_idx}...")
        else:
            final = []
            print("Starting from beginning...")
except (FileNotFoundError, json.JSONDecodeError):
    final = []
    print("Starting from beginning...")

# Determine effective start index
effective_start = max(arg_start, start_idx)
from threading import Thread

remaining_threads = 0
def process(client, q, cur_mem, i, final):
    global remaining_threads
    remaining_threads += 1
    try:
        completion = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model="doubao-seed-1-6-flash-250828",
            messages=[
                {"role": "user", "content": "query:"+q},
                {"role": "user", "content": "memorys:"+"\n".join(cur_mem)},
                {"role": "system", "content": '''你是大脑中的记忆单元调度中枢，负责处理哪些记忆会被唤醒。你需要根据近期聊天记录、新的消息，来筛选出哪些过去的聊天记忆是对回答新的消息有帮助、或对角色应对此时此刻下一秒的情景是有用的，并将消息返回到activated_memory。

                现在请根据以下规则进行筛选：
                1.与新的消息直接或间接有关的信息
                2.有些信息是在当下重要的，有些虽然重要但是回答时候并不一定需要参考因此不用选择
                3.站在角色的视角上，考虑当下情境哪些内容会被回忆起来
                4.激活的记忆总数不要超过10个，优先选择最有帮助的
                5.若没有任何需要被激活的记忆，输出[]空的即可

                输出格式：
                原文1\n原文2\n原文3
                例子:
                [2025年9月2日]<19:21>我告诉u89feAW寄件地址为公司楼下快递驿站，name写"小狮子"，并警告乱寄或骗人就直接拉黑。</19:27>\n[2025年9月14日]<15:18>用户u89feAW咨询创建原创角色（OC）的建议，我强调需收费并给出了加点反差萌设定的意见。</15:22>
                '''},
            ],
        )
        result = completion.choices[0].message.content
        final.append({"query": q, "memory": cur_mem, "result": result, "data_idx": i})
    except Exception as e:
        print(f"Error processing query '{q}': {e}")
    
    remaining_threads -= 1
    print(f"Finished processing query: '{i}', remaining threads: {remaining_threads}")

# Main loop
total_items = len(data)
threads = []
import queue
waitlist_threads = queue.Queue()
for i, item in enumerate(data):
    if i < effective_start or i > arg_end:
        continue

    print(f"Processing item {i+1}/{total_items}...")
    cur_mem = item["memory"]
    cur_chat = item["short_term_chat_history"]
    next_conv = item["next_conversation"]
    queries = []# + [next_conv]

    if len(cur_chat) <= CHAT_WINDOW:
        if cur_chat:
            queries.append(cur_chat)
    else:
        # start = 0
        # while start + 100 < len(cur_chat):
        #     queries.append(cur_chat[start:start+100])
        #     start += 50
        queries.append(cur_chat[-CHAT_WINDOW:])

    for q in queries:
        thread = Thread(target=process, args=(client, q, cur_mem, i, final), daemon=True)
        waitlist_threads.put(thread)
import time

NUM_THREAD = 100
while not waitlist_threads.empty():
    if remaining_threads > NUM_THREAD:
        time.sleep(1)
        continue

    thread = waitlist_threads.get()
    print("putting")
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)