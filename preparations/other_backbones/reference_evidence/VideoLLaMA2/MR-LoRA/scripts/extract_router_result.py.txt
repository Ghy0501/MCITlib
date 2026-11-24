import os
import json

def process_cl_results():
    # 1. 基础路径
    base_path = "/your_path/MCITlib_v3/VideoLLaMA2/MR-LoRA/results/CL-VISTA/each_dataset"
    
    # 2. 定义 CL 任务顺序 (Task ID -> Domain Name & Pred Code)
    # 映射结构: Pred Code -> (Domain Folder Name, Task ID)
    cl_mapping = {
        "F": {"domain": "counting", "task_id": 1},
        "G": {"domain": "space",    "task_id": 2},
        "D": {"domain": "traffic",  "task_id": 3},
        "A": {"domain": "movie",    "task_id": 4},
        "C": {"domain": "gui",      "task_id": 5},
        "E": {"domain": "science",  "task_id": 6},
        "H": {"domain": "sports",   "task_id": 7},
        "B": {"domain": "star",     "task_id": 8}
    }

    # 3. 预加载所有需要的专家数据
    # 结构: expert_cache[domain][task_id] = { str(id): result_entry }  <-- 注意这里 Key 存为字符串
    print("正在预加载专家数据...")
    expert_cache = {}

    for code, info in cl_mapping.items():
        domain = info["domain"]
        t_id = info["task_id"]
        
        if domain not in expert_cache:
            expert_cache[domain] = {}
        
        # 初始化该 Task 的缓存
        expert_cache[domain][t_id] = {}

        expert_file = os.path.join(base_path, domain, f"MR-LoRA-task{t_id}", "results.json")
        
        if os.path.exists(expert_file):
            try:
                with open(expert_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # === 关键修改 1: 强制将专家数据的 Key 转为字符串 ===
                    # results.json 的结构通常是 { "id_str": [...], ... }
                    for k, v in data.items():
                        expert_cache[domain][t_id][str(k)] = v
            except Exception as e:
                print(f"[Warning] 读取专家数据失败: {expert_file}, {e}")
        else:
            # print(f"[Warning] 专家文件不存在: {expert_file}") # 可选：减少刷屏
            pass

    print("专家数据加载完成。开始处理 Router...")

    # 4. 遍历 each_dataset 下的所有文件夹
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for folder_name in subfolders:
        current_folder_path = os.path.join(base_path, folder_name)
        
        # 遍历 Task 1 - 8
        for x in range(1, 9):
            router_name = f"MR-LoRA-router-task{x}"
            router_path = os.path.join(current_folder_path, router_name)
            
            if not os.path.exists(router_path):
                continue
            
            print(f"\n>> 处理: {folder_name} / {router_name}")
            
            merge_file = os.path.join(router_path, "merge.jsonl")
            if not os.path.exists(merge_file):
                print(f"   [Skip] merge.jsonl 不存在")
                continue

            # --- 开始处理 ---
            final_results = {}
            scores = []
            
            try:
                with open(merge_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # === 关键修改 2: 获取 ID 并转为字符串 ===
                        raw_id = item.get("id")
                        if raw_id is None: continue
                        q_id_str = str(raw_id) 

                        pred_code = item.get("pred")
                        
                        # 查找映射
                        mapping_info = cl_mapping.get(pred_code)
                        if not mapping_info:
                            continue 
                        
                        target_domain = mapping_info["domain"]
                        target_task_id = mapping_info["task_id"]

                        # 从缓存查找
                        domain_cache = expert_cache.get(target_domain, {})
                        task_cache = domain_cache.get(target_task_id, {})
                        
                        # 使用字符串 Key 进行查找
                        if q_id_str in task_cache:
                            result_entry = task_cache[q_id_str]
                            final_results[q_id_str] = result_entry
                            
                            if isinstance(result_entry, list) and len(result_entry) > 0:
                                val = result_entry[0].get("score", 0)
                                if isinstance(val, (int, float)):
                                    scores.append(val)
                        # else:
                        #     print(f"   [Debug] ID {q_id_str} (Pred: {pred_code}) 在 {target_domain} 中未找到")

            except Exception as e:
                print(f"   [Error] 处理 merge.jsonl 出错: {e}")
                continue

            # --- 保存结果 ---
            out_json = os.path.join(router_path, "results.json")
            try:
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"   [Error] 保存 JSON 失败: {e}")

            if len(scores) > 0:
                avg_original = sum(scores) / len(scores)
                avg_normalized = avg_original * 20
                
                out_txt = os.path.join(router_path, "results.txt")
                try:
                    with open(out_txt, "w", encoding='utf-8') as f:
                        f.write(f"Original Average Score (0-5): {avg_original:.4f}\n")
                        f.write(f"Normalized Average Score (0-100): {avg_normalized:.4f}\n")
                        f.write(f"Number of samples: {len(scores)}\n")
                    print(f"   [Done] Avg: {avg_normalized:.2f} (N={len(scores)})")
                except Exception as e:
                    print(f"   [Error] 保存 TXT 失败: {e}")
            else:
                print(f"   [Warning] 无有效分数，跳过 TXT。")

if __name__ == "__main__":
    process_cl_results()