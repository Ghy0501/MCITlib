"""
Router merge.jsonl 与专家 Result.json 对齐：每条用 merge 的 question_id（或 id / image）
在专家侧先按 question_id 精确匹配，再回退到 image 路径键（兼容未写 question_id 的旧结果）。
"""
import os
import json

# 与当前仓库结果目录一致；若使用 MLLM-DCL，请改此处路径
BASE_PATH = "/your_path/MCITlib_v3/LLaVA/MR-LoRA/results/DCL/each_dataset"

# 专家结果文件名候选（优先 Result.json）
EXPERT_RESULT_NAMES = ("Result.json", "result.json", "results.json")


def image_match_key(path: str) -> str:
    """用路径最后两段对齐 router 的 question_id 与专家 Result 里的 image（前缀可能不同）。"""
    if not path:
        return ""
    path = path.replace("\\", "/").strip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def resolve_expert_path(domain: str, task_id: int) -> str | None:
    d = os.path.join(BASE_PATH, domain, f"MR-LoRA-task{task_id}")
    for name in EXPERT_RESULT_NAMES:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def load_expert_entries(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    # 整文件一个 JSON 数组
    if raw[0] == "[":
        return json.loads(raw)
    # JSONL：每行一条
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def build_expert_lookup(entries: list) -> tuple[dict, dict]:
    """(按 image, 按 question_id) -> 整条专家记录。question_id 统一为 str 键。"""
    img_cache = {}
    qid_cache = {}
    for entry in entries:
        qid = entry.get("question_id")
        if qid is not None:
            qid_cache[str(qid).strip()] = entry
        img = entry.get("image")
        if img:
            s = str(img)
            k = image_match_key(s)
            img_cache[k] = entry
            bn = os.path.basename(s.replace("\\", "/"))
            if bn and bn not in img_cache:
                img_cache[bn] = entry
    return img_cache, qid_cache


def get_score(entry) -> float | None:
    if isinstance(entry, list):
        if not entry:
            return None
        entry = entry[0]
    if not isinstance(entry, dict):
        return None
    val = entry.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def merge_router_pred(item: dict) -> str:
    p = item.get("pred")
    if p is not None and str(p).strip():
        return str(p).strip().upper()[:1]
    t = item.get("text")
    if t is not None and str(t).strip():
        return str(t).strip().upper()[:1]
    return ""


def merge_sample_id(item: dict) -> str | None:
    rid = item.get("question_id")
    if rid is not None:
        return str(rid)
    rid = item.get("id")
    if rid is not None:
        return str(rid)
    rid = item.get("image")
    if rid is not None:
        return str(rid)
    return None


def lookup_expert(
    img_cache: dict,
    qid_cache: dict,
    router_ref: str,
) -> dict | None:
    """优先用 question_id 与专家 Result 对齐；否则按 image 路径规则（与旧数据兼容）。"""
    if router_ref:
        sref = str(router_ref).strip()
        if sref in qid_cache:
            return qid_cache[sref]
        # JSON 数字 id 与 str 差异：141822 vs "141822" 已由 merge_sample_id 统一为 str
        k = image_match_key(sref)
        if k in img_cache:
            return img_cache[k]
        bn = os.path.basename(sref.replace("\\", "/"))
        if bn in img_cache:
            return img_cache[bn]
    return None


def process_cl_results():
    cl_mapping = {
        "E": {"domain": "RS", "task_id": 1},
        "C": {"domain": "Med", "task_id": 2},
        "D": {"domain": "AD", "task_id": 3},
        "B": {"domain": "Sci", "task_id": 4},
        "A": {"domain": "Fin", "task_id": 5},
    }

    print("正在预加载专家数据...")
    expert_cache = {}

    for code, info in cl_mapping.items():
        domain = info["domain"]
        t_id = info["task_id"]
        if domain not in expert_cache:
            expert_cache[domain] = {}
        expert_cache[domain][t_id] = ({}, {})

        expert_file = resolve_expert_path(domain, t_id)
        if not expert_file:
            continue
        try:
            entries = load_expert_entries(expert_file)
            expert_cache[domain][t_id] = build_expert_lookup(entries)
        except Exception as e:
            print(f"[Warning] 读取专家数据失败: {expert_file}, {e}")

    print("专家数据加载完成。开始处理 Router...")

    if not os.path.isdir(BASE_PATH):
        print(f"[Error] 基础路径不存在: {BASE_PATH}")
        return

    subfolders = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]

    for folder_name in subfolders:
        current_folder_path = os.path.join(BASE_PATH, folder_name)

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

            final_results = {}
            scores = []

            try:
                with open(merge_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        sample_key = merge_sample_id(item)
                        if not sample_key:
                            continue

                        pred_code = merge_router_pred(item)
                        mapping_info = cl_mapping.get(pred_code)
                        if not mapping_info:
                            continue

                        target_domain = mapping_info["domain"]
                        target_task_id = mapping_info["task_id"]

                        img_cache, qid_cache = expert_cache.get(target_domain, {}).get(
                            target_task_id, ({}, {})
                        )
                        result_entry = lookup_expert(img_cache, qid_cache, sample_key)
                        if result_entry is None:
                            continue

                        final_results[sample_key] = result_entry
                        val = get_score(result_entry)
                        if val is not None:
                            scores.append(val)

            except Exception as e:
                print(f"   [Error] 处理 merge.jsonl 出错: {e}")
                continue

            out_json = os.path.join(router_path, "results.json")
            try:
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"   [Error] 保存 JSON 失败: {e}")

            if len(scores) > 0:
                avg_01 = sum(scores) / len(scores)
                avg_100 = avg_01 * 100.0

                out_txt = os.path.join(router_path, "results.txt")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(f"Average Score (0-1): {avg_01:.4f}\n")
                        f.write(f"Normalized Average Score (0-100): {avg_100:.4f}\n")
                        f.write(f"Number of samples: {len(scores)}\n")
                    print(f"   [Done] Avg 0-100: {avg_100:.2f} (N={len(scores)})")
                except Exception as e:
                    print(f"   [Error] 保存 TXT 失败: {e}")
            else:
                print(f"   [Warning] 无有效分数，跳过 TXT。")


if __name__ == "__main__":
    process_cl_results()
