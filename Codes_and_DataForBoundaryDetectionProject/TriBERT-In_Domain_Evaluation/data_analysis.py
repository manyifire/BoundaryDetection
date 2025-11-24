import json
from collections import Counter, defaultdict
from pprint import pprint
import os
import tempfile
import re

def clean_docstring(code):
    # 匹配def开头到第一个三引号docstring结束

    if code.startswith("```python"):
        # 去掉前缀，并去除可能紧跟的换行符
            code =  code[len("```python"):].lstrip("\r\n")
    pattern = r"(^\s*def[^\n]*:\s*\n?\s*)([\"']{3}[\s\S]*?[\"']{3}\n?)"
    match = re.match(pattern, code)
    if match:
        # 保留函数定义行，去除docstring
        return match.group(1) + code[match.end():]
    else:
        # 没有docstring则返回原始代码
        return code

def clean(s):
    if s.startswith("```python"):
    # 去掉前缀，并去除可能紧跟的换行符
        s =  s[len("```python"):].lstrip("\r\n")
    return ''.join(line.strip() for line in s.split('\n') if line.strip())

def match_code(full_code, matched_code):
    '''
    return pos: position of matched code
    '''
    
    if clean(full_code) == clean(matched_code):
        return None
    
    full_lines = clean_docstring(clean(full_code))
    matched_lines = clean(matched_code)
    full_lines_l = [line.strip() for line in full_code.split('\n') if line.strip()]
    matched_lines_l = [line.strip() for line in matched_code.split('\n') if line.strip()]
    n, m = len(full_lines_l), len(matched_lines_l)
    
    for i in range(n - m + 1):
        if full_lines_l[i:i + m] == matched_lines_l:
            return i

    return None

def code2lines(code):
    full_lines_l = [line.strip() for line in code.split('\n') if line.strip()]
    return full_lines_l
    

def match_mhmcode(full_code, matched_code):
    if clean(full_code) == clean(matched_code):
        return None
    
    full_lines = clean_docstring(clean(full_code))
    matched_lines = clean(matched_code)
    full_lines = [line.strip() for line in full_code.splitlines() if line.strip()]
    matched_lines = [line.strip() for line in matched_code.splitlines() if line.strip()]
    n, m = len(full_lines), len(matched_lines)
    
    for i in range(n - m + 1):
        if full_lines[i:i + m] == matched_lines:
            return i,i+m

    return None,None


def reassign_code_id(jsonl_path):
    count = 0
    
    new_id = 0
    with open(jsonl_path, 'r', encoding='utf-8') as rf, \
         open('hybridCodeData_cleanded.jsonl', 'w', encoding='utf-8') as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            boundary_ix = []

            obj['hybrid_code'] = clean_docstring(obj['hybrid_code'])
            lines = code2lines(obj['hybrid_code'])
            if obj.get("prompt_pattern") == "H_M_H_M":
                continue
            if obj['prompt_pattern'] == "M_H":
                boundary_ix.append(match_code(obj['hybrid_code'], obj['human_part']))
                if boundary_ix[0] == 1:
                    continue
                obj['boundary_ix'] = boundary_ix
                obj["human_part"] = lines[obj["boundary_ix"][0]:]
                obj["machine_part"] = lines[0:obj["boundary_ix"][0]]

            if obj['prompt_pattern'] == "H_M":
                obj["human_part"] = lines[0:obj["boundary_ix"][0]]
                obj["machine_part"] = lines[obj["boundary_ix"][0]:]
                
            if obj['prompt_pattern'] == "H_M_H":
                if obj['machine_part'] == "" or obj['machine_part'] == " " or obj['machine_part'] == "\n":
                    continue
                obj["human_part"] = lines[0:obj["boundary_ix"][0]] + lines[obj["boundary_ix"][1]:]
                obj["machine_part"] = lines[obj["boundary_ix"][0]:obj["boundary_ix"][1]]

            if obj['prompt_pattern'] == "M_H_M":
                if obj['machine_part'] == "" or obj['machine_part'] == " " or obj['machine_part'] == "\n":
                    continue
                a,b = match_mhmcode(obj['hybrid_code'], obj['human_part'])
                if a is None or b is None:
                    continue

                boundary_ix.append(a)
                boundary_ix.append(b)

                obj['boundary_ix'] = boundary_ix  
                obj["machine_part"] = lines[0:obj["boundary_ix"][0]] + lines[obj["boundary_ix"][1]:]
                obj["human_part"] = lines[obj["boundary_ix"][0]:obj["boundary_ix"][1]]

            obj["labeled_statements"] = [(s, "human") for s in obj["human_part"]] + \
                                            [(s, "machine") for s in obj["machine_part"]]
            
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    # 替换原文件
    # os.replace(tmp_path, jsonl_path)
    print(f"✅ {count}")

if __name__ == "__main__":
    import argparse

    jsonl_file = "usedForTest.jsonl"  # 默认文件名

    reassign_code_id(jsonl_file)

