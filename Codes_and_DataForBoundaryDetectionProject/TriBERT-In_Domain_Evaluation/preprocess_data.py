# import pandas as pd

# import json

# # 读取 jsonl 文件并处理每一行
# data = []
# data1 = []
# data2 = []
# with open('/data/wangmanyi/MGT_Localization/hybridCodeData_mixed.jsonl', 'r') as f:
#     for line in f:
#         obj = json.loads(line)
#         data.append(obj)
#         if(len(obj['boundary_ix'])==1):
#             data1.append(obj)
#         if(len(obj['boundary_ix'])==2):
#             data2.append(obj)

# # 转为 DataFrame
# df = pd.DataFrame(data)

# # 读取jsonl文件
# # df = pd.read_json('usedForTest.jsonl', lines=True)

# # 打乱顺序
# df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# n = len(df_shuffled)
# n_test = int(n * 0.15)
# n_valid = int(n * 0.15)

# # 修改 set_ix 字段
# df_shuffled.loc[:n_test-1, 'set_ix'] = 'test'
# df_shuffled.loc[n_test:n_test+n_valid-1, 'set_ix'] = 'valid'
# df_shuffled.loc[n_test+n_valid:, 'set_ix'] = 'train'

# # 保存结果
# df_shuffled.to_excel('hybridCodeData_mixed.xlsx', index=False)

# # with open('hybridCodeData_1boundary.jsonl', 'w', encoding='utf-8') as wf:
# #     wf.write(json.dumps(data1, ensure_ascii=False) + "\n")
# # with open('hybridCodeData_2boundary.jsonl', 'w', encoding='utf-8') as wf:
# #     wf.write(json.dumps(data2, ensure_ascii=False) + "\n")


# -------- test code blocks --------

import ast
import re
from typing import List, Dict

def extract_c_blocks(source_code: str) -> List[Dict]:
    # 清理源代码
    cleaned_code = re.sub(r'^```\w*\s*|\s*```$', '', source_code.strip())
    
    try:
        tree = ast.parse(cleaned_code)
    except SyntaxError as e:
        raise SyntaxError(f"代码语法错误: {e}") from e
    
    blocks = []
    lines = cleaned_code.splitlines()
    
    # 定义控制流节点类型
    control_flow_nodes = (
        ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith,
        ast.Break, ast.Continue, ast.Return, ast.Raise, ast.Assert
    )
    
    # 递归处理节点
    def process_node(node, parent_type=None):
        nonlocal blocks, lines
        
        # 获取节点的起始行和结束行
        start_line = getattr(node, 'lineno', None)
        end_line = getattr(node, 'end_lineno', None)
        
        if start_line is None:
            return
        
        # 计算代码块结束行
        if end_line is None:
            # 对于没有结束行的节点，尝试找到下一个同级节点的起始行
            if hasattr(node, 'parent') and hasattr(node.parent, 'body'):
                index = node.parent.body.index(node)
                if index + 1 < len(node.parent.body):
                    next_node = node.parent.body[index + 1]
                    end_line = getattr(next_node, 'lineno', len(lines)) - 1
                else:
                    end_line = len(lines)
            else:
                end_line = start_line
        
        # 提取对应行的源代码
        code_block_lines = lines[start_line-1:end_line]
        code_block = '\n'.join(code_block_lines)
        
        # 确定代码块类型
        block_type = "other"
        
        if isinstance(node, ast.ClassDef):
            block_type = "class_def"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            block_type = "function_def"
        elif isinstance(node, control_flow_nodes):
            block_type = "control_flow"
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AugAssign, ast.AnnAssign)):
            block_type = "other"  # 这些明确归类为"其他"
        
        # 添加到块列表
        blocks.append(code_block)
        
        # 对于函数和类，递归处理其子节点
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.body:
                if isinstance(child, ast.AST):
                    child.parent = node
                    process_node(child, block_type)
    
    # 为所有节点添加父节点引用
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    
    # 从顶级节点开始处理
    for node in tree.body:
        process_node(node)
    
    return blocks


bss = extract_c_blocks(code)
for b in bss:
    print("----significant block----")
    print(b)
