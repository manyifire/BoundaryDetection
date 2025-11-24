from nltk.tokenize import sent_tokenize
from scipy.spatial import distance
import datetime
import numpy as np
import torch
import os
import random
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import pandas as pd
import random
from scipy.spatial import distance
import argparse
import ast
import nltk
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import ast
import re
from typing import List, Dict, Tuple

nltk.download('punkt_tab')

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_block(source_code: str) -> List[Tuple[str, Tuple[int, int]]]:
    """
    将代码分割成不同的逻辑块
    
    参数:
        source_code: 要分割的代码字符串
        
    返回:
        List[Tuple[str, Tuple[int, int]]]: 每个元素是(代码块内容, (起始行, 结束行))
    """
    # 清理源代码
    cleaned_code = re.sub(r'^```\w*\s*|\s*```$', '', source_code.strip())
    
    try:
        tree = ast.parse(cleaned_code)
    except SyntaxError as e:
        # 如果解析失败，返回整个代码作为一个块
        lines = cleaned_code.splitlines()
        return [(cleaned_code, (0, len(lines)-1))]
    
    blocks = []
    lines = cleaned_code.splitlines()
    
    # 定义控制流节点类型
    control_flow_nodes = (
        ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith,
        ast.Break, ast.Continue, ast.Return, ast.Raise, ast.Assert
    )
    
    # 处理AST节点
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
        blocks.append((code_block, (start_line-1, end_line-1), block_type))
        
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
    
    # 如果没有找到任何块，返回整个代码作为一个块
    if not blocks:
        return [(cleaned_code, (0, len(lines)-1))]
    
    # 返回代码块和它们的行号范围
    return [(block, line_range) for block, line_range, _ in blocks]

def get_euclidean_distance(v1,v2):
    return round(distance.euclidean(v1,v2),4)

def get_one_triplet_example(data, target_essayset=None):
    try:
        while True:
            essayset = random.choice([e for e in [1,2,3,4,5,6,7,8] if e!= 99999])
            random_record = data[(data.set_ix == 'train')].sample(n = 1).to_dict(orient='records')[0]
            pos = random_record['human_part']
            neg = random_record['machine_part']
            if len(pos) >= 5 and len(neg) >= 5:
                break  # 满足条件才跳出循环，继续后续采样
        if random.random() > 0.5:
            p_sents = random.sample(pos,1)
            n_sents = random.sample(neg, 2)
            random.shuffle(n_sents)
            example = InputExample(texts=[n_sents[0], n_sents[1], p_sents[0]])
        else:
            p_sents = random.sample(pos,2)
            random.shuffle(p_sents)
            n_sents = random.sample(neg, 1)
            example = InputExample(texts=[p_sents[0], p_sents[1], n_sents[0]])
        return example
    except:
        return get_one_triplet_example(data)

    
def get_one_epoch_train_examples(data,examples_per_epoch, target_essayset=None):
    res=[]
    # 返回5000个triplet
    for i in range(examples_per_epoch):
        res.append( get_one_triplet_example(data) )
        
    return res

def predict_boundary_for_one_essay(model, essay):
    print(f'test essay: \n {essay}')
    # 使用新的块分割函数
    blocks_with_ranges = get_block(essay)
    blocks = [block for block, _ in blocks_with_ranges]
    print(f'blocks: {blocks}')

    distance_result = []
    blocks_emb = []
    
    for block in blocks:
        emb = model.encode([block]).flatten()
        blocks_emb.append(emb)

    for i in range(len(blocks_emb)-1):
        emb1 = blocks_emb[max(i+1-prototype_size, 0): i+1]
        emb2 = blocks_emb[i+1: i+1+prototype_size]
        
        p1 = np.mean(emb1, axis=0)
        p2 = np.mean(emb2, axis=0)
        dist = get_euclidean_distance(p1, p2)
        distance_result.append(dist)
    
    # 计算边界
    boundary = sorted([(dist, ix+1) for ix, dist in enumerate(distance_result)], 
                     key=lambda x: x[0], reverse=True)
    boundary = [ix for dist, ix in boundary][:topk]
    
    # 处理边界列表长度不足的情况
    if len(boundary) < topk:
        # 获取所有可能的候选索引
        all_indices = [i+1 for i in range(len(distance_result))]
        available_indices = [ix for ix in all_indices if ix not in boundary]
        
        # 如果边界列表为空，用随机候选填充
        if len(boundary) == 0:
            if len(available_indices) == 0:
                # 如果没有可用索引，全部用默认值（如1）填充
                boundary = [1] * topk
            else:
                # 随机选择足够的索引补全
                additional = random.sample(available_indices, 
                                         min(topk, len(available_indices)))
                boundary.extend(additional)
                # 如果仍然不足，重复最后一个元素
                if len(boundary) < topk:
                    boundary.extend([boundary[-1]] * (topk - len(boundary)))
        else:
            # 原有处理逻辑
            if len(available_indices) < (topk - len(boundary)):
                boundary.extend([boundary[-1]] * (topk - len(boundary)))
            else:
                boundary.extend(random.sample(available_indices, topk - len(boundary)))
        
        boundary.sort()  # 保持排序

    if len(distance_result) < topk:
        random_boundary = boundary
    else:
        random_boundary = random.sample([i+1 for i in range(len(distance_result))], topk)
    
    return boundary, random_boundary

def get_evaluation_result(predicted, labels):
    tpl = list(zip(predicted, labels))
    result = {'accurance':[], 'precision':[], 'recall':[], 'f1_score':[]}
    
    for p, l in tpl:
        intersection = set(p) & set(l)
        precision = 0.0001 + len(intersection) * 1.0 / topk
        recall = 0.0001 + len(intersection) * 1.0 / len(l)
       
        result['precision'].append(precision) 
        result['recall'].append(recall) 
        result['f1_score'].append(2 * precision * recall * 1.0 / (precision + recall))
    
    return round(np.mean(result['precision']), 3), round(np.mean(result['recall']), 3), round(np.mean(result['f1_score']), 3)

def get_classification_labels(code, predicted, labels):
    """
    修改为基于代码块的标签生成
    """
    blocks_with_ranges = get_block(code)
    num_blocks = len(blocks_with_ranges)
    
    real_label_H, real_label_M, pred_label_H, pred_label_M = None, None, None, None

    if len(labels) == 0:
        print('!!!!!!!!!!!!!!labels is empty: ', labels)
    
    # 将行号边界转换为块索引边界
    def line_boundary_to_block_boundary(line_boundaries, blocks_with_ranges):
        block_boundaries = []
        for line_boundary in line_boundaries:
            # 找到包含该行号的块
            for block_idx, (_, (start_line, end_line)) in enumerate(blocks_with_ranges):
                if start_line <= line_boundary <= end_line:
                    block_boundaries.append(block_idx + 1)  # 转换为1-based索引
                    break
            else:
                # 如果没有找到，使用最后一个块
                block_boundaries.append(len(blocks_with_ranges))
        return block_boundaries
    
    # 转换真实标签
    block_labels = line_boundary_to_block_boundary(labels, blocks_with_ranges) if labels else []
    
    if len(block_labels) == 1:
        # HM模式
        boundary = block_labels[0]
        real_label_H = [0] * boundary + [1] * (num_blocks - boundary)
        real_label_M = [1] * boundary + [0] * (num_blocks - boundary)
    
    elif len(block_labels) == 2:
        # HMH或MHM模式
        boundary1, boundary2 = block_labels
        real_label_H = [0] * boundary1 + [1] * (boundary2 - boundary1) + [0] * (num_blocks - boundary2)
        real_label_M = [1] * boundary1 + [0] * (boundary2 - boundary1) + [1] * (num_blocks - boundary2)
    
    # 处理预测边界
    if len(predicted) == 1:
        boundary = predicted[0]
        pred_label_H = [0] * boundary + [1] * (num_blocks - boundary)
        pred_label_M = [1] * boundary + [0] * (num_blocks - boundary)
    
    elif len(predicted) == 2:
        boundary1, boundary2 = predicted
        pred_label_H = [0] * boundary1 + [1] * (boundary2 - boundary1) + [0] * (num_blocks - boundary2)
        pred_label_M = [1] * boundary1 + [0] * (boundary2 - boundary1) + [1] * (num_blocks - boundary2)

    return real_label_H, real_label_M, pred_label_H, pred_label_M

def get_classification_results(real_label_H, real_label_M, pred_label_H, pred_label_M):
    metrics = {
        'H_H': {'real': real_label_H, 'pred': pred_label_H},
        'H_M': {'real': real_label_H, 'pred': pred_label_M},
        'M_H': {'real': real_label_M, 'pred': pred_label_H},
        'M_M': {'real': real_label_M, 'pred': pred_label_M},
    }

    best_accuracy = -1
    best_macro_f1 = -1
    precision = -1
    recall = -1

    for key, val in metrics.items():
        if len(val['real']) != len(val['pred']):
            continue
            
        current_accuracy = accuracy_score(np.array(val['real']), np.array(val['pred']))
        current_macro_f1 = f1_score(np.array(val['real']), np.array(val['pred']), average='macro')

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_macro_f1 = current_macro_f1
            precision = precision_score(np.array(val['real']), np.array(val['pred']), average=None)
            recall = recall_score(np.array(val['real']), np.array(val['pred']), average=None)
            
        elif current_accuracy == best_accuracy and current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            precision = precision_score(np.array(val['real']), np.array(val['pred']), average=None)
            recall = recall_score(np.array(val['real']), np.array(val['pred']), average=None)

    print("Accuracy: {:.1f}".format(best_accuracy * 100))
    print("Macro F1 Score: {:.1f}".format(best_macro_f1 * 100))
    print("Precision/Recall per class: ")
    precision_recall = ' '.join(["{:.1f}/{:.1f}".format(p*100, r*100) for p, r in zip(precision, recall)])
    print(precision_recall)

    result = {"precision": precision, "recall": recall, "accuracy": best_accuracy, "macro_f1": best_macro_f1,"precision_recall": precision_recall}
    
    print(result)
    return result

def evaluate_model(model, data, targetprompt, test_type):
    print('--------Start Evaluating: {}--------'.format(str(datetime.datetime.now())[:19]))
    if test_type == 'valid':
        prompt_data = data[
            (data.set_ix == 'valid')
        ]
    elif test_type == 'test':  # that is 'test' type on unseen prompt
        prompt_data = data[
            (data.set_ix == 'test')
        ]
    
    all_hybird_essay = list(prompt_data['hybrid_code'])
    all_labels = [ast.literal_eval(e) for e in list(prompt_data['boundary_ix'])] # each elements is a list
    all_predicted_boundary = []
    random_guess = []
    allreal_label_H = []
    allreal_label_M = []
    allpred_label_H = []
    allpred_label_M = []

    for idx, essay in enumerate(all_hybird_essay):
        if all_labels[idx] is None or len(all_labels[idx]) == 0:
            print('problem code: \n {}'.format(essay))
            continue
        if len(all_labels[idx]) == 1:
            if all_labels[idx][0] == None:
                print('problem code: \n {}'.format(essay))
                continue
        if len(all_labels[idx]) == 2:
            if all_labels[idx][0] == None or all_labels[idx][1] == None:
                print('problem code: \n {}'.format(essay))
                continue
        if type(essay) != str:
            print('problem code: \n {}'.format(essay))
            continue
        
        # 检查代码块数量
        blocks = get_block(essay)
        if len(blocks) <= 1:  # 至少需要2个块才能有边界
            continue
            
        predicted_boundary, rn_boundary = predict_boundary_for_one_essay(model, essay)
        all_predicted_boundary.append(predicted_boundary)
        random_guess.append(rn_boundary)

        real_label_H, real_label_M, pred_label_H, pred_label_M = get_classification_labels(essay, predicted_boundary, all_labels[idx])
        
        # 确保标签长度一致
        if (real_label_H is not None and pred_label_H is not None and 
            len(real_label_H) == len(pred_label_H)):
            allreal_label_H.extend(real_label_H)
            allpred_label_H.extend(pred_label_H)
        
        if (real_label_M is not None and pred_label_M is not None and 
            len(real_label_M) == len(pred_label_M)):
            allreal_label_M.extend(real_label_M)
            allpred_label_M.extend(pred_label_M)
        
    precision, recall, f1_score_val = get_evaluation_result(all_predicted_boundary, all_labels)
    rn_precision, rn_recall, rn_f1_score = get_evaluation_result(random_guess, all_labels)
    
    classification_result = get_classification_results(
        allreal_label_H, allreal_label_M, allpred_label_H, allpred_label_M
    )
        
    print('--------------------{}ing------------------------'.format(test_type))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('f1_score: {}'.format(f1_score_val))
    print('-----Random-------')
    print('rn_Precision: {}'.format(rn_precision))
    print('rn_Recall: {}'.format(rn_recall))
    print('rn_f1_score: {}'.format(rn_f1_score))

    print('-----classification-------')
    print('Precision: {}'.format(classification_result['precision']))
    print('Recall: {}'.format(classification_result['recall']))
    print('f1_score: {}'.format(classification_result['macro_f1']))
    print('Accuracy: {}'.format(classification_result['accuracy']))
    print('Precision/Recall per class: {}'.format(classification_result['precision_recall']))
        
    if test_type == 'test':
        result_file = 'lr_{}_predicted_results_{}.xlsx'.format(lr, targetprompt)
        prompt_data.to_excel(result_file, index=None)
        prompt_data = pd.read_excel(result_file)
        prompt_data['predicted_boundary'] = pd.Series(all_predicted_boundary)
        prompt_data['random_boundary'] = pd.Series(random_guess)
        prompt_data.to_excel(result_file, index=None)
        
    print('--------Ending Evaluating: {}--------'.format(str(datetime.datetime.now())[:19]))
        
    return f1_score_val       

def train_model(model, data, lr, num_epochs, examples_per_epoch, current_prompt=None,save_model = 1):
    
    best_metric_so_far = None
    best_model = 'lr{}_bd_{}.bert'.format(lr,current_prompt)

    for i in range(num_epochs):
        print('---------Epoch {} ----time: {}'.format(i + 1,str(datetime.datetime.now())[:19]))

        train_examples = get_one_epoch_train_examples(data,examples_per_epoch)
        
        train_dataloader = DataLoader(train_examples, shuffle = True, batch_size = 1)
        # Triplet Loss 主要用于度量学习（如句向量、图像检索等），它让“anchor-正样本”距离比“anchor-负样本”距离小至少 margin（这里是 1）
        train_loss = losses.TripletLoss(model = model,triplet_margin = 1) # euclidean distance

        model.fit(
            train_objectives = [(train_dataloader, train_loss)],
            #evaluator = evaluator,
            epochs = 1,
            scheduler = 'Constantlr',
            weight_decay = 0.01,
            show_progress_bar = True,#True,
            warmup_steps = 0,
            optimizer_params = {'lr': lr}
        )
        lr = lr * 0.8

        print('--------now evaluating on valid set----------')
        f1 = evaluate_model(model, data, current_prompt, test_type = 'valid')#evaluate_model

        if best_metric_so_far is None or best_metric_so_far < f1:

            old_f1 = best_metric_so_far
            best_metric_so_far = f1

            if os.path.exists(best_model):
                os.remove(best_model)
                print('---old model removed---')

            print('---Best f1 has been updated from {} to {}---, new best model saved'.format(round(old_f1, 4) if old_f1 is not None else None, round(best_metric_so_far,4)))
            torch.save(model,best_model)

        else:
            print('---Old f1 {} > New f1 {}---'.format(round(best_metric_so_far, 4), round(f1, 4)))
            print('No improvement detected compared to last validation round, early stop is triggered.')
            model = torch.load(best_model,weights_only=False)
            break

    print('--------now testing on unseen prompt {}----------'.format(current_prompt))
    evaluate_model(model, data, current_prompt, test_type = 'test')

    if os.path.exists(best_model) and save_model !=1:
        os.remove(best_model)
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="This is a description")
    parser.add_argument('--num_epochs',dest='num_epochs',required = True, type = int)
    parser.add_argument('--prototype_size',dest='prototype_size',required = False, type = int)
    parser.add_argument('--topk',dest='topk',required = False, type = int)
    parser.add_argument('--save_model',dest='save_model',required = False, type = int)
    parser.add_argument('--learning_rate',dest='learning_rate',required = True, type = float) 
    parser.add_argument('--data_file', dest='data_file', required = True, type = str)  
    # parser.add_argument('--targetprompt', dest='targetprompt', required = False, type = int)
    parser.add_argument('--examples_per_epoch',dest='examples_per_epoch',required = True, type = int)

    args = parser.parse_args()  

    
    num_epochs = args.num_epochs
    prototype_size = args.prototype_size if args.prototype_size is not None else 1
    topk = args.topk if args.topk is not None else 2
    save_model = args.save_model if args.save_model is not None else 0
    lr = args.learning_rate
    data_file = args.data_file
    # current_prompt = args.targetprompt
    examples_per_epoch = args.examples_per_epoch
    
    
    print('num_epochs: {}'.format(num_epochs))
    print('prototype_size: {}'.format(prototype_size))
    print('topk: {}'.format(topk))
    print('save_model: {}'.format(True if save_model == 1 else False))
    print('learning_rate: {}'.format(lr))
    print('data_file: {}'.format(data_file))
    # print('target_prompt: {}'.format(current_prompt))
    print('examples_per_epoch: {}'.format(examples_per_epoch))
  
    print('--------Start Loading Data: {}--------'.format(str(datetime.datetime.now())[:19]))

    data = pd.read_excel(data_file)
    print('Data loaded from {}'.format(data_file))
    # model = SentenceTransformer('all-mpnet-base-v2',device = device)
    # model = torch.load('/data/wangmanyi/SeqXGPT/SeqXGPT/model0620.ckpt',weights_only=False)
    model = SentenceTransformer('microsoft/codebert-base',device = device)
    # model = SentenceTransformer('FacebookAI/roberta-large',device = device)
    # model = SentenceTransformer('microsoft/unixcoder-base',device = device)
    print('Model loaded from HuggingFace')
    
    train_model(
            model = model, 
            data = data, 
            lr = lr, 
            num_epochs = num_epochs, 
            examples_per_epoch = examples_per_epoch, 
            # current_prompt = current_prompt,
            save_model = save_model
    )
    
    
