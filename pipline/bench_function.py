
import os
import json
import torch
import re
import torch.multiprocessing as mp
from typing import List, Union
import time
from tqdm import tqdm
from collections import Counter

# args = parse_args()


def get_api_key(filename: str, start_num: int, end_num: int) -> List[str]:
    """
    Retrieves API keys from a file.

    :param filename: Name of the file containing API keys
    :param start_num: Starting line number for reading the file
    :param end_num: Ending line number for reading the file
    :return: List of API keys
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    pattern = re.compile(r'sk-[\s\S]*?(?=\s*\n)')
    api_key_list = []
    
    for i in range(start_num, end_num):
        api_key = pattern.findall(lines[i])
        if len(api_key) != 0:
            api_key_list.append(api_key[0])
    
    return api_key_list


def extract_choice_answer(model_output, question_type, answer_lenth=None):
    """
    Extract choice answer from model output

    Format of model_output that is expected:
    'single_choice': choice answer should be the last Capital Letter of the model_output, e.g.: "...【答案】 A <eoa>"
    'multi_question_choice': "...【答案】A ... 【答案】C ..." or write the choice answers at the beginning of the model_output, e.g. "A C D E F...."
    'multi_choice': "...【答案】 ABD " or write the choice answers at the end of the model_output, e.g. "... ACD"
    'five_out_of_seven': choice answers should be the first five Capital Letters of the model_output, e.g. "A C D F B ...."
    """
    if question_type == 'A1+A2' or question_type == 'A3+A4' or question_type == 'B1':
        model_answer = []
        model_output = model_output[::-1]
        pattern = r"([A-Z]).*?案答"
        check_info = re.search(pattern, model_output)
        if check_info:
            pattern = r"\．[A-Z]"
            temp = re.findall(pattern, model_output)
            if len(temp) > 0:
                # answer = temp[0]
                answer = check_info.group(1)
                model_answer.append(answer)
            else:
                temp = re.findall(r'[A-E]', model_output)
                if len(temp) != 0:
                    answer = temp[0]
                    model_answer.append(answer)
        else:
            temp = re.findall(r'[A-E]', model_output)
            if len(temp) != 0:
                answer = temp[0]
                model_answer.append(answer)
    else:
        model_answer = []

    return model_answer

def A3_second_check(model_output):
    pattern = r"【答案】"
    temp = re.findall(pattern, model_output)
    check_answer = list()
    if len(temp) > 0:
        model_output = model_output.replace("\n", "")
        pattern = r"答案】.*?([A-Z])"
        temp = re.findall(pattern, model_output)
        if len(temp) > 0:
            answer = temp[-1]
            check_answer.append(answer)
        else:
            check_answer = []
    else:
        pattern = r"答案.*?([A-Z])"
        check_info = re.findall(pattern, model_output)
        if check_info:
            answer = check_info[-1]
            check_answer.append(answer)
        else:
            model_output = model_output[::-1]
            temp = re.findall(r'[A-E]', model_output)
            if len(temp) != 0:
                answer = temp[0]
                check_answer.append(answer)
    return check_answer

def pattern_second_check(ans_list, model_output):
    check_answer = list()
    ans_list = [
        ans.replace("A", "").replace("B", "").replace("C", "").replace("D", "").replace("E", "").replace(
            "．", "").replace(".", "") for ans in ans_list if len(ans) > 0]
    ans_id = -1
    candidate_ans = {}
    for ans in ans_list:
        a_list = re.split(r'[，。；\s]', ans)
        max_count = 0
        for a in a_list:
            if a in model_output:
                ans_id = ans_list.index(ans)
                c = model_output.count(a)
                max_count += c
        if max_count > 0:
            candidate_ans[ans] = max_count
    if len(candidate_ans) > 1:
        # 有多个选项
        # 如果多个选项中，有的选项出现的频率超过了其它选项，那么认为这个选项是正确答案
        max_value = max(candidate_ans.values())
        value_clist = Counter(candidate_ans.values())
        if value_clist[max_value] == 1:
            unique_max_key = [key for key, value in candidate_ans.items() if value == max_value][0]
            ans_id = ans_list.index(unique_max_key)
            check_answer.append(chr(ans_id + 65))# chr(ans_id + 65) 通过将 ans_id 加上 65，转换为相应的字母
        else:
            # print(candidate_ans)
            print("")
    elif ans_id >= 0:
        check_answer.append(chr(ans_id + 65))
    return check_answer

def choice_test_A12(kwargs):
    print("question is a12")
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    question_type = kwargs['question_type']
    save_directory = kwargs['save_directory']
    args = kwargs['args']
   
    model_answer_dict = []
    for i in range(start_num, end_num):
        print("the current num is ", i)
        start_time = time.time()
        index = data[i]['index']
        question = data[i]['question'].strip() + '\n'
        model_output = model_api.send_request_turbo(prompt, question)
        output_time = time.time()
        print("model inference time is ", output_time- start_time)
        model_answer = extract_choice_answer(model_output, question_type, 5)
        # print("extract time is ", time.time()- output_time)
        if len(model_answer) == 0:
            ans_list = question.split("\n")[1:]
            model_answer = pattern_second_check(ans_list, model_output)
        analysis_end_pos = model_output.find("【解析】") + len("【解析】")
        analysis = model_output[analysis_end_pos:]    
        dict = {
            'index': index,
            'question': question, 
            'answer': model_answer,
            'analysis': analysis
        }
        model_answer_dict.append(dict)

    file_name = f"seperate_{start_num}-{end_num}.json"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        output = {
            'keyword': keyword, 
            'example': model_answer_dict
        }
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.close()

def extract_label_NLI(output):
    # 使用正则表达式提取答案
    match = re.search(r'【答案】: (\d+) <eoe>', output)

    # 如果找到匹配项，则提取答案
    if match:
        answer = match.group(1).strip()
        print(f"Extracted answer: {answer}")
    else:
        print("No match found")
    return answer

def test_NLI(**kwargs):
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    question_type = kwargs['question_type']
    save_directory = kwargs['save_directory']
    args = kwargs['args']
    
    model_answer_dict = []
    for i in range(start_num, end_num):
        print(f"The current num is {i}")
        idx = data[i]['idx']
        premise = data[i]['premise']
        hypothesis = data[i]['hypothesis']
        start_time = time.time()
        model_output = model_api.send_request_GPT_NLI(prompt, premise, hypothesis)
        print(f"inference time is {time.time() - start_time}")
        # print(model_output)
        answer = extract_label_NLI(model_output)
        dict = {
            'idx': idx,
            'premise': premise, 
            'hypothesis': hypothesis,
            'label': answer
        }
        model_answer_dict.append(dict)
        
    file_name = f"seperate_{start_num}-{end_num}.json"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        output = {
            'keyword': keyword, 
            'example': model_answer_dict
        }
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.close()

def choice_test_A34(**kwargs):
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    question_type = kwargs['question_type']
    save_directory = kwargs['save_directory']
    args = kwargs['args']

    model_answer_dict = []
    for i in range(start_num, end_num):
        index = data[i]['index']
        question = data[i]['question']    # list() 包含多个小问题和答案
        share_content = data[i]['share_content']
        model_output = model_api.send_request_chat(prompt, share_content, question, question_type)

        question_list = []
        for sub_question, output in zip(question, model_output):
            model_answer = extract_choice_answer(output, question_type, 5)
            if question_type == 'A3+A4':
                model_answer = A3_second_check(output)
            if len(model_answer) == 0:
                if question_type == 'A3+A4':
                    ans_list = sub_question.split("\n")[1:]
                else:
                    ans_list = share_content.split("\n")[1:]
                model_answer = pattern_second_check(ans_list, output)
            sub_question_dict = {
                'sub_question': sub_question['sub_question'],
                'answer': model_answer,
                'analysis': output
            }
            question_list.append(sub_question_dict)
        # TODO: which content of temp we expect

        dict = {
            'index': index,
            'share_content': share_content,
            'question': question_list
        }
        model_answer_dict.append(dict)

    file_name = f"_seperate_{start_num}-{end_num}.json"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        output = {
            'keyword': keyword,
            'example': model_answer_dict
        }
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.close()

def export_union_json(directory: str, model_name: str, keyword: str, zero_shot_prompt_text: Union[str, List[str]], question_type: str) -> None:
    """
    Merges JSON files containing processed examples in a directory into a single JSON file.

    :param directory: Directory containing the JSON files
    :param model_name: Name of the model used to process the examples
    :param keyword: Keyword used to identify the JSON files
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    :param question_type: Type of questions in the JSON files (e.g. single_choice, five_out_of_seven, etc.)
    """
    
    save_directory = os.path.join(directory, f'{model_name}_{keyword}')
    if os.path.exists(save_directory):
        output = {
                        'keywords': keyword,
                        'model_name': model_name,
                        'prompt': zero_shot_prompt_text, 
                        'example': []
                    }
        
        # Iterate through the JSON files with the specified keyword in the directory
        
        print("Start to merge json files")
        files = [file for file in os.listdir(save_directory) if file.endswith('.json')]
        for file in files:
            file_path = os.path.join(save_directory, file)

            # Load and merge the data from the JSON files
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                output['example'] += (data['example'])
        # Save the merged data into a single JSON file
        merge_file = os.path.join(directory, f'{keyword}_predictions.json')
        output['example'] = sorted(output['example'], key=lambda x: x['index'])
        with open(merge_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


def export_distribute_json(
        model_api,
        model_name: str, 
        directory: str, 
        keyword: str, 
        zero_shot_prompt_text: Union[str , List[str]], 
        question_type: str,
        args,
        parallel_num: int = 1
    ) -> None:
    """
    Distributes the task of processing examples in a JSON file across multiple processes.

    :param model_name: Name of the model to use
    :param directory: Directory containing the JSON file
    :param keyword: Keyword used to identify the JSON file
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    :param question_type: Type of questions in the JSON file (e.g. single_choice, five_out_of_seven, etc.)
    :param parallel_num: Number of parallel processes to use (default: 5)
    
    """
    # Find the JSON file with the specified keyword
    for root, _, files in os.walk(directory):
        for file in files:
            if file == f'{keyword}.json':
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
    
    example_num = len(data['example'])
    print(" the example num is ", example_num)    
    # Prepare the list of keyword arguments for parallel processing
    kwargs_list = []
    batch_size = example_num // parallel_num + 1
    print(" the batch size is ", batch_size)    
    save_directory = os.path.join(directory, f'{model_name}_{keyword}')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for idx in range(parallel_num):
        start_num = idx * batch_size
        end_num = min(start_num + batch_size, example_num)
        if start_num >= example_num:
            break

        kwargs = {
            'model_api': model_api,
            'start_num': start_num,
            'end_num': end_num,
            'model_name': model_name, 
            'data': data, 
            'keyword': keyword,
            'prompt': zero_shot_prompt_text, 
            'question_type': question_type, 
            'save_directory': save_directory,
            'args': args,
        }
        
        kwargs_list.append(kwargs)   

    # 使用串行逻辑进行批次处理
    print("Start to run model:\n")
    if question_type in ["A1+A2"]:
        for kwargs in tqdm(kwargs_list, desc="Processing A1+A2 Questions"):
            choice_test_A12(kwargs)
    elif question_type in ["A3+A4", "B1"]:
        for kwargs in tqdm(kwargs_list, desc="Processing A3+A4/B1 Questions"):
            choice_test_A34(**kwargs)
    elif question_type in ["NLI"]:
        for kwargs in tqdm(kwargs_list, desc="Processing NLI Questions"):
            test_NLI(**kwargs)
    # if question_type in ["A1+A2"]:
    #     for kwargs in kwargs_list:
    #        choice_test_A12(**kwargs)
    # elif question_type in ["A3+A4", "B1"]:
    #     for kwargs in kwargs_list:
    #        choice_test_A34(**kwargs)

    
# def process_choice_test_A12(kwargs):
#     choice_test_A12(**kwargs)

# def process_choice_test_A34(kwargs):
#     choice_test_A34(**kwargs)