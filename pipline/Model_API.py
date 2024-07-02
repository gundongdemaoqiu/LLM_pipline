from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
class API():
    def __init__(self, api_key_list: str, model_name: str = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", temperature: float = 0.0,
                 max_tokens: int = 1024):
        self.api_key_list = api_key_list
        self.model_name = model_name  # 新的model, 支持1w+
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 检查并设置设备（CPU 或 GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the tokenizer and model using Hugging Face Transformers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        print(f"Tokenizer for model {self.model_name} initialized successfully.\n\n")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        print(f"Model {self.model_name} initialized successfully.\n")

        # 将模型移动到设备
        self.model.to(self.device)

        self.sampling_params = {
            'temperature': 0.7,  # 降低温度以加速生成
            'top_p': 0.9,  # 调整 top_p 值
            'max_new_tokens': 70, ###中文短文本生成。可以帮助inference时间5s左右。 因为GPU只有一个，所以如此限制
        }

    def send_request_GPT_NLI(self, prompt, premise, hypothesis):
        zero_shot_prompt_message = {'role': 'system', 'content': prompt}
        messages = [zero_shot_prompt_message]
        question = f"输入: \n问题是: \n前提是{premise},假设是{hypothesis}输出：\n"
        message = {"role": "user", "content": question}
        messages.append(message)
        
        model_output = ""
        try:
            if 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int8' in self.model_name:
                model_output = self.qwen15_14b_chat_api(messages)
        except Exception as e:
            print('Exception:', e)
        return model_output
    
    # GPT系列
    def send_request_turbo(self, prompt, question):
        zero_shot_prompt_message = {'role': 'system', 'content': prompt}
        messages = [zero_shot_prompt_message]
        question = f"输入：\n问题是：\n{question}输出：\n"
        message = {"role": "user", "content": question}
        messages.append(message)

        model_output = ""
        try:
            if 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int8' in self.model_name:
                model_output = self.qwen15_14b_chat_api(messages)
        except Exception as e:
            print('Exception:', e)
        return model_output

    # 多轮会话
    def send_request_chat(self, prompt, share_content, questions, question_type="A3+A4"):
        if question_type == "A3+A4":
            question_chose = "请根据相关的知识和案例的内容，选出唯一一个正确的选项\n"
        else:
            question_chose = "请根据相关的知识，选出在共享答案中唯一一个正确的选项\n"
        zero_shot_prompt_message = {'role': 'system', 'content': prompt}
        messages = [zero_shot_prompt_message]
        i = 0
        if question_type == "A3+A4":
            messages.append({
                'role': 'user',
                'content': f"案例是：{share_content}"
            })
        else:
            messages.append({
                'role': 'user',
                'content': f"{share_content}"
            })
        model_output_list = []
        while i < len(questions):
            sub_question = questions[i]['sub_question']
            question = f"输入：问题是：\n问题{sub_question}{question_chose}输出：\n"
            message = {"role": "user", "content": question}
            messages.append(message)
            model_output = ""
            try:
                if 'Qwen/Qwen1.5-14B-Chat' in self.model_name:
                    model_output = self.qwen15_14b_chat_api(messages)
            except Exception as e:
                print('Exception:', e)
            messages.append({
                "role": "assistant",
                "content": model_output
            })
            model_output_list.append(model_output)
            i += 1
        return model_output_list

    import time

    def qwen15_14b_chat_api(self, messages):
        cnt = 0
        model_output = ""
        while cnt < 10:
            start_time = time.time()
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            # print(f"Tokenization time: {time.time() - start_time}")

            start_time = time.time()
            outputs = self.model.generate(**inputs, **self.sampling_params)
            # print(f"Model generation time: {time.time() - start_time}")

            start_time = time.time()
            model_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(f"Decoding time: {time.time() - start_time}")
            # 处理 model_output，移除系统的输入部分
            model_output = self.clean_model_output(model_output, messages)
            if model_output.strip() and model_output.strip() != "答案：":
                break
            cnt += 1
        return model_output
    
    def clean_model_output(self, model_output, messages):
        """
        清理 model_output，移除系统的输入部分
        """
        system_input = messages[0]['content']
        user_input = messages[1]['content']
        
        # 找到 system_input 和 user_input 的结束位置
        system_end_pos = model_output.find(system_input) + len(system_input)
        user_end_pos = model_output.find(user_input, system_end_pos) + len(user_input)
        
        # 只保留输出部分
        output_start_pos = model_output.find("输出：\n", user_end_pos) + len("输出：\n")
        clean_output = model_output[output_start_pos:]
        
        assistant_end_pos = clean_output.find("assistant") + len("assistant")
        clean_output = clean_output[assistant_end_pos:]   #########  只保留【答案】：  。。。 <eoe>\n 【解析】： 。。。<eoe>
        return clean_output

















# # --*-- conding:utf-8 --*--
# # @Time : 2024/1/11 17:21
# # @Author : YWJ
# # @Email : 52215901025@stu.ecnu.edu.cn
# # @File : Model_API.py
# # @Software : PyCharm
# # @Description :  各个模型的API接口
# # import os
# # import openai
# # import requests
# # import urllib
# # import json
# # import time
# # from http import HTTPStatus
# # import dashscope
# # import random
# # from vllm import  SamplingParams
# from transformers import AutoModelForCausalLM, AutoTokenizer
# # from dashscope import Generation
# # from dashscope.api_entities.dashscope_response import Role


# class API():
#     def __init__(self, api_key_list: str, model_name: str = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", temperature: float = 0.0,
#                  max_tokens: int = 1024):
#         self.api_key_list = api_key_list
#         self.model_name = model_name  # 新的model, 支持1w+
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         # self.llm = LLM("/data/home-old/mguan/MyProject/ChatWK-main/Qwen-merge-lora/pt-chat-1-9-lora",tokenizer_mode='auto',
#         #       trust_remote_code=True,
#         #       enforce_eager=True,
#         #       enable_prefix_caching=True)
#         # self.llm = LLM("/data/home-old/mguan/MyProject/ChatWK-main/Qwen1.5-14B-Chat", tokenizer_mode='auto',
#         #       trust_remote_code=True,
#         #       enforce_eager=True,
#         #       enable_prefix_caching=True)
        
#         # self.tokenizer = AutoTokenizer.from_pretrained("/data/home-old/mguan/MyProject/ChatWK-main/Qwen1.5-14B-Chat", trust_remote_code=True)
#         # Initialize the tokenizer and model using Hugging Face Transformers
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
#         print(f"Tokenizer for model {self.model_name} initialized successfully.\n\n")
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
#         print(f"Model {self.model_name} initialized successfully.\n")

#         # self.sampling_params = SamplingParams(temperature=0.85, top_p=0.8, max_tokens=512)
#         self.sampling_params = {
#             'temperature': 0.85,
#             'top_p': 0.8,
#             'max_length': 512  # max_tokens equivalent in transformers
#         }

#     # GPT系列
#     def send_request_turbo(self, prompt, question):
#         """
#         """
        
#         zero_shot_prompt_message = {'role': 'system', 'content': prompt}

#         messages = [zero_shot_prompt_message]
#         # messages = []
#         question = f"输入：\n问题是：\n{question}输出：\n"
#         message = {"role": "user", "content": question}
#         messages.append(message)
#         # for m in messages:
#         #     print(m["role"])
#         #     print(m["content"])
#         output = {}
#         model_output = ""
#         try:
#             if 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int8' in self.model_name:
#                 model_output = self.qwen15_14b_chat_api(messages)
#         except Exception as e:
#             print('Exception:', e)
#         return model_output

#     # 多轮会话
#     def send_request_chat(self, prompt, share_content, questions, question_type="A3+A4"):
#         """
#         reference_lists: 里面的列表元素是每个问题对应的参考信息列表
#         title_lists
#         """
#         if question_type == "A3+A4":
#             question_chose = "请根据相关的知识和案例的内容，选出唯一一个正确的选项\n"
#         else:
#             question_chose = "请根据相关的知识，选出在共享答案中唯一一个正确的选项\n"
#         zero_shot_prompt_message = {'role': 'system', 'content': prompt}
#         messages = [zero_shot_prompt_message]
#         # messages = []
#         i = 0
#         if question_type == "A3+A4":
#             messages.append({
#                 'role': 'user',
#                 'content': f"案例是：{share_content}"
#             })
#         else:
#             messages.append({
#                 'role': 'user',
#                 'content': f"{share_content}"
#             })
#             # share_content_prompt = f"{share_content}"
#         model_output_list = []
#         while i < len(questions):
#             sub_question = questions[i]['sub_question']
#             question = f"输入：问题是：\n问题{sub_question}{question_chose}输出：\n"
#             message = {"role": "user", "content": question}
#             messages.append(message)
#             # for m in messages:
#             #     print(m["role"])
#             #     print(m["content"])
#             model_output = ""
#             try:
#                 if 'Qwen/Qwen1.5-14B-Chat' in self.model_name:
#                     model_output = self.qwen15_14b_chat_api(messages)
#             except Exception as e:
#                 print('Exception:', e)
#             messages.append({
#                 "role": "assistant",
#                 "content": model_output
#             })
#             model_output_list.append(model_output)
#             i += 1
#         # print(model_output)
#         return model_output_list

#     def qwen15_14b_chat_api(self, messages):
#         cnt = 0
#         model_output = ""
#         while cnt < 10:
#             # 使用预训练的 apply_chat_template 函数
#             text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             inputs = self.tokenizer(text, return_tensors="pt")
#             outputs = self.model.generate(**inputs, **self.sampling_params)
#             model_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             if model_output.strip() and model_output.strip() != "答案：":
#                 break
#             cnt += 1
#             # print('*' * 50, cnt, '*' * 50, f"Generated text: {model_output!r}")
#         return model_output
#         # cnt = 0
#         # while (cnt < 10):
#         #     text = self.tokenizer.apply_chat_template(
#         #                 messages,
#         #                 tokenize=False,
#         #                 add_generation_prompt=True
#         #             )
#         #     outputs = self.llm.generate(text, self.sampling_params)
#         #     for output in outputs:
#         #         # model_output = output.outputs[0].text.split("<|endoftext|>")[0]
#         #         model_output = output.outputs[0].text
#         #     if model_output != "" and model_output != "答案：":
#         #         break
#         #     cnt += 1
#         #     print('*'*50, cnt, '*'*50, f"Generated text: {model_output!r}")
#         # return model_output




