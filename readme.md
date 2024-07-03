# 中医知识理解模型pipeline

## 环境配置
    cd ~/code/demo_project
    pip install transformers
    pip install optimum
    pip install auto-gptq
    export HF_HOME=/mnt/vepfs/fs_users/lkn/huggingface
    export http_proxy=100.68.173.80:3128
    export https_proxy=100.68.173.80:3128
    nohup python pipline/choice_bench.py > output.log 2>&1 &

## 模型配置

### 模型名字：
    Qwen/Qwen1.5-7B-Chat-GPTQ-Int8
    选择原因： 开发机内存限制所以只能选择小模型
### 模型参数配置：
    ```python
    sampling_params = {
                'temperature': 0.7,  # 降低温度以加速生成
                'top_p': 0.9,  # 调整 top_p 值
                'max_new_tokens': 15, ###中文短文本生成。可以帮助inference时间5s左右。 因为GPU只有一个，所以如此限制
            }
    ```

    模型通过从huggingface网站读取权重和配置文件放到本地缓存，所以需要海外代理来访问。 
    下载模型权重之后就可以直接读取本地缓存加载模型。
    并且将模型放到Cuda上利用GPU加速计算
    ```python
    self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", trust_remote_code=True)
    self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat-GPTQ-Int8", trust_remote_code=True)
    self.model.to("cuda")
    ```

### 模型定义文件
    pipline/Model_API.py
    ```python
    def qwen15_14b_chat_api(self, messages): #### 主要文本生成预测函数
        cnt = 0
        model_output = ""
        while cnt < 10:
            ###利用预训练模型进行分词和预测操作
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            start_time = time.time()
            outputs = self.model.generate(**inputs, **self.sampling_params)

            start_time = time.time()
            model_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if model_output.strip() and model_output.strip() != "答案：":
                break
            cnt += 1
        return model_output
    ```    
### 模型运行表现
    每一个问题的inference time在5s左右。 test_A21文件有767个问题。暂时无法判断生成结果是否正确
## pipline 流程

### choice_bench.py
    文件定义了question——type以及测试文件路径和模型名字等参数。（每次通过改变路径和问题类型 参数来处理不同的任务）
    然后 export_distribute_json（）去进行问题的读取预处理以及文本生成和文件保存工作。
    最后 export_union_json （）合并所有batch的答案并按照题号排序成标准提交文件格式。

    注意！！！ 题目类型分为4类： A12，A3, B1 , NLI，其中前三类的处理函数在 export_distribute_json（）得到处理。 但是第四类问题并没有提供 处理函数！！！

### bench_function.py
    文件包含了具体的文件提取，问题分类处理，以及结果保存函数。 处理问题类型：（A12， A3, B1）
    其中关于batch的处理为串行逻辑， 由于本开发机的限制，只有一个GPU，所以无法在本开发机上实行并行处理。
    choice_test_{}函数是具体问题类型的处理函数，每次处理一个batch中，其中每一个问题的串行处理并保存预测结果到子json文件。
    整体函数调用流程：
        1. export_distribute_json（）读取所有问题，并分成batch，每一个batch根据对应题目类型调用choice——test函数处理
        2. choice-test（）函数对batch内部串行处理，调用model生成文本，将结果保存的子文件。
        3. export_union_json （） 合并所有batch的结果文件，并按照题号index排序，保证文件提交格式。

### 使用过程碰到的问题（进行中）
    pipline中model_output并没有清除掉题目输入，造成杂音。（进行中）
    需要修改prompt格式。 原始格式是先解析再选项答案。 问题： 因为max_new_token长度限制，如果先解析会导致答案不能生成。 所以修改为先答案再解析
    

   
