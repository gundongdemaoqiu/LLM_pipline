from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定特定的本地地址保存模型和分词器
local_cache_dir = "/vepfs/fs_users/lkn/huggingface/hub"

# 加载模型和分词器时指定 cache_dir
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=local_cache_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir=local_cache_dir)


# import numpy as np
# import json
# from pathlib import Path
# from docling.document_converter import DocumentConverter
# local_model_path = Path("/root/.cache/huggingface/hub/models--ds4sd--docling-models/snapshots/96e8ba4eb46f125ff2abbbdffbdc2a102d0150b4/")
# print("11111")
# source = "/root/code/pdfs/10.1016_j.msea.2003.10.319.pdf"
# converter = DocumentConverter(artifacts_path=local_model_path)
# print("22222")
# doc = converter.convert_single(source)
# # print(doc.render_as_dict()) 
# data= doc.render_as_dict()
# # 提取主文本内容并添加页码信息
# # 提取主文本内容并添加页码信息
# def extract_main_text(data):
#     main_text = []
#     for element in data.get('main-text', []):
#         # 检查每个条目是否具有 prov 属性
#         if 'prov' in element and element['prov']:
#             page_number = element['prov'][0]['page']
#         else:
#             page_number = 'unknown'  # 标记为未知页码
#         main_text.append({
#             'page': page_number,
#             'type': element.get('type', 'unknown'),
#             'name': element.get('name', 'unknown'),
#             'text': element.get('text', '')
#         })
#     return main_text

# # def extract_tables(data):
# #     tables = []
# #     for page in data.get('content', []):
# #         page_number = page.get('page_number', 'unknown')
# #         for table in page.get('elements', []):
# #             if table['type'] == 'table':
# #                 table_data = {
# #                     'page': page_number,
# #                     'title': table.get('text', ''),
# #                     'type': table.get('type', 'unknown'),
# #                     'columns': table.get('#-cols', 0),
# #                     'rows': table.get('#-rows', 0),
# #                     'data': []
# #                 }
# #                 for row in table.get('data', []):
# #                     row_data = []
# #                     for cell in row:
# #                         row_data.append(cell.get('text', ''))
# #                     table_data['data'].append(row_data)
# #                 tables.append(table_data)
# #     return tables
# # # 提取表格内容并添加页码信息
# def extract_tables(data):
#     tables = []
#     for table in data.get('tables', []):
#         page_number = table['prov'][0]['page']
#         table_data = {
#             'page': page_number,
#             'title': table['text'],
#             'type': table['type'],
#             'columns': table['#-cols'],
#             'rows': table['#-rows'],
#             'data': []
#         }
#         for row in table['data']:
#             row_data = []
#             for cell in row:
#                 row_data.append(cell['text'])
#             table_data['data'].append(row_data)
#         tables.append(table_data)
#     return tables

# # 创建最终的JSON结构
# final_data = {
#     'filename': data['file-info']['filename'],
#     'number-of-pages': data['file-info']['#-pages'],
#     # 'main-text': extract_main_text(data),
#     'tables': extract_tables(data)
# }

# # 格式化输出
# formatted_json = json.dumps(final_data, indent=4)
# print(formatted_json)


# # 是的，针对page的粒度进行筛选是一个合理的策略。这种方式在大多数情况下能够很好地平衡效率和准确性。以下是这种策略的详细步骤和实现：

# # 1. **全局筛选最相关的页面**：对于没有提供特定页面范围的问题，可以在全文范围内筛选最相关的页面。
# # 2. **局部筛选具体内容**：如果所选页面的内容长度超过模型的输入限制，可以进一步在页面内选择相关的段落或表格。

# # ### 实现步骤
# # 1. **提取PDF内容**：使用Docling提取PDF的结构化内容。
# # 2. **全局筛选最相关页面**：根据问题在文档全文范围内进行筛选，找到最相关的页面。
# # 3. **局部筛选具体内容**：在选定的页面内，提取段落和表格，并进行二次筛选，以确保输入长度在允许范围内。

# # ### 示例代码
# # 下面是一个逐页和逐段落筛选的完整实现示例：

# # ```python
# # import os
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # from docling.document_converter import DocumentConverter
# # import json
# # from pprint import pprint

# # PDF_PATH = '/path/to/pdfs/'

# # # 使用Docling提取PDF内容
# # def extract_pdf_content(pdf_path):
# #     converter = DocumentConverter()
# #     doc = converter.convert_single(pdf_path)
# #     return doc.to_dict()

# # # 格式化表格内容为文本形式
# # def format_table_content(table):
# #     table_text = [table.get('text', '')]
# #     for row in table.get('data', []):
# #         row_text = '\t'.join(cell.get('text', '') for cell in row)
# #         table_text.append(row_text)
# #     return '\n'.join(table_text)

# # # 解析和检索PDF内容
# # def parse_pdf_and_concate(obj):
# #     pdf_path = obj["doi"]
# #     pdf_path = pdf_path.replace('/', '_').replace(' (Supporting Information)', '_si')
# #     pdf_path = os.path.join(PDF_PATH, pdf_path + '.pdf')
# #     doc_content = extract_pdf_content(pdf_path)

# #     # 提取问题作为关键词
# #     question = next((entry["content"] for entry in obj["input"] if entry["role"] == "user"), "")

# #     all_pages_content = []
# #     page_content_map = {}
# #     for page in doc_content["content"]:
# #         page_number = page["page_number"]
# #         page_text = []
# #         for element in page["elements"]:
# #             if element["type"] in ["paragraph", "section-header", "table"]:
# #                 if element["type"] == "table":
# #                     page_text.append(format_table_content(element))
# #                 else:
# #                     page_text.append(element["text"])
# #         combined_text = " ".join(page_text)
# #         all_pages_content.append(combined_text)
# #         page_content_map[page_number] = combined_text

# #     # 使用TF-IDF和余弦相似度来检索相关页面
# #     vectorizer = TfidfVectorizer().fit_transform(all_pages_content)
# #     question_vec = vectorizer.transform([question])
# #     cosine_similarities = cosine_similarity(question_vec, vectorizer).flatten()

# #     # 找到最相关的页面索引
# #     relevant_page_indices = cosine_similarities.argsort()[-5:][::-1]  # 选择最相关的5个页面

# #     # 构建初步上下文
# #     relevant_pages = [all_pages_content[i] for i in relevant_page_indices]
# #     attached_file_content = "\nThe file is as follows:\n\n" + " ".join(relevant_pages)

# #     # 如果初步上下文长度超过限制，逐段筛选
# #     if len(attached_file_content) > 1024:
# #         all_paragraphs = []
# #         paragraph_map = {}
# #         for page_index in relevant_page_indices:
# #             page = doc_content["content"][page_index]
# #             page_number = page["page_number"]
# #             for element in page["elements"]:
# #                 if element["type"] in ["paragraph", "section-header"]:
# #                     all_paragraphs.append(element["text"])
# #                     paragraph_map[element["text"]] = (page_number, "text")
# #                 elif element["type"] == "table":
# #                     table_text = format_table_content(element)
# #                     all_paragraphs.append(table_text)
# #                     paragraph_map[table_text] = (page_number, "table")
        
# #         # 使用TF-IDF和余弦相似度来筛选具体段落或表格
# #         vectorizer = TfidfVectorizer().fit_transform(all_paragraphs)
# #         question_vec = vectorizer.transform([question])
# #         cosine_similarities = cosine_similarity(question_vec, vectorizer).flatten()

# #         # 找到最相关的段落或表格索引
# #         relevant_paragraph_indices = cosine_similarities.argsort()[-5:][::-1]  # 选择最相关的5个段落或表格

# #         # 构建最终上下文
# #         relevant_paragraphs = [all_paragraphs[i] for i in relevant_paragraph_indices]
# #         attached_file_content = "\nThe file is as follows:\n\n" + " ".join(relevant_paragraphs)

# #     # 选择性的获取上下文信息，限制长度
# #     attached_file_content = attached_file_content[:1024]  
# #     obj["input"].append({"role": "user", "content": attached_file_content})

# # # 示例对象
# # obj = {
# #     "doi": "10.1002_(Supporting Information)_12345",
# #     "pages": [5, 6],
# #     "input": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are the main findings in the experimental results section?"}]
# # }

# # # 调用函数
# # parse_pdf_and_concate(obj)

# # # 查看更新后的输入内容
# # print(json.dumps(obj["input"], indent=4))
# # ```

# # ### 技术细节和解释
# # 1. **全局筛选最相关页面**：使用TF-IDF和余弦相似度对所有页面进行比较，筛选出与问题最相关的几个页面。
# # 2. **局部筛选具体内容**：如果筛选出的页面内容长度超过限制，进一步筛选页面内的具体段落或表格内容。
# # 3. **多模态处理**：同时处理文本和表格内容，确保检索结果多样性和全面性。

# # ### 关键点
# # 1. **按页面粒度筛选**：初步筛选时按页面进行处理，确保筛选过程简洁高效。
# # 2. **具体段落检索**：当页面内容超出限制时，进一步按段落和表格内容进行精细检索。
# # 3. **准确上下文构建**：通过多轮筛选，确保提取的上下文既相关又符合输入长度限制。

# # 通过这种方法，可以充分利用页面和段落粒度的筛选策略，提高检索精度和上下文构建效果，确保模型能够更准确地回答用户问题。