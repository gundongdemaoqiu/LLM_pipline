import os
import json
from docling.document_converter import DocumentConverter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
PDF_PATH = "/root/code/LLM_pdf/LLM_pipline"
# 使用Docling提取PDF内容
def extract_pdf_content(pdf_path):
    converter = DocumentConverter()
    doc = converter.convert_single(pdf_path)
    return doc.render_as_dict()

# 提取主文本内容并添加页码信息
def extract_main_text(data):
    main_text = []
    for element in data.get('main-text', []):
        # 检查每个条目是否具有 prov 属性
        if 'prov' in element and element['prov']:
            page_number = element['prov'][0]['page']
        else:
            page_number = 'unknown'  # 标记为未知页码
        
        # 如果存在 $ref 字段，替换为引用的内容
        if '$ref' in element:
            ref = element['$ref']
            ref_index = int(ref.split('/')[-1])
            element_content = data.get('figures', [])[ref_index]
            main_text.append({
                'page': page_number,
                'type': element.get('type', 'unknown'),
                'name': element.get('name', 'unknown'),
                'text': element_content  # 使用引用内容替换
            })
        else:
            main_text.append({
                'page': page_number,
                'type': element.get('type', 'unknown'),
                'name': element.get('name', 'unknown'),
                'text': element.get('text', '')
            })
    return main_text

# 将表格数据转换为字符串，用于检索
def table_to_text(table):
    return table.get('title') + '\n' + '\n'.join(['\t'.join(row) for row in table.get('data', [])])

# 提取表格内容并添加页码信息
def extract_tables(data):
    tables = []
    for table in data.get('tables', []):
        page_number = table['prov'][0]['page']
        table_data = {
            'page': page_number,
            'title': table.get('text', ''),
            'type': table.get('type', 'unknown'),
            'columns': table.get('#-cols', 0),
            'rows': table.get('#-rows', 0),
            'text': ''
        }
        table_text = table.get('title', '') + '\n'
        table_text += '\n'.join(['\t'.join(cell.get('text', '') for cell in row) for row in table.get('data', [])])
        table_data['text'] = table_text
        tables.append(table_data)
    return tables

# 将表格信息组合成字符串
def combine_table_info(table):
    return f"Title: {table['title']}\nColumns: {table['columns']}\nRows: {table['rows']}\n{table['text']}"

# 构建最终的JSON结构
def build_final_json(data):
    main_texts = extract_main_text(data)
    tables = extract_tables(data)
    combined_content = []

    for item in main_texts:
        combined_content.append(item)
    
    # 将表格对象转换为统一的结构
    for table in tables:
        combined_content.append({
            'page': table['page'],
            'type': table['type'],
            'name': table['title'],
            'text': combine_table_info(table),
        })

    return {
        'filename': data['file-info']['filename'],
        'number-of-pages': data['file-info']['#-pages'],
        'combined-content': combined_content
    }
    
# 检索和组合PDF内容
def parse_pdf_and_concate(obj):
    pdf_path = obj["doi"].replace('/', '_').replace(' (Supporting Information)', '_si') + '.pdf'
    pdf_path = os.path.join(PDF_PATH, pdf_path)
    data = extract_pdf_content(pdf_path)

    # 构建最终的JSON结构
    final_json = build_final_json(data)
    # print(final_json)
    # 提取问题作为关键词
    question = next((entry["content"] for entry in obj["input"] if entry["role"] == "user"), "")

    all_texts = [
            element['text'] if isinstance(element['text'], str) else json.dumps(element['text'])
            for element in final_json['combined-content']
        ]
    
        # 初始化 TfidfVectorizer 实例
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # 对所有文档进行拟合（学习词汇表和IDF）
    fitted_vectorizer = tfidf_vectorizer.fit(all_texts)

    # 将文档集合转换为TF-IDF特征矩阵
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(all_texts)

    # 将问题转换为与文档集合相同空间的TF-IDF向量
    question_vec = fitted_vectorizer.transform([question])

    # 计算余弦相似度
    cosine_similarities = cosine_similarity(question_vec, tfidf_vectorizer_vectors).flatten()

    # 找到最相关的内容索引
    relevant_content_indices = cosine_similarities.argsort()[-5:][::-1]  # 选择最相关的5个内容

    # 构建初步上下文
    relevant_contents = [all_texts[i] for i in relevant_content_indices]
    attached_file_content = "\nThe file is as follows:\n\n" + " ".join(relevant_contents)

    # 选择性获取上下文信息，限制长度
    attached_file_content = attached_file_content[:1024]
    obj["input"].append({"role": "user", "content": attached_file_content})
    
    
# 示例对象
obj = {
    "doi": "10.1002_adem.201700820",
    "input": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "what is the difference of process condition ,such as temperature, between the 3rd and the 2nd Case in Aging treatment processing, Table 1?"}]
}

# 调用函数
parse_pdf_and_concate(obj)

# 查看更新后的输入内容
print(json.dumps(obj["input"], indent=4))