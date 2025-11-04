# 文档解析工具精选

> **只推荐生产环境验证的 5 个核心工具**

---

## 🎯 快速选型表

| 工具 | Stars | 支持格式 | 质量 | 速度 | 推荐场景 | 推荐度 |
|:---|:---:|:---|:---:|:---:|:---|:---:|
| **MinerU** | 10k+ | PDF/Word/PPT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 复杂PDF、多模态RAG | ⭐⭐⭐⭐⭐ |
| **Unstructured** | 8k+ | 20+格式 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 通用文档解析 | ⭐⭐⭐⭐ |
| **LlamaParse** | - | PDF | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | LlamaIndex 生态 | ⭐⭐⭐⭐ |
| **PyPDF2** | 8k+ | PDF | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 简单PDF提取 | ⭐⭐⭐ |
| **Docling** | 3k+ | PDF/Word | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | IBM 出品 | ⭐⭐⭐ |

---

## 📖 详细对比

### 1. MinerU ⭐⭐⭐⭐⭐ 最推荐

**链接**：https://github.com/opendatalab/MinerU

**核心优势**：
- ✅ 支持复杂 PDF（双栏、表格、公式）
- ✅ 保留文档结构
- ✅ 支持多模态（图片、表格同时提取）
- ✅ 中文友好

**适合场景**：
- 学术论文解析
- 企业合同、报表
- 多模态 RAG 系统

**使用示例**：
```bash
# 安装
pip install magic-pdf

# 解析 PDF
magic-pdf -p document.pdf -o output/
```

**解析质量**：⭐⭐⭐⭐⭐  
**速度**：中等（复杂文档 1 页/秒）  
**学习成本**：⭐⭐（1小时上手）

---

### 2. Unstructured ⭐⭐⭐⭐ 全能选手

**链接**：https://github.com/Unstructured-IO/unstructured

**核心优势**：
- ✅ 支持 20+ 种格式（PDF、Word、Excel、HTML等）
- ✅ 统一 API
- ✅ 结构化输出

**适合场景**：
- 多格式文档混合处理
- 企业文档中心
- 通用 RAG 系统

**使用示例**：
```python
from unstructured.partition.auto import partition

# 自动识别格式并解析
elements = partition(filename="document.pdf")
text = "\n".join([str(el) for el in elements])
```

**学习成本**：⭐⭐（1-2小时）

---

### 3. LlamaParse ⭐⭐⭐⭐ LlamaIndex 生态

**链接**：https://github.com/run-llama/llama_parse

**核心优势**：
- ✅ LlamaIndex 官方工具
- ✅ 深度集成
- ✅ 解析质量高

**适合场景**：
- LlamaIndex 项目
- 复杂 PDF

**学习成本**：⭐⭐

---

### 4. PyPDF2 ⭐⭐⭐ 简单轻量

**链接**：https://github.com/py-pdf/pypdf2

**核心优势**：
- ✅ 纯 Python，无依赖
- ✅ API 简单
- ✅ 速度快

**适用场景**：
- 简单 PDF 文本提取
- 不需要保留格式
- 快速开发

**使用示例**：
```python
from PyPDF2 import PdfReader

reader = PdfReader("document.pdf")
text = "".join([page.extract_text() for page in reader.pages])
```

**学习成本**：⭐（10分钟）

---

## 🤔 如何选择？

### 决策树

```
你的文档类型？
│
├─ 简单 PDF（纯文本）
│   → PyPDF2（最快、最简单）
│
├─ 复杂 PDF（表格、公式、多栏）
│   ├─ 学术论文 → MinerU（最强）
│   └─ 通用文档 → Unstructured
│
├─ 多种格式混合
│   → Unstructured（支持最全）
│
└─ 用 LlamaIndex
    → LlamaParse（深度集成）
```

---

## 💡 解析优化技巧

### 常见问题与解决方案

**问题1：表格解析不准**
- 解决：使用 MinerU + 后处理脚本
- 或：Table Transformer 单独提取表格

**问题2：解析速度慢**
- 解决：批处理 + 多进程
- 或：先用 PyPDF2 快速提取，失败才用 MinerU

**问题3：格式丢失**
- 解决：保留原始格式标记
- 使用 Markdown 格式输出

---

## 📊 文档解析 Pipeline

**标准流程**：
```
PDF文件
  ↓
文档解析 (MinerU/Unstructured)
  ↓
文本清洗 (去除噪声、格式规范化)
  ↓
文档分块 (Chunking)
  ↓
向量化 (Embedding)
  ↓
向量库索引 (Milvus/FAISS)
```

---

## 🎯 面试高频问题

**Q: 如何处理复杂 PDF（表格、公式、多栏）？**

**标准答案**：
```
我使用 MinerU 处理复杂 PDF，流程是：

1. 【文档解析】
   - MinerU 识别文档结构（标题、段落、表格）
   - 保留布局信息
   
2. 【表格处理】
   - 单独提取表格
   - 转换为 Markdown 格式
   - 添加表格描述（Table Caption）
   
3. 【图片处理】
   - OCR 提取图片中的文字
   - 保存图片引用
   - 多模态 Embedding（CLIP）
   
4. 【质量保证】
   - 人工抽查 10% 样本
   - 自动化测试（提取完整度）
```

---

## 📝 相关文档

- [向量数据库选型](./vector-db.md)
- [Embedding 模型选择](./embedding.md)
- [返回 RAG 资源总览](./README.md)


