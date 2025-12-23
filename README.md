ChemGraph-RAG: 化工装置智能双库问答系统 🧪🤖
![alt text](https://img.shields.io/badge/python-3.12+-blue.svg)

![alt text](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)

![alt text](https://img.shields.io/badge/LLM-Ollama-white.svg)

![alt text](https://img.shields.io/badge/license-MIT-green.svg)

ChemGraph-RAG 是一个专为化工行业设计的 RAG（检索增强生成）系统。它采用了 GraphRAG 架构，通过结合 Neo4j 知识图谱 的确定性拓扑逻辑与 ChromaDB 向量数据库 的非结构化语义理解，解决了大模型在工业垂直领域中容易产生的“幻觉”问题。

🌟 核心特性
双库联合检索：同时查询 Neo4j（位号连接、流向事实）与 ChromaDB（工艺原理、操作手册）。
智能意图解析：自动识别用户是在进行“路径分析”、“故障诊断”、“状态查询”还是“操作规程查询”。
层级语境切片：独创 Markdown 面包屑注入算法，将文档层级（H1 > H2 > H3）自动编码进向量切片。
工业级知识围栏：严格限制模型仅基于事实回答，若双库无匹配数据则拒绝回答，确保安全生产。
完全本地部署：基于 Ollama 实现，支持 DeepSeek-R1、Llama3 等模型，数据不出内网。
🏗️ 技术架构
用户输入：通过 Streamlit 捕获。
NLP 解析：正则提取位号（Tag）+ LLM 分析意图。
Cypher 构建：根据意图自动生成 Neo4j 查询语句。
混合检索：
Graph 路：Neo4j 返回设备间的物理连接与设计参数。
Vector 路：ChromaDB 返回工艺描述与 SOP 片段。
知识融合：将双路证据打包送入 LLM 进行总结。
🚀 快速开始

1. 前置条件
Python 3.12 (推荐版本)
Ollama: 下载安装
Neo4j: 启动一个实例（支持 Docker 或 Desktop）
2. 下载模型
code
Bash
# 下载向量化模型
ollama pull nomic-embed-text

# 下载对话模型 (推荐使用 deepseek-r1)
ollama pull deepseek-r1:latest
3. 安装依赖
code
Bash
git clone https://github.com/your-username/chem-graph-rag.git
cd chem-graph-rag
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
4. 环境配置
创建 .env 文件并填入你的凭据：

code
Text
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的密码
📖 使用指南

数据组织建议
为了获得最佳的向量检索效果，建议将 Markdown 文档按以下结构组织：

code
Markdown
# 苯酐工段操作手册
## 一、氧化反应器 (D-14)
### 1. 反应原理
#### 1.1 催化氧化
这里是具体的描述文本...
运行系统
code
Bash
streamlit run app.py
在侧边栏输入 Markdown 文档所在的本地目录。
点击 “开始增量同步”，系统将自动进行清洗、切片并存入 ChromaDB。
在主界面输入位号相关的问题，如：“D-14 的熔盐温度异常升高该如何处理？”
🛠️ 意图分析逻辑
系统会根据输入自动匹配以下查询模式：

路径分析 (Path_Analysis): 自动计算两台设备间的最短物料流向路径。
故障诊断 (Fault_Diagnosis): 向上游追溯可能的物料来源或故障源。
状态检查 (Status_Check): 查询 Neo4j 中记录的设计压力、温度及关联仪表。
流程查询 (Procedure_Query): 检索对应的操作规程。
🤝 贡献建议
我们非常欢迎来自化工与 AI 领域的贡献！

提交 Bug 反馈或功能需求（Issue）。
提交更高效的 Cypher 查询模版（PR）。
优化 Markdown 切片算法。
⚠️ 免责声明
本项目仅供辅助决策参考。在实际化工装置操作中，请务必以工厂实时 DCS 系统数据及纸质操作规程（SOP）为准，任何 AI 生成的建议在实施前均需经过专业工程师审核。

📄 开源协议
基于 MIT License 开源。

作者: [陈峰]
联系方式: [snchenfeng@163.com]