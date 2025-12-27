import streamlit as st
import sys
from dotenv import load_dotenv
import os
import re
import json
import ollama
import chromadb
import frontmatter
from neo4j import GraphDatabase

# ================= 0. 加载环境变量 =================
load_dotenv()  # 这行代码会自动读取项目根目录下的 .env 文件

# ================= 1. 从环境变量获取配置 =================
# 使用 os.getenv('变量名', '默认值') 的方式读取
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # 密码不建议设默认值

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chemical_kb")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:latest")

# 获取本地数据路径
DEFAULT_DATA_PATH = os.getenv("DATA_PATH", "./data/")

# ================= 2. 初始化连接 =================
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ================= 3. 核心功能函数 =================
# --- 在这里插入【递归拆分辅助函数】 ---
def recursive_split_text(text, max_chars=1200, overlap=200):
    """当段落过长时，按优先级进行递归拆分"""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        if end < len(text):
            # 寻找最后一个换行或句号，避免生硬切断
            last_break = max(chunk.rfind('\n'), chunk.rfind('。'), chunk.rfind('. '))
            if last_break > max_chars * 0.5: 
                end = start + last_break + 1
                chunk = text[start:end]
        chunks.append(chunk)
        start += (len(chunk) - overlap)
        if len(chunk) <= overlap: break
    return chunks

def extract_tags(text):
    # 1. 匹配模式：
    # ([a-zA-Z]{1,3})  -> 捕获1-3位字母前缀（不区分大小写）
    # [-_]?            -> 匹配可选的连字符或下划线
    # (\d{2,4})        -> 捕获2-4位数字
    # ([a-zA-Z]?)      -> 捕获可选的一位字母后缀
    pattern = r'([a-zA-Z]{1,3})[\s\-_]?(\d{2,4})([a-zA-Z]?)'
    # 找到所有符合条件的组合
    matches = re.findall(pattern, text)
    
    normalized_tags = []
    for prefix, digits, suffix in matches:
        # 2. 标准化处理：
        # 全部转为大写，并在字母与数字之间强制加上 "-"
        # 例子：d43 -> D-43, D_43 -> D-43, d-43a -> D-43A
        standard_tag = f"{prefix.upper()}-{digits}{suffix.upper()}"
        normalized_tags.append(standard_tag)
    
    # 返回去重后的结果
    return list(set(normalized_tags))
def is_noise_node(node_labels, node_tag):
    """判断是否为需要忽略的辅助节点（如三通、跨页符、仪表）"""
    ignore_labels = ['OffPageConnector', 'Instrument', 'Drawing']
    ignore_tags = ['TEE', 'TappingPoint']
    # 检查标签或位号是否包含忽略关键词
    if any(l in node_labels for l in ignore_labels): return True
    if any(t in str(node_tag).upper() for t in ignore_tags): return True
    return False

def clean_markdown(content):
    content = re.sub(r'^\\---', '---', content, flags=re.MULTILINE)
    content = re.sub(r'^\\(#+)', r'\1', content, flags=re.MULTILINE)
    content = re.sub(r'[\u200B-\u200D\uFEFF]', '', content)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\|[\s*:|-]+\|$', line.strip()): continue 
        if line.strip().startswith('|'):
            line = line.strip('|').replace('|', '   ')
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def hierarchical_chunking(content, file_path):
    file_name = os.path.basename(file_path).replace('.md', '')
    post = frontmatter.loads(content)
    doc_metadata = post.metadata
    main_content = post.content
    doc_title = doc_metadata.get('title', file_name)
    final_chunks = []
    
    # 设定参数
    MAX_CHUNK_LEN = 1200 
    OVERLAP_LEN = 200

    h3_blocks = re.split(r'(?=^###\s+)', main_content, flags=re.MULTILINE)
    for i, h3_block in enumerate(h3_blocks):
        h3_block = h3_block.strip()
        if not h3_block: continue
        
        h3_match = re.search(r'^###\s+(.*)$', h3_block, flags=re.MULTILINE)
        h3_title = h3_match.group(1).strip() if h3_match else "概览"
        h3_content = re.sub(r'^###\s+.*$', '', h3_block, flags=re.MULTILINE).strip()
        
        if '#### ' in h3_content:
            h4_blocks = re.split(r'(?=^####\s+)', h3_content, flags=re.MULTILINE)
            for j, h4_block in enumerate(h4_blocks):
                h4_block = h4_block.strip()
                if len(h4_block) < 20: continue
                
                h4_match = re.search(r'^####\s+(.*)$', h4_block, flags=re.MULTILINE)
                h4_title = h4_match.group(1).strip() if h4_match else ""
                content_body = re.sub(r'^####\s+.*$', '', h4_block, flags=re.MULTILINE).strip()
                
                breadcrumb = f"{doc_title} > {h3_title}" + (f" > {h4_title}" if h4_title else "")
                
                # --- 调用递归拆分 ---
                sub_parts = recursive_split_text(content_body, MAX_CHUNK_LEN, OVERLAP_LEN)
                for k, part in enumerate(sub_parts):
                    final_chunks.append({
                        "id": f"{file_name}-{i}-{j}-p{k}",
                        "text": f"【语境：{breadcrumb}】\n{part}",
                        "metadata": {**doc_metadata, "breadcrumb": breadcrumb, "source": file_path}
                    })
        else:
            if len(h3_content) > 20:
                breadcrumb = f"{doc_title} > {h3_title}"
                # --- 调用递归拆分 ---
                sub_parts = recursive_split_text(h3_content, MAX_CHUNK_LEN, OVERLAP_LEN)
                for k, part in enumerate(sub_parts):
                    final_chunks.append({
                        "id": f"{file_name}-{i}-p{k}",
                        "text": f"【语境：{breadcrumb}】\n{part}",
                        "metadata": {**doc_metadata, "breadcrumb": breadcrumb, "source": file_path}
                    })
    return final_chunks

def analyze_intent_with_llm(prompt, extracted_tags):
    # 将模型提示词也改为中文，有助于模型更准确地按中文逻辑思考
    system_prompt = f"""你是一个工业专家级意图分析助手。请分析用户问题的意图并返回 JSON。
    
    意图分类标准：
    1. Path_Analysis: 询问工艺流程、物料流向、路径、经过哪些设备、跨页流程。
    2. Fault_Diagnosis: 询问故障原因、上游溯源、串料分析、异常波动来源。
    3. Status_Check: 询问设备设计参数（压力/温度/材质）、监控仪表位号、量程。
    4. Procedure_Query: 询问操作步骤、启动/停止顺序、安全注意事项、SOP。
    5. Info_Query: 询问基本定义、术语解释、通用常识。

    用户提取位号：{extracted_tags}
    返回格式：{{"intent": "意图名称", "start_node": "起点位号", "end_node": "终点位号", "target_tag": "目标位号"}}"""
    
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ], format='json')
        return json.loads(response['message']['content'])
    except:
        return {"intent": "Info_Query"}

def build_cypher(llm_result, extracted_tags, user_text):
    intent = llm_result.get("intent", "Info_Query")
    tags = extracted_tags
    
    # 路径分析强制修正
    if len(tags) >= 2 and any(k in user_text for k in ["到", "流", "经过", "去往", "流程"]):
        intent = "Path_Analysis"
        
    cypher = ""; params = {}

    if intent == "Path_Analysis":
        # ... (保持之前优化的跨页路径代码) ...
        start = llm_result.get("start_node") or (tags[0] if tags else None)
        end = llm_result.get("end_node") or (tags[1] if len(tags)>1 else None)
        if start and end:
            cypher = """
            MATCH (start:Asset), (end:Asset)
            WHERE (start.Tag STARTS WITH $startTag OR replace(start.Tag, '-', '') = $startTagAlt)
              AND (end.Tag STARTS WITH $endTag OR replace(end.Tag, '-', '') = $endTagAlt)
            MATCH path = shortestPath((start)-[:PIPE|LINKS_TO*..60]->(end))
            RETURN 'Path_Analysis' as intent,
                   [n in nodes(path) | {tag: n.Tag, desc: n.desc, labels: labels(n)}] as nodes_data,
                   [r in relationships(path) | properties(r)] as rels_data,
                   length(path) as total_hops
            """
            params = {"startTag": start, "startTagAlt": start.replace("-", ""), "endTag": end, "endTagAlt": end.replace("-", "")}

    elif intent == "Status_Check":
        # 【新增】状态检查：查询设备属性 + 关联仪表
        cypher = """
        UNWIND $tags AS qTag
        MATCH (e:Asset) WHERE e.Tag = qTag OR replace(e.Tag, '-', '') = replace(qTag, '-', '')
        OPTIONAL MATCH (i:Instrument)-[:MEASURES]->(e)
        RETURN 'Status_Check' as intent, e.Tag as tag, properties(e) as params,
               collect({tag: i.Tag, desc: i.desc, unit: i.unit, range: i.range}) as sensors
        """
        params = {"tags": tags}

    elif intent == "Fault_Diagnosis":
        # ... (保持之前优化的溯源代码) ...
        cypher = """
        UNWIND $tags AS qTag
        MATCH (target:Asset) WHERE target.Tag = qTag OR replace(target.Tag, '-', '') = replace(qTag, '-', '')
        MATCH path = (target)<-[:PIPE|LINKS_TO*1..6]-(source:Asset)
        WHERE NOT source:OffPageConnector AND source.Tag <> 'TEE'
        WITH target, source, relationships(path)[0] as r
        RETURN 'Fault_Diagnosis' as intent, target.Tag as tag,
               collect(DISTINCT {
                   source: source.Tag, 
                   s_desc: source.desc, 
                   fluid: r.fluid,
                   pipe_desc: r.desc  // <--- 新增：抓取管线描述
               }) as upstream_trace
        """
        params = {"tags": tags}

    elif intent == "Procedure_Query":
        # 【新增】规程查询：图数据库仅用于确认位号描述，主要靠向量库
        cypher = """
        UNWIND $tags AS qTag
        MATCH (e:Asset) WHERE e.Tag = qTag OR replace(e.Tag, '-', '') = replace(qTag, '-', '')
        RETURN 'Procedure_Query' as intent, e.Tag as tag, e.desc as desc
        """
        params = {"tags": tags}

    else: # Info_Query
        # ... (保持基础查询代码) ...
        cypher = """
        UNWIND $tags AS qTag
        MATCH (e:Asset) WHERE e.Tag = qTag OR replace(e.Tag, '-', '') = replace(qTag, '-', '')
        OPTIONAL MATCH (e)-[r:PIPE|CONTROLS|MEASURES]-(neighbor:Asset)
        RETURN 'Info_Query' as intent, e.Tag as tag, e.desc as desc, e.type as type,
               collect(DISTINCT {
                   type: type(r), 
                   neighbor: neighbor.Tag, 
                   n_desc: neighbor.desc,
                   fluid: r.fluid, 
                   rel_desc: r.desc,   // <--- 新增：抓取关系描述
                   from_reg: r.fromRegion, 
                   to_reg: r.toRegion
               }) as topology
        """
        params = {"tags": tags}

    return cypher, params

def query_neo4j(query, params):
    if not query: return []
    # 终端调试信息保留英文，方便排查
    print(f"\n[调试] 执行 Cypher: {query}\n[调试] 参数: {params}", file=sys.stderr, flush=True)
    try:
        with neo4j_driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    except Exception as e:
        print(f"[错误] Neo4j 查询失败: {e}", file=sys.stderr, flush=True)
        return []
# ==============================================================================
# 👇👇👇 请在这里插入新增的辅助函数 👇👇👇
# ==============================================================================

def translate_region(region_code):
    """将英文区域代码翻译为中文语义"""
    if not region_code: return "通用接口"
    mapping = {
        'ShellSide': '壳程',
        'ShellSide:Vapor': '壳程(气相)',
        'ShellSide:Liquid': '壳程(液相)',
        'TubeSide': '管程',
        'TubeSide:Liquid': '管程(液相)',
        'TubeSide:Vapor': '管程(气相)',
        'Jacket': '夹套',
        'InnerVessel': '内胆',
        'ControlSignal': '控制信号接口',
        'UpperSaltChannel': '上盐道',
        'LowerSaltChannel': '下盐道'
    }
    return mapping.get(region_code, region_code)

def format_graph_data(data, intent):
    """语义无损的路径压缩与属性聚合算法"""
    if not data: return "未查询到相关图谱事实。"
    text_lines = []

    # --- 场景 A: 状态检查 ---
    if intent == "Status_Check":
        for record in data:
            text_lines.append(f"📋 **设备参数档案**: {record['tag']}")
            params = record.get('params', {})
            for k, v in params.items():
                if k not in ['Tag', 'desc', 'id'] and v and v != "None":
                    text_lines.append(f"   - {k}: {v}")
            sensors = record.get('sensors', [])
            if sensors:
                text_lines.append(f"   **关联监控仪表**:")
                for s in sensors:
                    text_lines.append(f"   - 🏷️ {s['tag']} ({s['desc']}) | 量程: {s.get('range','--')} {s.get('unit','')}")
            text_lines.append("")

    # --- 场景 B: 路径分析 (核心聚合逻辑) ---
    elif intent == "Path_Analysis":
        for record in data:
            nodes = record.get('nodes_data', [])
            rels = record.get('rels_data', [])
            if not nodes: continue
            
            text_lines.append(f"🛣️ **全链路工艺追踪 (跨度: {record.get('total_hops', 0)} 步)**:")
            current_equip = nodes[0]
            attr_accumulator = {} 

            for i in range(len(rels)):
                rel = rels[i]
                next_node = nodes[i+1]
                # 累积管线属性
                for k, v in rel.items():
                    if v and v != "None": attr_accumulator[k] = v
                
                # 遇到真实设备才输出
                if not is_noise_node(next_node.get('labels', []), next_node.get('tag', '')):
                    src_str = f"**{current_equip['tag']}** ({current_equip.get('desc','设备')})"
                    tgt_str = f"**{next_node['tag']}** ({next_node.get('desc','设备')})"
                    
                    fluid = attr_accumulator.get('fluid', '未知介质')
                    dn = f"{attr_accumulator['dn']}" if attr_accumulator.get('dn') else ""
                    mat = attr_accumulator.get('material', '')
                    p_desc = attr_accumulator.get('desc', '') # 获取管线描述
                    
                    from_reg = translate_region(rels[i - (i if i==0 else 0)].get('fromRegion'))
                    to_reg = translate_region(rel.get('toRegion'))

                    # 【修正点】：将 p_desc 放入 join 列表中
                    pipe_detail = " | ".join(filter(None, [fluid, dn, mat, p_desc])) 
                    line = f"   📍 {src_str} `[{from_reg}]` ==( 🌊 {pipe_detail} )==> `[{to_reg}]` {tgt_str}"
                    text_lines.append(line)
                    
                    current_equip = next_node
                    attr_accumulator = {}
            text_lines.append("*(注：已自动合并跨页连接符及三通节点的物理属性)*\n")

    # --- 场景 C: 故障诊断 ---
    elif intent == "Fault_Diagnosis":
        for record in data:
            text_lines.append(f"🛠️ **故障溯源目标**: {record['tag']}")
            for trace in record.get('upstream_trace', []):
                # 【修正点】：增加管线描述展示
                p_desc = f"[{trace['pipe_desc']}]" if trace.get('pipe_desc') else ""
                text_lines.append(f"   ⬆️ 来源: **{trace['source']}**({trace.get('s_desc','设备')}) --({trace.get('fluid','介质')}{p_desc})--> 进入目标")

    # --- 场景 D: 规程查询 ---
    elif intent == "Procedure_Query":
        for record in data:
            text_lines.append(f"📖 **正在检索关于 {record['tag']}({record['desc']}) 的操作规程...**")

    # --- 场景 E: 基础查询 ---
    else:
        for record in data:
            text_lines.append(f"ℹ️ **设备档案**: {record.get('tag')} ({record.get('desc', '无描述')})")
            # 【修正点】：基础查询也应该展示周围的拓扑关系
            for t in record.get('topology', []):
                p_desc = f"[{t['rel_desc']}]" if t.get('rel_desc') else ""
                reg = translate_region(t.get('from_reg') or t.get('to_reg'))
                text_lines.append(f"   - 关联: **{t['neighbor']}**({t.get('n_desc','设备')}) | {t['fluid']} {p_desc} | 接口: {reg}")

    return "\n".join(text_lines)
# ==============================================================================
# 👆👆👆 插入结束 👆👆👆
# ============================================================================

# ================= 4. Streamlit 界面显示 (中文中文化) =================
st.set_page_config(page_title="化工知识图谱", layout="wide", page_icon="🧪")
st.title("🧪 化工装置图谱 + 文档向量混合知识库")

# --- 侧边栏：管理面板 ---
with st.sidebar:
    st.header("🛠️ 系统后台管理")
    
    # 显示向量库统计
    try:
        db_count = collection.count()
        st.metric("已存储知识切片数", db_count)
    except:
        st.error("向量数据库连接失败")
    
    st.markdown("---")
    
    # 知识同步功能
    st.subheader("📁 数据同步")
    path_input = st.text_area("Markdown 文档路径", "/Users/chenfeng/培训资料/化工苯酐基本知识/")
    
    if st.button("🚀 开始增量同步"):
        all_chunks = []
        with st.spinner("正在扫描并解析文档..."):
            if os.path.exists(path_input):
                for root, _, files in os.walk(path_input):
                    for file in files:
                        if file.endswith('.md'):
                            try:
                                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                    cleaned = clean_markdown(f.read())
                                    chunks = hierarchical_chunking(cleaned, os.path.join(root, file))
                                    all_chunks.extend(chunks)
                            except: pass
            
        if all_chunks:
            total = len(all_chunks)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, chunk in enumerate(all_chunks):
                status_text.text(f"正在向量化 ({idx+1}/{total}): {chunk['id']}")
                try:
                    emb = ollama.embeddings(model=EMBED_MODEL, prompt=chunk['text'][:2000])['embedding']
                    collection.upsert(
                        ids=[chunk['id']], 
                        embeddings=[emb], 
                        documents=[chunk['text']], 
                        metadatas=[{k:str(v) for k,v in chunk['metadata'].items()}]
                    )
                except: pass
                progress_bar.progress((idx + 1) / total)
            
            status_text.text("✅ 同步圆满完成！")
            st.balloons()
            st.rerun()
        else:
            st.warning("未在该路径下找到有效文档")

    if st.button("🗑️ 危险操作：清空向量库"):
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        st.warning("库已清空")
        st.rerun()

# --- 对话区域 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
# --- 对话区域 (优化版) ---
if prompt := st.chat_input("您可以问我：D-14 反应器的设计参数是什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. 检索阶段 (ChromaDB + Neo4j)
        graph_data = []
        vector_docs = []
        
        with st.status("🔍 正在检索双库事实...", expanded=True) as status:
            extracted_tags = extract_tags(prompt)
            # 【修正点 1】：先识别意图
            intent_res = analyze_intent_with_llm(prompt, extracted_tags)
            current_intent = intent_res.get('intent', 'Info_Query') # 获取真实意图
            st.write(f"🏷️ **识别位号**: `{', '.join(extracted_tags)}` | 🎯 **意图**: `{current_intent}`")
            
            # 【修正点 2】：根据真实意图决定检索文档数
            n_docs = 6 if current_intent == "Procedure_Query" else 3
            
            cypher, params = build_cypher(intent_res, extracted_tags, prompt)
            if cypher:
                graph_data = query_neo4j(cypher, params)
            
            q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=prompt)['embedding']
            vector_res = collection.query(query_embeddings=[q_emb], n_results=n_docs)
            vector_docs = vector_res['documents'][0]
            
            status.update(label=f"✅ 检索完成: {current_intent}", state="complete", expanded=False)

        # 2. 回答生成阶段
        full_response = ""
        
        # --- 【动态图标方案】创建一个化学/AI 动态占位符 ---
        thinking_container = st.empty()
        
        with thinking_container.container():
            st.markdown(
                """
                <style>
                @keyframes pulse-ring {
                  0% { transform: scale(.33); }
                  80%, 100% { opacity: 0; }
                }
                @keyframes pulse-dot {
                  0% { transform: scale(.8); }
                  50% { transform: scale(1); }
                  100% { transform: scale(.8); }
                }
                .ai-thinking-container {
                  display: flex;
                  flex-direction: column;
                  align-items: center;
                  justify-content: center;
                  padding: 20px;
                }
                .pulse-wrapper {
                  position: relative;
                  width: 60px;
                  height: 60px;
                }
                .pulse-dot {
                  position: absolute;
                  top: 15px; left: 15px;
                  width: 30px; height: 30px;
                  background-color: #007bff;
                  border-radius: 50%;
                  animation: pulse-dot 1.25s cubic-bezier(0.455, 0.03, 0.515, 0.955) -.4s infinite;
                  display: flex;
                  align-items: center;
                  justify-content: center;
                  z-index: 2;
                }
                .pulse-ring {
                  position: absolute;
                  top: 0; left: 0;
                  width: 60px; height: 60px;
                  background-color: #007bff;
                  border-radius: 50%;
                  animation: pulse-ring 1.25s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
                  z-index: 1;
                }
                .thinking-text {
                  margin-top: 15px;
                  font-family: sans-serif;
                  color: #007bff;
                  font-weight: bold;
                  letter-spacing: 1px;
                }
                </style>
                
                <div class="ai-thinking-container">
                    <div class="pulse-wrapper">
                        <div class="pulse-ring"></div>
                        <div class="pulse-dot">
                            <span style="font-size: 18px;">🧪</span>
                        </div>
                    </div>
                    <div class="thinking-text">努力分析中...</div>
                </div>
                """, 
                unsafe_allow_html=True
            )


        # 最终回答打字机显示的占位符
        response_placeholder = st.empty()
        
        if not graph_data and not vector_docs:
            thinking_container.empty() # 没找到数据，直接清除提示
            response_placeholder.warning("⚠️ 根据目前知识库记录，未找到与该提问相关的位号事实或文档说明。")
        else:
             # === [修改开始] 使用新的格式化函数 ===
            
            # 1. 将图数据转换为链式叙述文本
            current_intent = intent_res.get('intent', 'Info_Query')
            graph_text_narrative = format_graph_data(graph_data, current_intent)
            
            # 2. 构造更清晰的上下文
            h_context = f"""
【图谱事实 (物理拓扑与工艺语义)】:
{graph_text_narrative}

【知识库文档 (操作规程与原理)】:
{' '.join(vector_docs)}
            """
            # === [修改结束] ===
            
            # --- 提示词微调 (确保模型不会太啰嗦) ---
            intent_guidance = {
            "Path_Analysis": "当前任务是【工艺流程分析】，请重点描述设备的作用、物料流向、腔室切换及介质变化。",
            "Fault_Diagnosis": "当前任务是【故障诊断】，请分析上游可能的风险源。",
            "Status_Check": "当前任务是【状态检查】，请核对设备参数与仪表监控范围。",
            "Procedure_Query": "当前任务是【规程查询】，请详细说明操作步骤和安全要求。",
            "Info_Query": "当前任务是【信息查询】，请解释相关位号的功能定义。"
        }
        task_context = intent_guidance.get(current_intent, "")

            # 修改原有的 sys_prompt，在开头注入 task_context
        sys_prompt = f"""你是一位严谨的化工装置工艺工程师。
                
                ### 0. 当前任务重点
                {task_context}


                ### 1. 知识围栏 (Knowledge Guardrails) - 核心准则
                - **仅限上下文回答**：你只能根据下方提供的【图谱事实】和【文档资料】进行回答。严禁使用你自身训练数据中关于特定工厂、特定位号的外部知识。
                - **严禁推测连接**：如果【图谱事实】中没有显示 A 设备与 B 设备之间的路径，即使在常规工艺中它们通常相连，你也必须回答“当前图谱未记录 A 与 B 的直接连接”。
                - **诚实告知缺失**：如果用户询问的位号在【图谱事实】中不存在，或者询问的操作在【文档资料】中未提及，请明确回答：“根据现有知识库记录，无法提供关于 [位号/操作] 的信息”。
                - **禁止幻觉补全**：严禁为了使流程完整而自行补全中间的阀门、管段或仪表。

                ### 2. 证据溯源要求
                - 你的每一句关键结论都应暗示其来源。
                - 涉及物理连接、介质、材质、腔室逻辑时，请表述为：“根据图谱拓扑记录...”。
                - 涉及操作步骤、安全要求、工艺原理时，请表述为：“根据操作规程记载...”。

                ### 3. 物理语义约束
                - 必须尊重腔室逻辑：明确区分壳程(ShellSide)、管程(TubeSide)、夹套(Jacket)。如果物料流向了错误的腔室，请在回答中作为潜在风险点指出。

                ### 4. 回答风格
                - 风格：极其专业、冷峻、客观。
                - 结构：
                1. 【核心结论】：一句话直接回答问题，包括设备的位号和名称。
                2.  根据任务重点详细讲解。
                3. 【安全提醒】：（如有）基于事实的风险告知。

                ### 5. 负面约束 (Negative Constraints)
                - 绝对禁止使用： “我猜”、“通常情况下”、“经验表明”、“可能”。
                - 绝对禁止回答： 与当前装置无关的通用化工常识（除非用户明确询问定义）。
                """

        try:
                # 调用模型
                stream = ollama.chat(model=LLM_MODEL, messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': f"事实背景: {h_context}\n问题: {prompt}"}
                ], stream=True)

                # --- 核心改进：精确控制清除时机 ---
                for chunk in stream:
                    content = chunk['message']['content']
                    
                    # 只有当模型真正输出了内容（且非空）时，才清除“思考中”提示
                    if content.strip() and not full_response:
                        thinking_container.empty()
                    
                    full_response += content
                    # 动态展示打字机效果
                    response_placeholder.markdown(full_response + "▌")
                
                # 完成输出，移除光标
                response_placeholder.markdown(full_response)
                
        except Exception as e:
                thinking_container.empty()
                st.error(f"❌ 模型生成失败: {e}")

        # 3. 证据溯源显示 (优化版)
        if graph_data or vector_docs:
            with st.expander("🔍 原始检索证据"):
                tab1, tab2 = st.tabs(["图谱事实 (链式叙述)", "文档片段"])
                
                with tab1:
                    # === [修改] 使用 format_graph_data 渲染 ===
                    if graph_data:
                        # 复用之前计算好的 intent
                        current_intent = intent_res.get('intent', 'Info_Query')
                        formatted_text = format_graph_data(graph_data, current_intent)
                        st.markdown(formatted_text)
                    else:
                        st.info("无图谱数据")
                
                with tab2:
                    if vector_docs:
                        for i, d in enumerate(vector_docs):
                            st.info(f"**片段 {i+1}**:\n{d}")
                    else:
                        st.info("无文档数据")

    # 将助手的回答存入对话历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})