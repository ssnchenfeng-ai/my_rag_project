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
    system_prompt = f"""你是一个工业意图分析助手。请分析用户问题的意图并返回 JSON。
    可选意图：
    - Path_Analysis: 询问物料流向、路径、经过哪里。
    - Fault_Diagnosis: 询问故障原因、上游溯源。
    - Status_Check: 询问设备设计参数、监控仪表。
    - Procedure_Query: 询问操作步骤、熟化流程。
    - Info_Query: 询问基本定义或通用信息。

    用户已提取位号：{extracted_tags}
    返回格式：{{"intent": "意图名称", "start_node": "起点位号", "end_node": "终点位号", "target_name": "设备名称"}}"""
    try:
        # 这里建议统一使用一个能聊天的模型
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
    
    # 强制修正：如果涉及两个位号且包含流程动词，设为路径分析
    if len(tags) >= 2 and any(k in user_text for k in ["到", "流", "经过", "去往", "流程", "联系", "工艺"]):
        intent = "Path_Analysis"
        
    cypher = ""; params = {}

    if intent == "Path_Analysis":
        start = llm_result.get("start_node") or (tags[0] if tags else None)
        end = llm_result.get("end_node") or (tags[1] if len(tags)>1 else None)
        
        if start and end:
            # 路径分析：返回极其详尽的节点和管道属性
            cypher = """
            MATCH (start:Asset), (end:Asset)
    WHERE (start.Tag STARTS WITH $startTag OR replace(start.Tag, '-', '') = $startTagAlt)
      AND (end.Tag STARTS WITH $endTag OR replace(end.Tag, '-', '') = $endTagAlt)
    
    // 1. 寻找顺流方向的最短路径
    MATCH path = shortestPath((start)-[:PIPE|MEASURES*..30]->(end))
    
    // 2. 格式化返回：保留所有物理语义并按工艺顺序交织
    RETURN 
        'Path_Analysis' as intent,
        [i IN range(0, length(path)-1) | {
            // 起点设备
            from_equipment: CASE 
                WHEN nodes(path)[i].Tag <> "TEE" AND nodes(path)[i].type <> "Instrument" AND nodes(path)[i].type <> "TappingPoint"
                THEN {tag: nodes(path)[i].Tag, desc: nodes(path)[i].desc, type: nodes(path)[i].type}
                ELSE "辅助连接点(TEE/测点)" 
            END,
            
            // 管道语义（12项完整属性）
            pipeline_semantics: {
                fluid: relationships(path)[i].fluid,
                dn: relationships(path)[i].dn,
                material: relationships(path)[i].material,
                insulation: relationships(path)[i].insulation,
                pn: relationships(path)[i].pn,
                fromPort: relationships(path)[i].fromPort,
                toPort: relationships(path)[i].toPort,
                fromDesc: relationships(path)[i].fromDesc,
                toDesc: relationships(path)[i].toDesc,
                fromRegion: relationships(path)[i].fromRegion,
                toRegion: relationships(path)[i].toRegion,
                tag: relationships(path)[i].tag
            },
            
            // 终点设备
            to_equipment: CASE 
                WHEN nodes(path)[i+1].Tag <> "TEE" AND nodes(path)[i+1].type <> "Instrument" AND nodes(path)[i+1].type <> "TappingPoint"
                THEN {tag: nodes(path)[i+1].Tag, desc: nodes(path)[i+1].desc, type: nodes(path)[i+1].type}
                ELSE "辅助连接点(TEE/测点)" 
            END
        }] as structured_process_flow,
        length(path) as total_hops
            """
            params = {
                "startTag": start, "startTagAlt": start.replace("-", ""),
                "endTag": end, "endTagAlt": end.replace("-", "")
            }

    elif intent == "Fault_Diagnosis":
        # 故障诊断：侧重于追溯上游设备及其描述
        cypher = """
        UNWIND $tags AS qTag
        
        // 1. 模糊匹配找到目标设备
        MATCH (target:Asset) 
        WHERE target.Tag = qTag OR replace(target.Tag, '-', '') = replace(qTag, '-', '')
        
        // 2. 查找上游路径 (使用 path 变量捕获完整拓扑)
        // 这里查找 1 到 3 跳的上游设备，排除 TEE (三通) 这种无意义节点作为终点，但保留路径中的关系
        MATCH path = (target)<-[:PIPE*1..3]-(source:Asset)
        WHERE source.Tag <> 'TEE'
        
        // 3. 展开路径中的每一段关系 (Relationship)
        UNWIND relationships(path) AS r
        
        // 4. 提取关系的起点(start)和终点(end)
        // 注意：虽然我们是往上游查，但物理流向依然是 start -> end
        WITH target, startNode(r) AS start, endNode(r) AS end, r
        
        // 5. 返回结构化的拓扑数据
        RETURN target.Tag as tag, 
               'Fault_Diagnosis' as intent, 
               collect(DISTINCT {
                   // 连线起点 (上游)
                   source: start.Tag,
                   source_type: start.type,
                   source_desc: start.desc,
                   
                   // 连线终点 (下游)
                   target: end.Tag,
                   target_type: end.type,
                   
                   // === 核心物理语义 (AI 诊断的关键) ===
                   fluid: r.fluid,           // 介质 (如: Steam, Water)
                   from_region: r.fromRegion,// 起点区域 (如: ShellSide)
                   to_region: r.toRegion,    // 终点区域 (如: TubeSide) -> 诊断串料/干烧的关键
                   insulation: r.insulation  // 保温/伴热 -> 诊断冻结/结晶的关键
               }) as upstream_trace
        """
        params = {"tags": tags}

    elif intent == "Status_Check":
        # 仪表检查：侧重于 MEASURES 关系和 Instrument 节点的参数
        cypher = """
        UNWIND $tags AS qTag
        MATCH (target:Asset) WHERE target.Tag = qTag OR replace(target.Tag, '-', '') = replace(qTag, '-', '')
        OPTIONAL MATCH (target)-[:MEASURES]-(sensor:Instrument)
        RETURN target.Tag as tag, target.desc as desc, 
               {temp: target.design_temp, press: target.design_press, spec: target.spec} as design_params,
               collect(DISTINCT {tag: sensor.Tag, desc: sensor.desc, range: sensor.range, unit: sensor.unit}) as sensors
        """
        params = {"tags": tags}

    else:
        # 基础查询：返回位号、描述和类型
        cypher = """
         UNWIND $tags AS qTag
        
        // 1. 模糊匹配找到中心设备
        MATCH (center:Asset) 
        WHERE center.Tag = qTag OR replace(center.Tag, '-', '') = replace(qTag, '-', '')
        
        // 2. 双向扩展：查找距离中心设备 1 到 3 跳的所有路径
        // 注意这里没有箭头，表示双向查找 (Upstream & Downstream)
        // 包含 PIPE (管线), CONTROLS (控制), MEASURES (测量)
        MATCH path = (center)-[:PIPE|CONTROLS|MEASURES*1..3]-(neighbor:Asset)
        
        // 3. 展开路径中的每一段关系
        UNWIND relationships(path) AS r
        
        // 4. 提取物理流向 (无论查询方向如何，startNode->endNode 永远代表物理流向)
        WITH center, startNode(r) AS source, endNode(r) AS target, r, type(r) as relType
        
        // 5. 过滤掉无意义的纯连接节点 (如 TEE)，除非它是路径的中间环节
        // (这里选择保留 TEE 的连接关系，但在展示时由前端或 LLM 决定是否忽略)
        
        // 6. 返回去重后的拓扑结构
        RETURN center.Tag as tag, 
               'Info_Query' as intent,
               // 汇总该设备周围的所有属性
               {
                   type: center.type,
                   desc: center.desc,
                   spec: center.spec,
                   material: center.material
               } as self_info,
               collect(DISTINCT {
                   // 关系类型 (PIPE/CONTROLS/MEASURES)
                   type: relType,
                   
                   // 起点 (流出方)
                   source: source.Tag,
                   source_type: source.type,
                   
                   // 终点 (流入方)
                   target: target.Tag,
                   target_type: target.type,
                   
                   // 物理语义细节
                   fluid: r.fluid,
                   from_region: r.fromRegion, // 关键：从哪个腔室出来
                   to_region: r.toRegion,     // 关键：进哪个腔室
                   tag: r.tag                 // 管段号
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
    """
    将 Neo4j 返回的 JSON 列表转换为 LLM 友好的链式叙述文本
    增强版：明确标注了来源端口(fromRegion)和到达端口(toRegion)
    """
    if not data:
        return "未查询到相关图谱数据。"
    
    text_lines = []
    
    # === 场景 1: 路径分析 (Path_Analysis) ===
    if intent == "Path_Analysis":
        for path_idx, record in enumerate(data):
            text_lines.append(f"🛣️ **物理路径 #{path_idx + 1} (总跳数: {record.get('total_hops', 0)})**:")
            steps = record.get('structured_process_flow', [])
            
            for i, step in enumerate(steps):
                # 1. 提取起点及来源端口
                src = step['from_equipment']
                pipe = step['pipeline_semantics']
                
                src_tag = src['tag'] if isinstance(src, dict) else src
                src_desc = f"({src['desc']})" if isinstance(src, dict) and src.get('desc') else ""
                from_reg = translate_region(pipe.get('fromRegion')) # 新增：来源端口
                
                # 格式化起点：🏭 设备 (描述) [出口: 壳程]
                src_str = f"🏭 **{src_tag}**{src_desc}"
                if from_reg != "通用接口":
                    src_str += f" `[出口: {from_reg}]`"
                
                # 2. 管道/关系语义
                fluid = pipe.get('fluid', '未知介质')
                p_tag = pipe.get('tag') or '无管号'
                insulation = pipe.get('insulation', 'None')
                conn_desc = f" ==( 🌊{fluid} | 🏷️{p_tag}"
                if insulation != 'None': conn_desc += f" | 🔥{insulation}"
                conn_desc += " )==> "
                
                # 3. 提取终点及进入端口
                tgt = step['to_equipment']
                tgt_tag = tgt['tag'] if isinstance(tgt, dict) else tgt
                tgt_desc = f"({tgt['desc']})" if isinstance(tgt, dict) and tgt.get('desc') else ""
                to_reg = translate_region(pipe.get('toRegion')) # 保持：进入端口
                
                # 格式化终点：[入口: 管程] 🏭 设备 (描述)
                tgt_str = f"**{tgt_tag}**{tgt_desc}"
                if to_reg != "通用接口":
                    tgt_str = f"`[入口: {to_reg}]` 🏭 {tgt_str}"
                else:
                    tgt_str = f"🏭 {tgt_str}"
                
                text_lines.append(f"   {i+1}. {src_str}{conn_desc}{tgt_str}")
            text_lines.append("") 

    # === 场景 2: 故障诊断 (Fault_Diagnosis) ===
    elif intent == "Fault_Diagnosis":
        for record in data:
            target_tag = record.get('tag')
            text_lines.append(f"🛠️ **目标设备**: {target_tag}")
            text_lines.append("   **上游溯源 (Upstream Trace):**")
            
            traces = record.get('upstream_trace', [])
            for trace in traces:
                source_tag = trace.get('source')
                from_reg = translate_region(trace.get('from_region')) # 新增：来源端口
                to_reg = translate_region(trace.get('to_region'))     # 保持：进入端口
                fluid = trace.get('fluid', 'Unknown')
                
                # 增强版诊断语义：[来源设备][出口接口] --(介质)--> [目标设备][入口接口]
                line = f"   ⬆️ 来源: **{source_tag}** `[{from_reg}]` "
                line += f" --输送: {fluid}--> "
                line += f"进入目标设备的 **[{to_reg}]**"
                text_lines.append(line)
            text_lines.append("")

    # === 场景 3: 信息查询 (Info_Query) ===
    elif intent == "Info_Query":
        for record in data:
            self_info = record.get('self_info', {})
            text_lines.append(f"ℹ️ **设备档案**: {record.get('tag')}")
            text_lines.append(f"   **详细拓扑 (Topology Detail):**")
            
            topo = record.get('topology', [])
            for t in topo:
                # 识别当前设备是起点还是终点
                is_source = (t.get('source') == record.get('tag'))
                neighbor = t.get('target') if is_source else t.get('source')
                direction = "➡️ 流出至" if is_source else "⬅️ 接收来自"
                
                # 关键：同时展示本端接口和对端接口
                local_reg = translate_region(t.get('from_region') if is_source else t.get('to_region'))
                fluid = t.get('fluid', 'N/A')
                
                line = f"   - {direction} **{neighbor}** (介质: {fluid} | 本端接口: {local_reg})"
                text_lines.append(line)
            text_lines.append("")

    else:
        text_lines.append(json.dumps(data, ensure_ascii=False, indent=2))

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
            st.write(f"🏷️ **识别位号**: `{', '.join(extracted_tags) if extracted_tags else '未识别'}`")
            
            intent_res = analyze_intent_with_llm(prompt, extracted_tags)
            st.write(f"🎯 **解析意图**: `{intent_res.get('intent', 'Info_Query')}`")
            
            cypher, params = build_cypher(intent_res, extracted_tags, prompt)
            if cypher:
                graph_data = query_neo4j(cypher, params)
                # --- 这里的逻辑改为条件显示 ---
                if graph_data:
                        st.write("✅ **图谱事实**: 已成功检索到关联拓扑")
                else:
                     st.write("⚠️ **图谱事实**: 未能在图数据库中找到匹配的路径或节点")
            
            q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=prompt)['embedding']
            vector_res = collection.query(query_embeddings=[q_emb], n_results=3)
            vector_docs = vector_res['documents'][0]
            st.write(f"📄 **文档知识**: 已匹配相关描述片段")
            
            status.update(label=f"✅ 检索完成: 命中 {len(graph_data)} 条事实, {len(vector_docs)} 段文档", state="complete", expanded=False)

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
            sys_prompt = f"""你是一个专业的化工装置专家。请结合【图谱事实】和【文档资料】回答用户的【问题】。如果【图谱事实】和【知识库文档】中没有足够的信息，就直接说'根据我现有的知识，无法回答这个问题'，不要编造答案。
            
             
            ### 回答策略
                        1. **综合判断**: 图谱提供了准确的设备位号功能描述和连接关系的来源去向等，知识库提供了详细的操作步骤和原理。
                        2 . **故障诊断**: 如果图谱显示多条供料支路，请分别分析。结合知识库中的故障处理方法。
                        3. **冲突处理**: 涉及设备连接关系时，以图谱为准；涉及操作细节时，以知识库为准。

            
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