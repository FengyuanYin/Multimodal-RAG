"""
图存储模块
=========
支持 NetworkX（内存）和 Neo4j（生产）两种后端。
负责知识图谱的构建、查询和社区检测。
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class GraphEntity:
    """图实体节点"""
    id: str
    name: str
    type: str = "concept"  # person | organization | location | concept | event
    properties: dict = field(default_factory=dict)


@dataclass
class GraphRelation:
    """图关系边"""
    source_id: str
    target_id: str
    relation_type: str = "related_to"
    properties: dict = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class GraphCommunity:
    """图社区（聚类结果）"""
    community_id: str
    entities: List[str]
    summary: str = ""
    metadata: dict = field(default_factory=dict)


class BaseGraphStore:
    """图存储基类"""

    def add_entity(self, entity: GraphEntity) -> bool:
        raise NotImplementedError

    def add_relation(self, relation: GraphRelation) -> bool:
        raise NotImplementedError

    def add_entities(self, entities: List[GraphEntity]) -> int:
        count = 0
        for e in entities:
            if self.add_entity(e):
                count += 1
        return count

    def add_relations(self, relations: List[GraphRelation]) -> int:
        count = 0
        for r in relations:
            if self.add_relation(r):
                count += 1
        return count

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        raise NotImplementedError

    def search_entity(self, name: str, type: Optional[str] = None) -> List[GraphEntity]:
        raise NotImplementedError

    def get_neighbors(self, entity_id: str, max_depth: int = 2) -> List[Tuple[GraphEntity, GraphRelation]]:
        raise NotImplementedError

    def detect_communities(self) -> List[GraphCommunity]:
        raise NotImplementedError

    def get_community_summary(self, community_id: str) -> Optional[str]:
        raise NotImplementedError

    def clear(self) -> bool:
        raise NotImplementedError

    # ── 媒体引用图（RAG-Anything 风格：文本块 → 图片/表格） ──

    def add_chunk_node(self, chunk_id: str, doc_id: str, content_preview: str = "") -> bool:
        """添加文本块节点（type=chunk）"""
        raise NotImplementedError

    def add_media_node(self, media_id: str, media_type: str, doc_id: str = "",
                       page: int = 1, label: str = "", caption: str = "",
                       properties: Optional[dict] = None) -> bool:
        """添加媒体节点（type=media，图片/表格）"""
        raise NotImplementedError

    def add_reference(self, chunk_id: str, media_id: str, label: str = "",
                      page: int = 1, offset: int = 0, media_type: str = "image") -> bool:
        """添加引用边：chunk --references--> media（记录引用位置）"""
        raise NotImplementedError

    def get_media_by_chunk(self, chunk_id: str) -> List[dict]:
        """根据文本块 ID 返回其引用的媒体元数据列表（不含二进制数据）"""
        raise NotImplementedError

    def get_media_node(self, media_id: str) -> Optional[dict]:
        """获取媒体节点元数据"""
        raise NotImplementedError

    def list_media(self) -> List[dict]:
        """列出全部媒体节点元数据"""
        raise NotImplementedError


class NetworkXStore(BaseGraphStore):
    """NetworkX 内存图存储"""

    def __init__(self):
        self._graph = None
        self._init_graph()

    def _init_graph(self):
        import networkx as nx
        self._graph = nx.MultiDiGraph()
        logger.info("NetworkX 图存储初始化完成")

    def add_entity(self, entity: GraphEntity) -> bool:
        if not self._graph.has_node(entity.id):
            self._graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                properties=entity.properties,
            )
            return True
        return False

    def add_relation(self, relation: GraphRelation) -> bool:
        if not self._graph.has_node(relation.source_id):
            logger.warning(f"源节点不存在: {relation.source_id}")
            return False
        if not self._graph.has_node(relation.target_id):
            logger.warning(f"目标节点不存在: {relation.target_id}")
            return False
        self._graph.add_edge(
            relation.source_id,
            relation.target_id,
            key=f"{relation.source_id}->{relation.target_id}_{relation.relation_type}",
            relation_type=relation.relation_type,
            properties=relation.properties,
            weight=relation.weight,
        )
        return True

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        if not self._graph.has_node(entity_id):
            return None
        data = self._graph.nodes[entity_id]
        return GraphEntity(
            id=entity_id,
            name=data.get("name", ""),
            type=data.get("type", "concept"),
            properties=data.get("properties", {}),
        )

    def search_entity(self, name: str, type: Optional[str] = None) -> List[GraphEntity]:
        results = []
        for node_id, data in self._graph.nodes(data=True):
            if type and data.get("type") != type:
                continue
            if name.lower() in data.get("name", "").lower():
                results.append(GraphEntity(
                    id=node_id,
                    name=data["name"],
                    type=data.get("type", "concept"),
                    properties=data.get("properties", {}),
                ))
        return results

    def get_neighbors(self, entity_id: str, max_depth: int = 2) -> List[Tuple[GraphEntity, GraphRelation]]:
        if not self._graph.has_node(entity_id):
            return []

        from collections import deque
        visited = {entity_id}
        queue = deque([(entity_id, 0)])
        results = []

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for neighbor_id in self._graph.successors(current_id):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    edge_data = self._graph.get_edge_data(current_id, neighbor_id)
                    if edge_data:
                        first_key = list(edge_data.keys())[0]
                        rel_data = edge_data[first_key]
                        neighbor_entity = self.get_entity(neighbor_id)
                        if neighbor_entity:
                            results.append((
                                neighbor_entity,
                                GraphRelation(
                                    source_id=current_id,
                                    target_id=neighbor_id,
                                    relation_type=rel_data.get("relation_type", "related_to"),
                                    properties=rel_data.get("properties", {}),
                                    weight=rel_data.get("weight", 1.0),
                                ),
                            ))
                    queue.append((neighbor_id, depth + 1))

            for neighbor_id in self._graph.predecessors(current_id):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    edge_data = self._graph.get_edge_data(neighbor_id, current_id)
                    if edge_data:
                        first_key = list(edge_data.keys())[0]
                        rel_data = edge_data[first_key]
                        neighbor_entity = self.get_entity(neighbor_id)
                        if neighbor_entity:
                            results.append((
                                neighbor_entity,
                                GraphRelation(
                                    source_id=neighbor_id,
                                    target_id=current_id,
                                    relation_type=rel_data.get("relation_type", "related_to"),
                                    properties=rel_data.get("properties", {}),
                                    weight=rel_data.get("weight", 1.0),
                                ),
                            ))
                    queue.append((neighbor_id, depth + 1))

        return results

    def detect_communities(self) -> List[GraphCommunity]:
        try:
            # 优先使用内置 Louvain 算法（networkx>=3.0）
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(self._graph.to_undirected(), seed=42)
            result = []
            for i, community in enumerate(communities):
                result.append(GraphCommunity(
                    community_id=f"community_{i:04d}",
                    entities=list(community),
                    summary=f"社区 {i}: {len(community)} 个实体",
                ))
            logger.info(f"社区检测完成: {len(result)} 个社区")
            return result
        except Exception:
            try:
                # 降级：使用贪心模块度
                from networkx.algorithms.community import greedy_modularity_communities
                communities = greedy_modularity_communities(self._graph.to_undirected())
                result = []
                for i, community in enumerate(communities):
                    result.append(GraphCommunity(
                        community_id=f"community_{i:04d}",
                        entities=list(community),
                        summary=f"社区 {i}: {len(community)} 个实体",
                    ))
                return result
            except Exception as e:
                logger.warning(f"社区检测失败（降级方案不可用）: {e}")
                return []

    def get_community_summary(self, community_id: str) -> Optional[str]:
        # 简单实现：返回社区中所有实体的名称列表
        try:
            from networkx.algorithms.community import louvain_communities
            communities = list(louvain_communities(self._graph.to_undirected(), seed=42))
            idx = int(community_id.split("_")[-1])
            if idx < len(communities):
                entities = [self._graph.nodes[n].get("name", n) for n in communities[idx]]
                return f"社区包含: {', '.join(entities)}"
        except Exception:
            pass
        return None

    def clear(self) -> bool:
        self._init_graph()
        return True

    # ── 媒体引用图（RAG-Anything 风格） ──

    @staticmethod
    def _chunk_node_id(chunk_id: str) -> str:
        return f"chunk:{chunk_id}"

    @staticmethod
    def _media_node_id(media_id: str) -> str:
        return f"media:{media_id}"

    def add_chunk_node(self, chunk_id: str, doc_id: str, content_preview: str = "") -> bool:
        node_id = self._chunk_node_id(chunk_id)
        if not self._graph.has_node(node_id):
            self._graph.add_node(
                node_id,
                name=chunk_id,
                type="chunk",
                properties={"doc_id": doc_id, "content_preview": content_preview[:200]},
            )
            return True
        return False

    def add_media_node(self, media_id: str, media_type: str, doc_id: str = "",
                       page: int = 1, label: str = "", caption: str = "",
                       properties: Optional[dict] = None) -> bool:
        node_id = self._media_node_id(media_id)
        if not self._graph.has_node(node_id):
            self._graph.add_node(
                node_id,
                name=label or media_id,
                type="media",
                properties={
                    "media_id": media_id,
                    "media_type": media_type,
                    "doc_id": doc_id,
                    "page": page,
                    "label": label,
                    "caption": caption,
                    **(properties or {}),
                },
            )
            return True
        return False

    def add_reference(self, chunk_id: str, media_id: str, label: str = "",
                      page: int = 1, offset: int = 0, media_type: str = "image") -> bool:
        chunk_node = self._chunk_node_id(chunk_id)
        media_node = self._media_node_id(media_id)
        if not self._graph.has_node(chunk_node):
            self._graph.add_node(chunk_node, name=chunk_id, type="chunk", properties={})
        if not self._graph.has_node(media_node):
            self._graph.add_node(media_node, name=media_id, type="media",
                                 properties={"media_id": media_id, "media_type": media_type,
                                             "page": page, "label": label})
        self._graph.add_edge(
            chunk_node,
            media_node,
            key=f"{chunk_node}->{media_node}_references",
            relation_type="references",
            properties={"label": label, "page": page, "offset": offset, "media_type": media_type},
            weight=1.0,
        )
        return True

    def get_media_by_chunk(self, chunk_id: str) -> List[dict]:
        chunk_node = self._chunk_node_id(chunk_id)
        if not self._graph.has_node(chunk_node):
            return []
        results = []
        for media_node in self._graph.successors(chunk_node):
            data = self._graph.nodes[media_node]
            if data.get("type") != "media":
                continue
            edge_data = self._graph.get_edge_data(chunk_node, media_node)
            ref_props = {}
            if edge_data:
                first_key = list(edge_data.keys())[0]
                ref_props = edge_data[first_key].get("properties", {})
            results.append({
                "media_id": data.get("properties", {}).get("media_id", media_node),
                "media_type": data.get("properties", {}).get("media_type", "image"),
                "doc_id": data.get("properties", {}).get("doc_id", ""),
                "page": data.get("properties", {}).get("page", 1),
                "label": data.get("properties", {}).get("label", ""),
                "caption": data.get("properties", {}).get("caption", ""),
                "ref": ref_props,
            })
        return results

    def get_media_node(self, media_id: str) -> Optional[dict]:
        node_id = self._media_node_id(media_id)
        if not self._graph.has_node(node_id):
            return None
        data = self._graph.nodes[node_id]
        return {
            "media_id": data.get("properties", {}).get("media_id", media_id),
            "media_type": data.get("properties", {}).get("media_type", "image"),
            "doc_id": data.get("properties", {}).get("doc_id", ""),
            "page": data.get("properties", {}).get("page", 1),
            "label": data.get("properties", {}).get("label", ""),
            "caption": data.get("properties", {}).get("caption", ""),
        }

    def list_media(self) -> List[dict]:
        results = []
        for node_id, data in self._graph.nodes(data=True):
            if data.get("type") != "media":
                continue
            props = data.get("properties", {})
            results.append({
                "media_id": props.get("media_id", node_id),
                "media_type": props.get("media_type", "image"),
                "doc_id": props.get("doc_id", ""),
                "page": props.get("page", 1),
                "label": props.get("label", ""),
                "caption": props.get("caption", ""),
            })
        return results

    @property
    def stats(self) -> dict:
        import networkx as nx
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph),
        }


class Neo4jStore(BaseGraphStore):
    """Neo4j 图存储（生产环境）"""

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._init_driver()

    def _init_driver(self):
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Neo4j 连接成功: {self.uri}")
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            raise

    def _run_query(self, query: str, params: Optional[dict] = None) -> list:
        with self._driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def add_entity(self, entity: GraphEntity) -> bool:
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name, e.type = $type, e.properties = $properties
        RETURN e
        """
        self._run_query(query, {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "properties": entity.properties,
        })
        return True

    def add_relation(self, relation: GraphRelation) -> bool:
        query = """
        MATCH (s:Entity {id: $source_id})
        MATCH (t:Entity {id: $target_id})
        MERGE (s)-[r:RELATED {type: $relation_type}]->(t)
        SET r.weight = $weight, r.properties = $properties
        RETURN r
        """
        self._run_query(query, {
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "relation_type": relation.relation_type,
            "weight": relation.weight,
            "properties": relation.properties,
        })
        return True

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        query = "MATCH (e:Entity {id: $id}) RETURN e"
        results = self._run_query(query, {"id": entity_id})
        if not results:
            return None
        e = results[0]["e"]
        return GraphEntity(
            id=e["id"],
            name=e["name"],
            type=e.get("type", "concept"),
            properties=e.get("properties", {}),
        )

    def search_entity(self, name: str, type: Optional[str] = None) -> List[GraphEntity]:
        if type:
            query = "MATCH (e:Entity) WHERE e.name CONTAINS $name AND e.type = $type RETURN e"
            results = self._run_query(query, {"name": name, "type": type})
        else:
            query = "MATCH (e:Entity) WHERE e.name CONTAINS $name RETURN e"
            results = self._run_query(query, {"name": name})
        return [
            GraphEntity(id=r["e"]["id"], name=r["e"]["name"],
                        type=r["e"].get("type", "concept"),
                        properties=r["e"].get("properties", {}))
            for r in results
        ]

    def get_neighbors(self, entity_id: str, max_depth: int = 2) -> List[Tuple[GraphEntity, GraphRelation]]:
        query = f"""
        MATCH (s:Entity {{id: $id}})-[r*1..{max_depth}]-(n:Entity)
        RETURN DISTINCT n, last(r) as rel
        LIMIT 100
        """
        results = self._run_query(query, {"id": entity_id})
        neighbors = []
        for r in results:
            n = r["n"]
            rel = r.get("rel")
            neighbors.append((
                GraphEntity(id=n["id"], name=n["name"],
                           type=n.get("type", "concept"),
                           properties=n.get("properties", {})),
                GraphRelation(
                    source_id=entity_id,
                    target_id=n["id"],
                    relation_type=rel.get("type", "related_to") if rel else "related_to",
                    properties=rel.get("properties", {}) if rel else {},
                    weight=rel.get("weight", 1.0) if rel else 1.0,
                ),
            ))
        return neighbors

    def detect_communities(self) -> List[GraphCommunity]:
        # Neo4j GDS 社区检测
        try:
            query = """
            CALL gds.louvain.stream('myGraph')
            YIELD nodeId, communityId
            RETURN communityId, collect(gds.util.asNode(nodeId).id) AS members
            """
            results = self._run_query(query)
            return [
                GraphCommunity(
                    community_id=f"community_{r['communityId']}",
                    entities=r["members"],
                )
                for r in results
            ]
        except Exception:
            logger.warning("Neo4j GDS 不可用，返回空社区列表")
            return []

    def get_community_summary(self, community_id: str) -> Optional[str]:
        return None

    # ── 媒体引用图（RAG-Anything 风格） ──

    def add_chunk_node(self, chunk_id: str, doc_id: str, content_preview: str = "") -> bool:
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.doc_id = $doc_id, c.content_preview = $preview
        RETURN c
        """
        self._run_query(query, {"id": chunk_id, "doc_id": doc_id, "preview": content_preview[:200]})
        return True

    def add_media_node(self, media_id: str, media_type: str, doc_id: str = "",
                       page: int = 1, label: str = "", caption: str = "",
                       properties: Optional[dict] = None) -> bool:
        query = """
        MERGE (m:Media {id: $id})
        SET m.media_type = $media_type, m.doc_id = $doc_id, m.page = $page,
            m.label = $label, m.caption = $caption, m.properties = $properties
        RETURN m
        """
        self._run_query(query, {
            "id": media_id, "media_type": media_type, "doc_id": doc_id,
            "page": page, "label": label, "caption": caption, "properties": properties or {},
        })
        return True

    def add_reference(self, chunk_id: str, media_id: str, label: str = "",
                      page: int = 1, offset: int = 0, media_type: str = "image") -> bool:
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        MERGE (m:Media {id: $media_id})
        MERGE (c)-[r:REFERENCES {media_type: $media_type}]->(m)
        SET r.label = $label, r.page = $page, r.offset = $offset
        RETURN r
        """
        self._run_query(query, {
            "chunk_id": chunk_id, "media_id": media_id, "label": label,
            "page": page, "offset": offset, "media_type": media_type,
        })
        return True

    def get_media_by_chunk(self, chunk_id: str) -> List[dict]:
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[r:REFERENCES]->(m:Media)
        RETURN m, r
        """
        results = self._run_query(query, {"chunk_id": chunk_id})
        return [{
            "media_id": r["m"]["id"],
            "media_type": r["m"].get("media_type", "image"),
            "doc_id": r["m"].get("doc_id", ""),
            "page": r["m"].get("page", 1),
            "label": r["m"].get("label", ""),
            "caption": r["m"].get("caption", ""),
            "ref": {
                "label": r["r"].get("label", ""),
                "page": r["r"].get("page", 1),
                "offset": r["r"].get("offset", 0),
                "media_type": r["r"].get("media_type", "image"),
            },
        } for r in results]

    def get_media_node(self, media_id: str) -> Optional[dict]:
        query = "MATCH (m:Media {id: $id}) RETURN m"
        results = self._run_query(query, {"id": media_id})
        if not results:
            return None
        m = results[0]["m"]
        return {
            "media_id": m["id"],
            "media_type": m.get("media_type", "image"),
            "doc_id": m.get("doc_id", ""),
            "page": m.get("page", 1),
            "label": m.get("label", ""),
            "caption": m.get("caption", ""),
        }

    def list_media(self) -> List[dict]:
        query = "MATCH (m:Media) RETURN m"
        results = self._run_query(query)
        return [{
            "media_id": r["m"]["id"],
            "media_type": r["m"].get("media_type", "image"),
            "doc_id": r["m"].get("doc_id", ""),
            "page": r["m"].get("page", 1),
            "label": r["m"].get("label", ""),
            "caption": r["m"].get("caption", ""),
        } for r in results]

    def clear(self) -> bool:
        self._run_query("MATCH (n) DETACH DELETE n")
        return True

    def close(self):
        if self._driver:
            self._driver.close()


class GraphStoreFactory:
    """图存储工厂"""

    @staticmethod
    def create(
        db_type: str = "networkx",
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> BaseGraphStore:
        if db_type == "neo4j":
            if not all([uri, user, password]):
                raise ValueError("Neo4j 需要提供 uri, user, password")
            return Neo4jStore(uri, user, password)
        return NetworkXStore()