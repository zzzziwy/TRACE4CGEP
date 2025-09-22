# -*- coding: utf-8 -*-


class GraphDeactivationConfig:
    
    def __init__(self):
        # 基础失活率
        self.node_deactivate_rate = 0.10
        self.edge_deactivate_rate = 0.15
        
        # 重要性计算权重
        self.topology_weight = 0.4
        self.semantic_weight = 0.3
        self.prediction_weight = 0.3
        
        # 图算法参数
        self.use_betweenness_centrality = True
        self.use_degree_centrality = True
        self.use_path_coverage = True
        
        # 语义表示参数
        self.use_pretrained_embeddings = False
        self.embedding_dim = 768
        self.similarity_metric = 'cosine'
        
        # 失活策略参数
        self.inverse_proportional = True
        self.preserve_mask_node = True
        self.min_importance_threshold = 0.1
        self.importance_alpha = 1
        
        # 随机性参数
        self.seed = 42
        self.deterministic = False
        
        # 性能参数
        self.max_graph_size = 1000
        self.timeout_seconds = 30
        
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        if hasattr(args, 'node_deactivate_rate'):
            self.node_deactivate_rate = args.node_deactivate_rate
        if hasattr(args, 'edge_deactivate_rate'):
            self.edge_deactivate_rate = args.edge_deactivate_rate
        if hasattr(args, 'seed'):
            self.seed = args.seed
        if hasattr(args, 'topology_weight'):
            self.topology_weight = args.topology_weight
        if hasattr(args, 'semantic_weight'):
            self.semantic_weight = args.semantic_weight
        if hasattr(args, 'prediction_weight'):
            self.prediction_weight = args.prediction_weight

# default config
default_config = GraphDeactivationConfig()

# predefined config templates
def get_aggressive_config():
    config = GraphDeactivationConfig()
    config.node_deactivate_rate = 0.20
    config.edge_deactivate_rate = 0.20
    config.topology_weight = 0.5
    config.semantic_weight = 0.3
    config.prediction_weight = 0.2
    return config

def get_conservative_config():
    config = GraphDeactivationConfig()
    config.node_deactivate_rate = 0.05
    config.edge_deactivate_rate = 0.05
    config.topology_weight = 0.3
    config.semantic_weight = 0.4
    config.prediction_weight = 0.3
    return config

def get_balanced_config():
    config = GraphDeactivationConfig()
    config.node_deactivate_rate = 0.10
    config.edge_deactivate_rate = 0.10
    config.topology_weight = 0.4
    config.semantic_weight = 0.3
    config.prediction_weight = 0.3
    return config
