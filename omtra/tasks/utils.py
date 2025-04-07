from omtra.tasks.tasks import Task
from omtra.tasks.modalities import Modality
from omegaconf import DictConfig

from omtra.data.graph import edge_types as all_edge_types
from omtra.data.graph import to_canonical_etype

def get_edges_for_task(task: Task, graph_config: DictConfig) -> set:
    ntypes = set()
    modality_edges = set()
    for m in task.modalities_present:
        if m.is_node:
            ntypes.add(m.entity_name)
        else:
            modality_edges.add(m.entity_name)

    task_edges = set()
    graph_config_edges = set(list(graph_config.get("edges").keys()))
    for etype in all_edge_types:
        src_ntype, _, dst_ntype = to_canonical_etype(etype)
        if not (src_ntype in ntypes and dst_ntype in ntypes):
            continue
        # if the edge requires node types not supported, skip it
        # if the edge is not in the graph config, skip it, unless it has a modality on it or is a covalent edge
        predetermined = etype in modality_edges or 'covalent' in etype
        if etype not in graph_config_edges and not predetermined:
            continue
        task_edges.add(etype)
    return task_edges