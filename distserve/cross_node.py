"""Fixed two-node placement; no silent local fallback."""
import math
from urllib.parse import urlparse

def validate_config(config, context, decoding):
    for p in (context, decoding):
        if (p.tensor_parallel_size, p.pipeline_parallel_size) != (2, 1):
            raise ValueError("Cross-node mode requires TP=2, PP=1 on both stages")
    if not config.get("prefill_node_id") or not config.get("decode_node_id"):
        raise ValueError("Explicit Ray node IDs are required")
    if config["prefill_node_id"] == config["decode_node_id"]:
        raise ValueError("Prefill and Decode must occupy different nodes")
    address = urlparse(config.get("init_method", ""))
    if address.scheme != "tcp" or not address.hostname or not address.port:
        raise ValueError("init_method must be tcp://<prefill address>:<free port>")
    if not math.isfinite(config.get("timeout_s", 0)) or config["timeout_s"] <= 0:
        raise ValueError("timeout_s must be positive")

def create_placement_groups(config):
    import ray
    from ray.util.placement_group import placement_group
    if not ray.is_initialized():
        raise RuntimeError("Connect explicitly to the two-node Ray cluster first")
    nodes = {n["NodeID"]: n for n in ray.nodes() if n["Alive"]}
    groups = []
    try:
        for role in ("prefill", "decode"):
            node = nodes[config[role + "_node_id"]]
            if node["Resources"].get("GPU") != 2:
                raise ValueError("Each dedicated Ray node must expose exactly two assigned GPUs")
            resource = "node:" + node["NodeManagerAddress"]
            group = placement_group([{"GPU": 1, resource: 0.001}] * 2,
                                    strategy="STRICT_PACK")
            groups.append(group)
            ray.get(group.ready(), timeout=config["timeout_s"])
    except BaseException:
        for group in groups:
            ray.util.remove_placement_group(group)
        raise
    return [groups[0]], [groups[1]]
