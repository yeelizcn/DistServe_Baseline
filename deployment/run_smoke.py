"""Run one bounded two-node request; invoke from repository root."""
import argparse
import asyncio
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--prompt", default="用一句话介绍北京。")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["model_id"] != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("Unexpected model")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg.get("revision") or ""):
        raise ValueError("Set an immutable model revision")
    model_path = Path(cfg["model_path"])
    if model_path.name != cfg["revision"] or model_path.parent.name != "snapshots":
        raise ValueError("Use the local Hugging Face snapshots/<revision> directory on BOTH nodes")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
    if not cfg.get("ray_address"):
        raise ValueError("Explicit ray_address is required")
    import ray
    from distserve.config import (ModelConfig, ParallelConfig, DisaggParallelConfig,
        CacheConfig, ContextStageSchedConfig, DecodingStageSchedConfig)
    from distserve.engine import LLMEngine
    from distserve.request import SamplingParams
    ray.init(address=cfg["ray_address"])
    model = ModelConfig(str(model_path), tokenizer=str(model_path), dtype="fp16")
    limit = cfg["max_context"]
    if not 16 <= limit <= model.get_max_model_len():
        raise ValueError("Invalid context limit")
    engine = LLMEngine(model, DisaggParallelConfig(
        ParallelConfig(2), ParallelConfig(2), cfg["cross_node"]),
        CacheConfig(16, (limit + 15)//16, gpu_memory_utilization=0.8, cpu_swap_space=0),
        ContextStageSchedConfig("fcfs", 1, limit),
        DecodingStageSchedConfig("fcfs", 1, limit))
    tokens = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True)
    if len(tokens) + cfg["max_tokens"] > limit:
        raise ValueError("Prompt plus output exceeds context limit")

    async def run():
        await engine.initialize()
        loops = asyncio.create_task(engine.start_all_event_loops())
        async def consume():
            ids = []
            async for out in engine.generate(None, tokens, SamplingParams(
                    temperature=0, max_tokens=cfg["max_tokens"])):
                ids.append(out.new_token_id)
            return ids
        request = asyncio.create_task(consume())
        try:
            done, _ = await asyncio.wait([loops, request], timeout=cfg["cross_node"]["timeout_s"],
                                        return_when=asyncio.FIRST_COMPLETED)
            if loops in done:
                await loops
                raise RuntimeError("Engine loops ended unexpectedly")
            if request not in done:
                raise TimeoutError("Smoke request timed out; no success recorded")
            ids = await request
            if not ids:
                raise RuntimeError("Empty output")
            print(json.dumps({"token_ids": ids, "text": engine.tokenizer.decode(ids),
                              "revision": cfg["revision"]}, ensure_ascii=False))
        finally:
            request.cancel(); loops.cancel()
            await asyncio.gather(request, loops, return_exceptions=True)
    try:
        asyncio.run(run())
    finally:
        for stage in (engine.context_engine, engine.decoding_engine):
            for workers in stage.workers:
                for worker in workers:
                    ray.kill(worker)
        ray.shutdown()

if __name__ == "__main__":
    main()
