# OAT: One-class Agent Tracing

Implementation of the paper: Tracing Agentic Failure from the Flow of Success.

## Quick Start

```bash
python extract_states.py --dataset mcp_atlas --aggregation mean
python extract_states.py --dataset who_and_when --aggregation mean
python run_pipeline.py baselines --dataset mcp_atlas
python run_pipeline.py train --dataset mcp_atlas
python run_pipeline.py ood --train-dataset mcp_atlas --test-dataset who_and_when
```

`who_and_when` is failure-only, so OAT training should use `ood` with success trajectories from `mcp_atlas`.
