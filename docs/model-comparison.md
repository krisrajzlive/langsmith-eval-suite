# Application model comparison

Generated 2026-10-12 08:03 UTC - prompt `v2` - judge `bedrock:mistral.mistral-large-3-675b-instruct` - repetitions 1.

| model | composite | correct | grounded | cited | recall | no-loop | tools | path | no-leak | safe | no-abuse | latency s | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bedrock:openai.gpt-oss-20b-1:0` | 0.95 | 0.92 | 0.96 | 1.00 | 0.95 | 1.00 | 0.90 | 0.90 | 0.83 | 0.83 | 0.67 | 1.6 | 2121 |
| `openai:gpt-4o-mini` | 0.91 | 0.88 | 0.89 | 0.95 | 0.89 | 1.00 | 1.00 | 0.90 | 0.92 | 0.92 | 1.00 | 2.4 | 1467 |
| `bedrock:amazon.nova-lite-v1:0` | 0.90 | 0.83 | 0.90 | 1.00 | 0.93 | 1.00 | 0.90 | 0.90 | 0.92 | 0.92 | 0.67 | 1.6 | 2377 |
| `bedrock:openai.gpt-oss-120b-1:0` | 0.89 | 0.96 | 0.87 | 0.59 | 0.95 | 1.00 | 0.90 | 0.90 | 0.83 | 0.92 | 1.00 | 2.3 | 2009 |
| `bedrock:moonshot.kimi-k2-thinking` | 0.89 | 0.88 | 0.95 | 0.91 | 0.89 | 0.92 | 1.00 | 1.00 | 0.75 | 0.83 | 1.00 | 3.5 | 2624 |
| `bedrock:amazon.nova-micro-v1:0` | 0.89 | 0.83 | 0.89 | 0.95 | 0.89 | 1.00 | 0.90 | 0.90 | 0.92 | 0.92 | 1.00 | 1.5 | 2397 |
| `bedrock:amazon.nova-pro-v1:0` | 0.88 | 0.83 | 0.86 | 0.91 | 0.89 | 1.00 | 1.00 | 0.90 | 0.83 | 0.92 | 1.00 | 2.0 | 2495 |
| `bedrock:qwen.qwen3-32b-v1:0` | 0.86 | 0.79 | 0.88 | 0.95 | 0.86 | 1.00 | 0.80 | 0.90 | 0.75 | 0.83 | 0.67 | 1.5 | 1888 |
| `bedrock:minimax.minimax-m2` | 0.86 | 0.88 | 0.95 | 0.91 | 0.95 | 1.00 | 1.00 | 0.90 | 0.67 | 0.92 | 1.00 | 4.2 | 2684 |
| `bedrock:mistral.mistral-large-3-675b-instruct` | 0.82 | 0.88 | 0.76 | 0.55 | 0.93 | 1.00 | 0.90 | 0.90 | 1.00 | 0.92 | 1.00 | 3.2 | 2020 |
| `bedrock:qwen.qwen3-next-80b-a3b` | 0.75 | 0.67 | 0.91 | 0.82 | 0.59 | 1.00 | 0.80 | 0.80 | 0.83 | 0.92 | 1.00 | 2.0 | 1698 |

Columns: composite = weighted quality score; correct/grounded/cited/recall from the Q&A benchmark; no-loop = finished within the step cap; tools/path = trajectory match and judge; no-leak/safe/no-abuse = safety set. Scores are means over 24 / 10 / 12 examples, so differences under about 0.1 are within run-to-run noise.
