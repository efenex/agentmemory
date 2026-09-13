| system | corpus | recall@5 | recall@10 | NDCG@10 | MRR | embed ms | search ms/q |
|---|---|---|---|---|---|---|---|
| bm25-only (today) | synthetic | 42.5% | 57.9% | 84.7% | 91.7% | 0 | 2.9 |
| nomic-embed-text (768d) | synthetic | 44.3% | 62.5% | 92.9% | 100.0% | 1469 | 33.9 |
| embeddinggemma (768d) | synthetic | 44.8% | 58.8% | 88.7% | 100.0% | 1717 | 42.0 |
| qwen3-embedding:0.6b (1024d) | synthetic | 43.8% | 59.5% | 88.3% | 95.8% | 5526 | 28.5 |
| qwen3-embedding:8b (1024d) | synthetic | 43.0% | 61.5% | 89.8% | 96.7% | 11998 | 54.2 |
| bm25-only (today) | coding-life | 100.0% | 100.0% | 89.8% | 86.7% | 0 | 0.1 |
| nomic-embed-text (768d) | coding-life | 100.0% | 100.0% | 95.5% | 93.3% | 90 | 268.0 |
| embeddinggemma (768d) | coding-life | 100.0% | 100.0% | 95.0% | 93.3% | 2353 | 35.6 |
| qwen3-embedding:0.6b (1024d) | coding-life | 100.0% | 100.0% | 95.0% | 93.3% | 2147 | 30.0 |
| qwen3-embedding:8b (1024d) | coding-life | 100.0% | 100.0% | 95.5% | 93.3% | 744 | 57.5 |
