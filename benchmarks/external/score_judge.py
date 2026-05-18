#!/usr/bin/env python3
"""Score 71 BEAM-10M judge jobs."""
import json

INPUT = '/home/alca/projects/mazemaker/benchmarks/external/results/beam_10m_judge_jobs_20260511T115739Z.jsonl'
OUTPUT = '/home/alca/projects/mazemaker/benchmarks/external/results/beam_10m_v2_scored.jsonl'

# Hand-graded scores keyed by (ability, q_idx, rubric_item_idx) -> (score, reason)
SCORES = {
    # abstention
    ('abstention', 0, 0): (0.0, "Response confabulates Johnny as 'security analyst' instead of stating no qualifications/expertise information exists."),
    ('abstention', 1, 0): (0.0, "Response fabricates a detailed multi-section agenda rather than acknowledging no agenda/format info exists."),
    # contradiction_resolution 0 — response only asserts logs are set up; doesn't acknowledge contradiction
    ('contradiction_resolution', 0, 0): (0.0, "Response does not mention there is contradictory information; it asserts logs are set up."),
    ('contradiction_resolution', 0, 1): (1.0, "Response explicitly affirms the user wired up diagnostic logging for shard-distribution errors, matching the rubric."),
    ('contradiction_resolution', 0, 2): (0.0, "Response does not mention the user said they had never set up such logs."),
    ('contradiction_resolution', 0, 3): (0.0, "Response does not ask 'which statement is correct?' or invite clarification."),
    # contradiction_resolution 1 — response mentions both sides but doesn't explicitly say contradictory or ask which is correct
    ('contradiction_resolution', 1, 0): (0.5, "Response surfaces that the stated rule and the user's evidence don't align, implying contradiction without using the word 'contradictory'."),
    ('contradiction_resolution', 1, 1): (1.0, "Response cites the user's personal rule of always including exact error messages."),
    ('contradiction_resolution', 1, 2): (1.0, "Response quotes the user saying 'I've never actually logged any errors for this'."),
    ('contradiction_resolution', 1, 3): (0.0, "Response does not pose 'which statement is correct?'"),
    # event_ordering 0 — 20 items in response; evaluate each rubric topic
    ('event_ordering', 0, 0): (1.0, "Item 2 'Plan the document-ingestion pipeline as foundation for enterprise search' satisfies 'Core ingestion pipeline initiation'."),
    ('event_ordering', 0, 1): (0.0, "No mention of batch vs streaming ingestion strategies in the 20-item list."),
    ('event_ordering', 0, 2): (0.0, "No mention of metadata extraction and normalization."),
    ('event_ordering', 0, 3): (0.0, "No mention of vectorization or indexing workflows; items focus on partitioning/sharding."),
    ('event_ordering', 0, 4): (0.0, "No mention of vector database cluster setup."),
    ('event_ordering', 0, 5): (0.0, "No mention of sparse retrieval index implementation as a distinct step."),
    ('event_ordering', 0, 6): (0.0, "No mention of core API scaffolding."),
    ('event_ordering', 0, 7): (0.0, "No mention of authentication and authorization integration."),
    ('event_ordering', 0, 8): (1.0, "Items 14-15 'Plan detailed logging' and 'Set up logs to capture granular query-failure details' satisfy logging and monitoring foundation."),
    ('event_ordering', 0, 9): (0.0, "No mention of infrastructure as code."),
    ('event_ordering', 0, 10): (1.0, "Item 11 'Develop a hybrid-retrieval prototype that blends sparse and dense retrieval' matches rubric."),
    ('event_ordering', 0, 11): (0.0, "No mention of dense vector search with approximate nearest neighbors."),
    ('event_ordering', 0, 12): (0.0, "No mention of combining retrieval scores for hybrid ranking."),
    ('event_ordering', 0, 13): (1.0, "Items 12-13 about sparse-dense integration in query pipelines satisfy query pipeline prototyping with hybrid retrieval."),
    ('event_ordering', 0, 14): (0.0, "No mention of query rewriting for improved recall."),
    ('event_ordering', 0, 15): (0.0, "No mention of evaluation metrics and relevance testing."),
    ('event_ordering', 0, 16): (0.0, "No mention of extending APIs for hybrid search."),
    ('event_ordering', 0, 17): (0.0, "No mention of multi-language tokenization."),
    ('event_ordering', 0, 18): (0.0, "No mention of caching strategies for frequent queries."),
    ('event_ordering', 0, 19): (1.0, "Items 15-17 about query-failure logging and automated alerting satisfy 'Logging query performance and errors'."),
    # event_ordering 1 — model abstains, claims only 8 topics; none of 11 rubric items listed
    ('event_ordering', 1, 0): (0.0, "Model abstains from providing the sequence; token limit and segmentation errors not listed."),
    ('event_ordering', 1, 1): (0.0, "Model abstains; context window resizing/mismatch errors not listed."),
    ('event_ordering', 1, 2): (0.0, "Model abstains; index scoring errors not listed."),
    ('event_ordering', 1, 3): (0.0, "Model abstains; rerank score and feedback parse errors not listed."),
    ('event_ordering', 1, 4): (0.0, "Model abstains; version conflict errors not listed."),
    ('event_ordering', 1, 5): (0.0, "Model abstains; metric calculation and spell check errors not listed."),
    ('event_ordering', 1, 6): (0.0, "Model abstains; encryption key and documentation format errors not listed."),
    ('event_ordering', 1, 7): (0.0, "Model abstains; query parse and synonym mismatch errors not listed."),
    ('event_ordering', 1, 8): (0.0, "Model abstains; intent reform and encoding mismatch errors not listed."),
    ('event_ordering', 1, 9): (0.0, "Model abstains; language detection and vector alignment errors not listed."),
    ('event_ordering', 1, 10): (0.0, "Model abstains; stemming rule, relevance score, and code switch errors not listed."),
    # information_extraction
    ('information_extraction', 0, 0): (1.0, "Response states '98% detection rate' matching rubric exactly."),
    ('information_extraction', 1, 0): (0.0, "Response states Milvus 2.2.0, not Milvus 2.3.1; numbers not equivalent."),
    # instruction_following
    ('instruction_following', 0, 0): (1.0, "Response includes concrete latency numbers (220ms target, 200ms, drop to 50-100ms)."),
    ('instruction_following', 0, 1): (1.0, "Response mentions timing metrics including 95th/99th percentile latency, GC pause, request latency."),
    ('instruction_following', 1, 0): (1.0, "Response provides numerical latency goals (<=200ms for 90% requests, <500ms for 99th percentile, query latency <=100ms)."),
    # knowledge_update
    ('knowledge_update', 0, 0): (0.0, "Response states no information available rather than '17 tasks'."),
    ('knowledge_update', 0, 1): (0.0, "Response states no information available rather than '88%'."),
    ('knowledge_update', 1, 0): (0.0, "Response says '15 tasks', not '14 tasks'; numerically distinct."),
    ('knowledge_update', 1, 1): (0.0, "Response says '90%', not '85%'; numerically distinct."),
    # multi_session_reasoning
    ('multi_session_reasoning', 0, 0): (0.0, "Response says ~1.3 million documents, not 1.8 million; off by 500k."),
    ('multi_session_reasoning', 1, 0): (1.0, "Response states 'about 5 000 queries per second' which equals 5,000."),
    # preference_following
    ('preference_following', 0, 0): (1.0, "Response includes AWS m5.large at $0.107/hour which rounds to $0.11/hour (paraphrase tolerance applies)."),
    ('preference_following', 0, 1): (0.0, "Example uses 15 instances rather than 500 instances; no calculation for 500 shown."),
    ('preference_following', 1, 0): (1.0, "Response mentions 'Milvus 2.2+' which encompasses Milvus 2.3.0 as compatible version."),
    ('preference_following', 1, 1): (1.0, "Response explicitly addresses indexing strategies for million+ vector workloads (HNSW, IVF families, sharding)."),
    # summarization 0
    ('summarization', 0, 0): (1.0, "Response covers exploring vector indexing strategies (IVF_SQ8, HNSW, Faiss, PCA)."),
    ('summarization', 0, 1): (1.0, "Response discusses trade-offs: recall vs latency, memory footprint, scalability, accuracy."),
    ('summarization', 0, 2): (1.0, "Response describes Loki integration for log aggregation alongside vector search, with real-time alerts and metrics."),
    ('summarization', 0, 3): (0.0, "Response mentions Faiss but does NOT combine Elasticsearch with Faiss for HA architecture."),
    ('summarization', 0, 4): (1.0, "Response describes API refinements (async endpoints, query-level batching, request-ID propagation) supporting vector search."),
    ('summarization', 0, 5): (1.0, "Response covers Prometheus metrics, Grafana dashboards, PagerDuty/Slack alerting for reliability and performance."),
    # summarization 1
    ('summarization', 1, 0): (1.0, "Response describes modular RAG architecture, high daily query volumes (high-load queries), latency and reliability targets."),
    ('summarization', 1, 1): (0.5, "Response mentions load-balancing hooks but no explicit advanced LB algorithms or health-check implementations."),
    ('summarization', 1, 2): (0.5, "Response mentions LRU/Redis-style caching but not specifically Redis Cluster distributed caching for fault tolerance."),
    ('summarization', 1, 3): (0.0, "Response does not mention microservices, container orchestration, or message queues."),
    ('summarization', 1, 4): (0.5, "Response covers monitoring setup and run-book/documentation but no explicit CI/CD pipeline configuration."),
    # temporal_reasoning
    ('temporal_reasoning', 0, 0): (0.0, "Response computes 3 days, not 14 days."),
    ('temporal_reasoning', 0, 1): (0.0, "Response gives Feb 15 to Feb 18, 2025, not Feb 15 to March 1, 2025."),
    ('temporal_reasoning', 1, 0): (0.0, "Response computes 53 days, not 45 days."),
    ('temporal_reasoning', 1, 1): (0.0, "Response uses Nov 1 to Dec 24, 2024, not Nov 1 to Dec 16, 2024."),
}


def main():
    with open(INPUT) as f:
        jobs = [json.loads(l) for l in f]
    assert len(jobs) == 71, f"Expected 71 jobs, got {len(jobs)}"

    out_lines = []
    for j in jobs:
        key = (j['ability'], j['q_idx'], j['rubric_item_idx'])
        if key not in SCORES:
            raise KeyError(f"Missing score for {key}")
        score, reason = SCORES[key]
        out = dict(j)
        out['judge_score'] = float(score)
        out['judge_reason'] = reason
        out_lines.append(json.dumps(out, ensure_ascii=False))

    with open(OUTPUT, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')

    # Distribution
    dist = {1.0: 0, 0.5: 0, 0.0: 0}
    for j in jobs:
        key = (j['ability'], j['q_idx'], j['rubric_item_idx'])
        dist[SCORES[key][0]] += 1
    print(f"Scored {len(jobs)} jobs.")
    print(f"Distribution: 1.0={dist[1.0]}, 0.5={dist[0.5]}, 0.0={dist[0.0]}")


if __name__ == '__main__':
    main()
