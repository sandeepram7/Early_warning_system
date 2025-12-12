# Early Warning System for Mental Health Risks

## Project Overview

This project establishes a large-scale, automated Early Warning System (EWS) to detect at-risk users in mental health-focused online communities, utilizing data scraped from Reddit. By combining Natural Language Processing (NLP) deep learning architectures with advanced Graph Theory, the system reconstructs digital communities as "signed interaction networks."

**Authors:** Arunachalam Vijayanand, Sandeep Ram, Mohil Ahuja, Vaibhav Dabas
**Supervisors:** Vedika Gupta (O.P Jindal University, India), Rajesh Sharma (Plaksha University)

## Key Objectives

1. **Semantic Emotional Classification:** Bypass simple positive/negative keyword matching and deploy deep contextual representation (`RoBERTa-base` fine-tuned on GoEmotions) across ~1 million posts/comments to detect granular emotional states.
2. **Signed Directed Network Construction:** Build a graph where users are nodes and directed replies are edges. Embed the computed emotional polarity as "signed" weights (uplifting, distressing, or neutral).
3. **Graph Mining for User States:** Compute specialized graph metrics (in-degree/out-degree, eigenvector centrality, clustering) to mathematically isolate two sub-phenomena:
   * **The "Lonely Core" (At-Risk Users):** High volume of negative output, lacking positive incoming support.
   * **Reciprocal Upliftment (Hope Speech):** Users who form bilateral positive sub-graphs by mutually supporting one another.
4. **Architectural & Algorithmic Scalability:** Ensure the pipeline safely circumvents memory leakage, Out-of-Memory (OOM) combinatorial explosions, and GPU VRAM bottlenecks to handle data at scale.

## Methodology

### 1. Sentiment Classification Pipeline (`sentiment_analysis_v2.ipynb`)

- **Model:** `RoBERTa-base` fine-tuned on Google GoEmotions (predicts 28 granular emotions mapped down to Positive, Neutral, Negative polarities).
- **Hardware Optimizations:** Inference was run on an NVIDIA RTX 4050 GPU. Employed PyTorch `DataLoader` with optimal batch sizes and Mixed Precision (FP16 via `torch.cuda.amp.autocast`) to halve memory utilization, avoid VRAM limits, and drastically reduce execution time.
- **Cross-Validation:** Validated RoBERTa's predictions against a VADER baseline and standard BERT, proving RoBERTa handles sarcastic and complex expressions much better.

### 2. Network Analysis Pipeline (`network_analysis_v2.ipynb`)

- **Architecture:** Translated relational data into a Signed Interaction Network using `NetworkX`.
- **Scale Handled:** Processed a graph containing **283,843 unique users (nodes)** connected by **892,331 paths of interaction (edges)**.
- **Algorithmic Optimizations:** Identified an $O(N^2)$ algorithm crash during graph traversal. Solved this by refactoring logic from nested iterative object generation to **Vectorized Relational Data processing** using Pandas inner joins (`pd.merge()`). Execution time collapsed from hours to under 5 seconds with flat memory consumption.

## Datasets

Dataset collected from Arctic Shift, focused on these mental-health subreddits:

- `r/anxiety`
- `r/lonely`
- `r/mentalhealth`
- `r/socialanxiety`
- `r/suicidewatch`

## Results Summary

- **Analyzed:** ~899,300 posts and 1,209,643 threaded comments.
- **EWS Detections:** Mathematically isolated **52,872 at-risk users** based on negative communication paired with graph isolation.
- **Community Support:** Discovered **291 pure reciprocal upliftment paths**, providing a statistical baseline to study positive intervention in digital spaces.

## Reports and Files

- `sentiment_analysis_v2.ipynb`: NLP emotion extraction and sentiment classification pipeline.
- `network_analysis_v2.ipynb`: Graph generation, processing, and at-risk user isolation.
- `IEEE_Report_Final.md`: Comprehensive academic reports summarizing the methodology, results, and literature review.

