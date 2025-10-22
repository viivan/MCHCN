# MCHCN: Multi-Channel Hypergraph Convolutional Network for Web API Recommendation
<p float="left">
    <img src="https://img.shields.io/badge/python-v3.7-red"> 
    <img src="https://img.shields.io/badge/tensorflow-v1.14-blue"> 
</p>


## Introduction
**MCHCN** (Multi-Channel Hypergraph Convolutional Network Guided by Hybrid Random Walks) is a Python-based framework dedicated to **Web API recommendation** in service-oriented computing scenarios, as proposed in our paper. It addresses the core challenges of complex mashup-API relationships and data sparsity in service recommendation, achieving efficient and accurate Web API recommendation by integrating hypergraph modeling, hybrid random walks, and contrastive learning.  

The framework is fully compatible with Python 3.7.4 and TensorFlow 1.14+, and its design strictly aligns with the technical details and experimental settings in *MCHCN paper*.


## Architecture
The overall architecture of MCHCN  consists of three core modules, as below:  
[MCHCN Architecture]
1. **Multi-Channel Hypergraph Construction**: Extract 4 types of service-specific motifs (derived from mashup-API calling relationships and mashup tag information) to build hypergraphs, capturing high-order interactions between mashups and APIs that traditional pairwise graphs cannot represent.  
2. **Hybrid Random Walk-Guided Convolution**: Use a hybrid random walk strategy (combining partially absorbing and biased random walks) to optimize hypergraph convolution weights, reducing noise and enhancing global information capture.  
3. **Channel Attention & Contrastive Learning**: Dynamically adjust multi-channel weights via a channel error attention mechanism, and introduce dual-scale contrastive learning to compensate for information loss, ensuring high recommendation accuracy.


## Core Features (From MCHCN paper)
1. **Motif-Based Hypergraph for Sparsity Mitigation**  
   As detailed in *MCHCN paper* Section 3.3, by studying mashup-API calling relationships and mashup tag information, 4 types of motifs are defined:  
   - Cross Correlated Motif (shared tags between mashups)  
   - Multiple Strongly Correlated Motif (shared ≥2 tags between mashups)  
   - Single Relation Motif (mashups calling the same API)  
   - Multi-relational Motif (mashups sharing tags and calling the same API)  
   The constructed hypergraphs not only extract effective high-order information but also significantly alleviate the data sparsity issue in mashup-API calling matrices.  

2. **Hybrid Random Walk for Weight Optimization**  
   Per *MCHCN paper* Section 3.4.1, the hybrid random walk retains the advantages of partially absorbing random walks (capturing global hypergraph information) and biased random walks (balancing homophily and structural equivalence). It optimizes the hypergraph convolution weight matrix, avoiding noise introduced by direct convolution on raw hypergraphs.  

3. **Channel Attention & Contrastive Learning for Accuracy Enhancement**  
   - *Channel Error Attention Mechanism* (Section 3.4.3): Measures the error of each hypergraph channel to dynamically allocate weights, prioritizing channels with stronger discriminative ability for final embedding aggregation.  
   - *Dual-Scale Contrastive Learning* (Section 3.5): Performs contrastive learning on embeddings from "mashup-mashup interaction" and "mashup-API interaction" paths, maximizing consistency for the same node and difference for different nodes, and reducing noise from complex aggregations.  


## Requirements
To replicate the experiments in *MCHCN paper* and run the framework stably, install the following dependencies (version consistency ensures alignment with paper results):  
```
gensim==4.1.2
joblib==1.1.0
mkl==2022.0.0
mkl_service==2.4.0
networkx==2.6.2
numba==0.53.1
numpy==1.20.3
scipy==1.6.2
tensorflow==1.14.0
```  



## Usage

### 1. Dataset Preparation

- **Paper-Specific Dataset (From MCHCN paper)**: The **ProgrammableWeb dataset** used in the experiments of *MCHCN.paper* (Section 4.1) — which contains 6,217 mashup services, 11,930 Web APIs, and corresponding mashup tag information — is hosted in our dedicated GitHub repository. To replicate the experimental results in *MCHCN paper* (e.g., performance comparison in Section 4.4, robustness test in Section 4.6), please access the repository for dataset download and detailed usage guidelines:https://github.com/viivan/mashup-and-Web-API-data1.

### 2. Configuration 
 Navigate to the `./config/` directory and modify `MCHCN.conf`—all core parameters are pre-configured to match the optimal settings in *MCHCN paper* (Section 4.7 Parameter Experiment):  
   ```ini
   [MCHCN]
   -emb_size 100          # Embedding dimension (Section 4.1: 100)
   -lRate 0.001           # Learning rate (Section 4.1: 0.001)
   -maxEpoch 30           # Training epochs (Section 4.1: 30 for sparse datasets)
   -regU 0.001            # Mashup regularization coefficient (Section 4.6: 0.001)
   -regI 0.001            # API regularization coefficient (Section 4.6: 0.001)
   -n_layer 3             # Hypergraph convolution layers (Section 4.7: 3)
   -ss_rate 0.01          # Contrastive loss weight (Section 4.7: 0.01)
   -batch_size 32         # Batch size (Section 4.1: 32)
   -neg_sample_ratio 4    # Negative sample ratio (Section 4.1: 1 positive : 4 negative)
   -tag_path ./data/MCHCN_tag.csv  # Mashup tag data path (Section 3.3: required for motif construction)
   -delta 0.4 -p 4 -q 1   # Hybrid random walk parameters (Section 4.7: optimal combination)
   ```  



### 3. Run the Model
Execute `main.py` directly :  
```bash
python main.py
```  

- **Training Log**: Real-time logs (epoch, batch, BPR loss, contrastive loss) will be printed.  
- **Result Saving**: Evaluation metrics (NDCG@5, HR@5) are saved to `./results/MCHCN/`.  


