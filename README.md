# ProteEC-CLA: Protein EC Number Prediction with Contrastive Learning and Agent Attention
ProteEC-CLA is a novel model designed for predicting enzyme Commission (EC) numbers using contrastive learning and an agent attention mechanism. The model leverages the power of protein language models and deep learning techniques to accurately predict EC numbers, significantly improving the efficiency of enzyme function annotation.
![Figure1](https://github.com/user-attachments/assets/731a950a-8d5b-49ba-97c3-41c62e8ed23e)
## Key Features
1. High-accuracy EC number prediction up to the **4th level**.  
2. Utilizes **contrastive learning** to improve model performance with unlabeled data.  
3. **Agent attention mechanism** enhances the model’s ability to capture both local and global sequence features.  
4. Integrates a **pre-trained protein language model (ESM2)** to generate informative sequence embeddings for deeper functional analysis.
## Requirements
* h5py==3.6.0  
* numpy<=1.22  
* scikit-learn==0.24.2  
* torch==1.10.0  
* transformers==4.17.0  
* sentencepiece  
## Datasets
ProteEC-CLA has been tested on two independent datasets:
1. Standard dataset: Contains labeled protein sequences for benchmark testing.
2. Clustered split dataset: A more challenging dataset that tests the model’s ability to generalize to unseen protein families.
