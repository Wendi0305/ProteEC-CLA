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
ProteEC-CLA has been tested on two independent and representative datasets:
1. **Random Split**
   This dataset contains labeled protein sequences that have been annotated with enzyme commission (EC) numbers. It serves as a benchmark for evaluating the performance of the model. The dataset is carefully curated to ensure a balanced distribution of protein sequences across different EC classes. It is commonly used for model validation and comparison with existing methods.
2.**Clustered split dataset**
  This is a more challenging dataset designed to assess the model's ability to generalize across unseen protein families. The dataset is split into several clusters of related protein sequences, and the model is tasked with predicting EC numbers for sequences in clusters that were not seen during training. This dataset tests the robustness of the model and its ability to handle complex relationships between protein sequences and their functions.
## Data Access and Usage
The datasets have been uploaded to the **master branch** of the repository. You can access and download the datasets directly from the following links:
* 
* 
To use the datasets for testing or training your own models, follow the steps outlined in the repository's instructions.
## Data Format
Data Format
Both datasets are in FASTA format for protein sequences and include additional metadata such as EC numbers and sequence identifiers. The files have been organized to ensure easy integration into the model, and scripts for preprocessing are also provided in the repository.
