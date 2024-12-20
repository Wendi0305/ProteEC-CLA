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
* **Random split dataset**
   This dataset contains labeled protein sequences that have been annotated with enzyme commission (EC) numbers. It serves as a benchmark for evaluating the performance of the model. The dataset is carefully curated to ensure a balanced distribution of protein sequences across different EC classes. It is commonly used for model validation and comparison with existing methods.  
* **Clustered split dataset**
  This is a more challenging dataset designed to assess the model's ability to generalize across unseen protein families. The dataset is split into several clusters of related protein sequences, and the model is tasked with predicting EC numbers for sequences in clusters that were not seen during training. This dataset tests the robustness of the model and its ability to handle complex relationships between protein sequences and their functions.
## Data Access and Usage
The datasets have been uploaded to the **master branch** of the repository. You can access and download the datasets directly from the following links:[master](https://github.com/Wendi0305/ProteEC-CLA/master).
## Data Format
Data Format
Both datasets are in FASTA format for protein sequences and include additional metadata such as EC numbers and sequence identifiers. The files have been organized to ensure easy integration into the model, and scripts for preprocessing are also provided in the repository.
## Installation
To install and set up the environment for running ProteEC-CLA, follow the steps below:
**Clone the repository:**
```
git clone https://github.com/Wendi0305/ProteEC-CLA.git
cd ProteEC-CLA
```
**Install dependencies:**
It is recommended to use virtualenv or conda to create an isolated environment for the project.
```
pip install -r requirements.txt
```
**Download Pre-trained Models:**
ProteEC-CLA uses the pre-trained **ESM2** model for sequence embedding. If the model weights are not found locally, they will be automatically downloaded.
## Training and Testing
To train and test the **ProteEC-CLA** model, follow these steps:  
### Training  
```
python TRAIN-ESM2-Agent.py
```
### Testing  
```
python TEST-ESM2-Agent.py --fasta_fn /path/to/your/test_file.fasta -ont ec -v
```
* `--fasta_fn`:Path to your test dataset in FASTA format (replace`/path/to/your/test_file.fasta` with the actual path to your file).
