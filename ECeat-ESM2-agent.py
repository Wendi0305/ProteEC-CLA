from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import h5py

import random
from model.AgentAttention import AgentAttention
import gc
gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
class ProtTucker(nn.Module):
    def __init__(self, input_dim=1280, embed_dim=64, window=14):
        super(ProtTucker, self).__init__()

        self.embed_dim = embed_dim
        self.window = window

        # 将输入嵌入转换为二维表示的层
        self.embed_to_img = nn.Linear(input_dim, embed_dim * window * window)
        
        # AgentAttention模块
        self.agent_attention = AgentAttention(dim=embed_dim, num_heads=8, 
                                              qkv_bias=False, attn_drop=0., proj_drop=0.,
                                              agent_num=49, window=window)

        self.protTucker = nn.Sequential(
            nn.Linear(1280, 256),           # 将输入维度为 1280 的数据转换为输出维度为 256 
            nn.Tanh(),                      # 对线性层的输出进行非线性变换
            nn.Linear(256, 128),            # 将输入维度为 256 的数据转换为输出维度为 128
        )
        # 将AgentAttention的输出维度调整为ProtTucker所需的维度
        self.adjust_dim = nn.Linear(embed_dim * window * window, 256)

    def single_pass(self, X):
        X = X.float()
        return self.protTucker(X)

    def forward(self, X):                        # X (256, 3, 1280)
        batch_size = X.shape[0]

        # 将输入X转换为二维表示
        X_embedded = self.embed_to_img(X.view(-1, X.size(-1)))
        X_reshaped = X_embedded.view(batch_size * 3, self.window * self.window, self.embed_dim)  # 这里重塑为 [batch_size * 3, num_patches, embed_dim]
        
        
        # 应用AgentAttention
        attention_output = self.agent_attention(X_reshaped)
        
        # 调整维度以匹配ProtTucker
        X_adjusted = self.adjust_dim(attention_output.view(batch_size * 3, -1))
        X_adjusted = X_adjusted.view(batch_size, 3, -1)
        
        anchor = self.single_pass(X[:, 0, :])    # anchor (256, 128)
        pos = self.single_pass(X[:, 1, :])       # pos (256, 128)
        neg = self.single_pass(X[:, 2, :])       # neg (256, 128)
        return (anchor, pos, neg)
    


class Evaluator():
    def __init__(self, predictions):
        self.Ys, self.Yhats, self.reliabilities = zip(*[(query_label, lookup_label, eat_dist)
                                                        for _, query_label, _, lookup_label, eat_dist, nn_iter in predictions
                                                        if nn_iter == 0
                                                        ]
                                                      )  # 从predictions中提取，但只包括nn_iter（最近邻迭代次数）为0的元素

    def compute_performance(self): #计算性能指标和误差范围，并打印结果
        error_estimates = self.compute_err()
        for metric, (performance, bootstrap_err) in error_estimates.items(): # 遍历error_estimates中的性能指标和误差数据，以及它们的键（metric）和值（performance, bootstrap_err）
            print("{}={:.3f} +/-{:.3f}".format(
                metric,
                performance,
                1.96*np.std(np.array(bootstrap_err), ddof=1)    # 使用np.std()计算误差的标准差，然后乘1.96来计算95%的置信区间范围。
            )  # 打印性能指标和相应的误差范围
            )

        return None

    def compute_err(self, n_bootstrap=1000): # 使用了一种称为自助法（bootstrap）的统计方法来估计性能指标的置信区间
                                             # 自助法中进行采样的次数，原n_bootstrap = 1000。（原测试集：490）
        n_total = len(self.Ys)  # total number of predictions 预测总数（测试集：39313）
        idx_list = range(n_total)

        Ys, Yhats = np.array(self.Ys), np.array(self.Yhats) # 将 真实标签 Ys 和 预测标签 Yhats 转换为 NumPy 数组

        # acc = accuracy_score(Ys, Yhats)
        # f1 = f1_score(Ys, Yhats, average="weighted") #average="weighted"  计算加权平均值
        # bAcc = balanced_accuracy_score(Ys, Yhats)

        # 这些是性能指标的初始值
        acc = accuracy_score(Ys, Yhats)
        f1 = f1_score(Ys, Yhats, average="weighted") #average="weighted"  计算加权平均值
        bAcc = balanced_accuracy_score(Ys, Yhats)
        cm = confusion_matrix(Ys, Yhats)
        print("Confusion Matrix:")
        print(cm)

        # 创建用于存储自助法采样得到的性能指标的列表
        accs_btrap, f1s_btrap, bAccs_btrap = list(), list(), list() 
        # accs_btrap, f1s_btrap, bAccs_btrap, pres_btrap, recalls_btrap= list(), list(), list(),list(), list()

        n_skipped = 0
        for _ in range(n_bootstrap):
            rnd_subset = random.choices(idx_list, k=n_total) #从索引列表 idx_list 中随机选择 n_total 个索引，构成一个随机子集
            # skip bootstrap iterations where predictions might hold labels not part of groundtruth 跳过引导迭代，因为预测可能包含不属于 GroundTruth 的标签

            if not set(Yhats[rnd_subset]).issubset(Ys[rnd_subset]):
                n_skipped += 1
                continue

            accs_btrap.append(accuracy_score(
                Ys[rnd_subset], Yhats[rnd_subset])
                )
            f1s_btrap.append(
                f1_score(Ys[rnd_subset], Yhats[rnd_subset], average="weighted")
                )
            bAccs_btrap.append(balanced_accuracy_score(
                Ys[rnd_subset], Yhats[rnd_subset])
                )

        print("Skipped {}/{} bootstrap iterations due to mismatch of Yhat and Y.".format(n_skipped, n_bootstrap))
        
        result = {
            "ACCs": (acc, accs_btrap), 
            "bACCs": (bAcc, bAccs_btrap), 
            "F1": (f1, f1s_btrap),
        }

        print("ACCs:", result["ACCs"])
        print("bACCs:", result["bACCs"])
        print("F1:", result["F1"])

        return {"ACCs": (acc, accs_btrap), "bACCs": (bAcc, bAccs_btrap), "F1": (f1, f1s_btrap)}

class Embedder():
    def __init__(self):
        self.embedder, self.tokenizer = self.get_esm1b()       # 调用 get_prott5 方法来初始化 embedder 和 tokenizer，这些是用于生成蛋白质嵌入的关键组件

    def get_esm1b(self):
        start=time.time()
        # Load your checkpoint here  在此处加载检查点
        # Currently, only the encoder-part of ProtT5 is loaded in half-precision  目前只有 ProtT5 的编码器部分以半精度加载
        from transformers import AutoModel, AutoTokenizer
        print("Start loading ESM2...")
        model_name = "facebook/esm2_t33_650M_UR50D"
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        print("Finished loading {} in {:.1f}[s]".format(model_name,time.time()-start))
        return model, tokenizer
    
    def write_embedding_list(self,emb_p, ids,embeddings):                       # 参数：嵌入数据、对应ID、输出文件的路径
        embeddings=embeddings.detach().cpu().numpy().squeeze()                  # 将嵌入数据从pytorch张量转换为numpy数组，写入H5文件
        with h5py.File(str(emb_p),"w") as hf:
            for idx, seq_id in enumerate(ids):
                hf.create_dataset(seq_id,data=embeddings[idx])
        return None
                
    def write_embeddings(self, emb_p, embds):                                   # 与上边类似，但接受一个字典 embds 其中包含序列ID到嵌入数据的映射
        with h5py.File(str(emb_p), "w") as hf:
            for sequence_id, embedding in embds.items():
                # noinspection PyUnboundLocalVariable
                hf.create_dataset(sequence_id, data=embedding)
        return None
    
    def get_embeddings_batch(self, id2seq, max_residues=4000, max_seq_len=1000, max_batch=1):   # id2seq ：序列ID 和 序列；（最大残基数、最大序列长度、最大批次大小）
        print("Start generating embeddings for {} proteins.".format(len(id2seq)) +
              "This process might take a few minutes." +
              "Using batch-processing! If you run OOM/RuntimeError, you should use single-sequence embedding by setting max_batch=1.")
        start = time.time()
        ids = list()
        embeddings = list()
        batch = list()
        
        id2seq = sorted( id2seq.items(), key=lambda kv: len( id2seq[kv[0]] ), reverse=True ) #对输入的 id2seq 字典按照序列的长度进行排序，以便首先处理较长的序列
        # 遍历排好序的蛋白质序列
        for seq_idx, (protein_id, original_seq) in enumerate(id2seq):               # seq_idx:序列的索引；protein_id：蛋白质序列ID；original_seq：原始蛋白质序列
            seq = original_seq.replace('U','X').replace('Z','X').replace('O','X')   # 将 'U'、'Z' 和 'O' 替换为 'X' 的目的是：将这些非标准或未知的氨基酸都表示为相同的占位符 'X'
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((protein_id,seq,seq_len))      # 将序列转换为以空格分隔的字符列表，并将其添加到当前批次中。还记录了序列的长度。
            
            
            n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len    # 计算了当前批次中所有蛋白质序列的总残基数，并加上当前序列的残基数。

            # 条件检查：检查是否达到生成一个新批次的条件
            # 批次大小达到 max_batch 或者 总残基数达到 max_residues 或者 已经处理了所有序列 或者 当前序列长度超过了 max_seq_len
            if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(id2seq) or seq_len>max_seq_len:
                protein_ids, seqs, seq_lens = zip(*batch)       # 满足条件，就将当前批次的信息拆分成三个列表
                batch = list()

                # 使用 tokenizer 的 batch_encode_plus 方法对当前批次的蛋白质序列进行编码，并获得输入 IDs 和注意力掩码
                token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device) # 注意力掩码，它告诉模型哪些位置是有效的

                # 使用加载的 embedder 模型来计算蛋白质序列的嵌入表示
                try:
                    with torch.no_grad():
                        # get embeddings extracted from last hidden state  获取从最后一个隐藏状态提取的嵌入
                        batch_emb = self.embedder(input_ids, attention_mask=attention_mask).last_hidden_state # [B, L, 1024]
                except RuntimeError as e :
                    print(e)
                    print("RuntimeError during embedding for {} (L={})".format(protein_id, seq_len))
                    continue
                
                # 遍历批次中的每个序列，计算每个序列的平均嵌入表示，并将结果添加到 ids 和 embeddings 列表中。
                for batch_idx, identifier in enumerate(protein_ids):
                    s_len = seq_lens[batch_idx]
                    emb = batch_emb[batch_idx,:s_len].mean(dim=0,keepdims=True)
                    ids.append(protein_ids[batch_idx])
                    embeddings.append(emb.detach())

        print("Creating per-protein embeddings took: {:.1f}[s]".format(time.time()-start))
        embeddings = torch.vstack(embeddings)           # 将生成的嵌入表示作为 PyTorch 张量返回
        return ids, embeddings


# EAT: Embedding-based Annotation Transfer 基于嵌入的注释传输
class EAT():
    def __init__(self, lookup_p, query_p, output_d, use_tucker, num_NN,
                 lookupLabels, queryLabels):

        self.output_d = output_d
        Path.mkdir(output_d, exist_ok=True)  #exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
        
        self.num_NN = num_NN
        self.Embedder = None                # 将最近邻数量（num_NN）和 Embedder 属性初始化为空
        
        self.lookup_ids, self.lookup_embs = self.read_inputs(lookup_p)      # 使用 read_inputs 方法，读取查找数据和查询数据的 ids 和embeddings
        self.query_ids, self.query_embs = self.read_inputs(query_p)

        if use_tucker:  # create ProtTucker(ProtT5) embeddings
            
            # github上新加
            self.lookup_embs = self.lookup_embs.to(torch.float)             # 将嵌入的数据类型转换为浮点数
            self.query_embs = self.query_embs.to(torch.float)

            self.lookup_embs = self.tucker_embeddings(self.lookup_embs)     # 调用 tucker_embeddings 方法  （加载预训练模型） 来处理这些嵌入
            self.query_embs = self.tucker_embeddings(self.query_embs)

        self.lookupLabels = self.read_label_mapping(self.lookup_ids, lookupLabels)  # 调用 read_label_mapping 方法来读取查找数据和查询数据的标签映射
        self.queryLabels = self.read_label_mapping(self.query_ids, queryLabels)

    def tucker_embeddings(self, dataset):                    # 用于处理数据集的嵌入：下载预训练的 Tucker 模型并应用它来转换嵌入数据
        weights_p = self.output_d / "weights-EC(ESM2-agent).pt"

        # if no pre-trained model is available, yet --> download it  预训练的tucker模型权重文件，不存在就下载
        if not weights_p.exists():
            import urllib.request
            print("No existing model found. Start downloading pre-trained ESM2...")
            weights_link = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"           
            urllib.request.urlretrieve(weights_link, str(weights_p))

        print("Loading Tucker checkpoint from: {}".format(weights_p))
        state = torch.load(weights_p)['state_dict']            # 加载已下载的 Tucker 模型的权重文件，然后使用 PyTorch 的 torch.load 函数加载权重文件中的模型状态字典（state_dict）。

        model = ProtTucker().to(device)                      # 创建了一个 Tucker 类的实例，对数据集进行嵌入转换
        model.load_state_dict(state)                    # 并将加载的模型状态字典 state 应用到model实例上（这将导入预训练模型的权重和参数）
        model=model.eval()                              # 将模型切换为评估模式，可以确保在嵌入计算期间不会执行与训练相关的操作，以提高性能并避免不必要的计算。

        start = time.time()
        dataset = model.single_pass(dataset)
        print("Tuckerin' took: {:.4f}[s]".format(time.time()-start))
        return dataset

    def read_inputs(self, input_p):                         # 根据输入文件的类型（FASTA 或 H5）来读取数据，如果是 FASTA，则会计算嵌入。
        # define path for storing embeddings 定义存储嵌入的路径

        if not input_p.is_file():
            print("Neither input fasta, nor embedding H5 could be found for: {}".format(input_p))
            print("Files are expected to either end with .fasta or .h5")
            raise FileNotFoundError

        if input_p.name.endswith(".h5"): # if the embedding file already exists （嵌入文件已存在）
            return self.read_embeddings(input_p)
        
        elif input_p.name.endswith(".fasta"): # compute new embeddings if only FASTA available
            if self.Embedder is None: # avoid re-loading the pLM
                self.Embedder = Embedder()
            id2seq = self.read_fasta(input_p)                               # id2seq中存放FASTA文件中的蛋白质序列
            
            ids, embeddings = self.Embedder.get_embeddings_batch(id2seq)    # 使用 Embedder 对象的 get_embeddings_batch 方法来计算嵌入。
                                                                            # 该方法接受一个字典，键：蛋白质标识符；值：蛋白质序列
            emb_p  = self.output_d / input_p.name.replace(".fasta", ".h5")  # 构建了一个输出文件的路径 emb_p，将输入文件的 ".fasta" 扩展名替换为 ".h5"。
            self.Embedder.write_embedding_list(emb_p, ids,embeddings)       # 使用 self.Embedder 对象的 write_embedding_list 方法将计算得到的embeddings和ids写入到输出文件中。
            return ids, embeddings
        else:
            print("The file you passed neither ended with .fasta nor .h5. " +
                  "Only those file formats are currently supported.")
            raise NotImplementedError

    def read_fasta(self, fasta_path):                       # 解析 FASTA 文件（ECval.fasta），将蛋白质序列存储为字典，并执行一些预处理操作。
        '''    
            将 fasta 文档中的序列存储为字典，键是 fasta 标头，值是序列。
            此外，替换串行中的间隙字符和插入，因为在序列生成嵌入时，ProtT5 无法处理这些字符和插入。
            此外，替换 FASTA 标头中的特殊字符，因为从 H5 加载预先计算的嵌入时，这些字符被解释为特殊标记。
        '''
        sequences = dict()
        with open(fasta_path, 'r') as fasta_f:
            for line in fasta_f:
                line=line.strip()
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    if '|' in line and (line.startswith(">tr") or line.startswith(">sp")):
                        seq_id = line.split("|")[1]
                    else:
                        seq_id = line.replace(">","")
                    # replace tokens that are mis-interpreted when loading h5
                    seq_id = seq_id.replace("/", "_").replace(".", "_")
                    sequences[seq_id] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines  
                    # drop gaps and cast to upper-case 
                    sequences[seq_id] += ''.join(
                        line.split()).upper().replace("-", "")
        return sequences

    def read_embeddings(self, emb_p):                       # 读取嵌入数据，从预先计算的 H5 文件中加载嵌入。
        start = time.time()
        h5_f = h5py.File(emb_p, 'r')
        dataset = {pdb_id: np.array(embd) for pdb_id, embd in h5_f.items()}
        keys, embeddings = zip(*dataset.items())
        if keys[0].startswith("cath"):
            keys = [key.split("|")[2].split("_")[0] for key in keys ]
        # matrix of values (protein-embeddings); n_proteins x embedding_dim 
        embeddings = np.vstack(embeddings)
        print("Loading embeddings from {} took: {:.4f}[s]".format(
            emb_p, time.time()-start))
        return list(keys), torch.tensor(embeddings).to(device).float()

    def read_label_mapping(self, set_ids, label_p):         # 读取标签映射文件（EClist-val.txt），将蛋白质标识符(EC)映射到标签(蛋白质ID)。
        if label_p is None:
            return {set_id: None for set_id in set_ids}
        
        # in case you pass your own label mapping, you might need to adjust the function below 如果您传递自己的标签映射，则可能需要调整以下函数
        # EC编号在pkl文件中 "D:\projects\deepec torch\data\CNN2_old\data\data_old\uniprot_sprot3d_old_test.pkl"

        with open(label_p, 'r') as in_f:
            # protein-ID : label
            label_mapping = {line.strip().split(
                ' ')[0]: line.strip().split('EC:')[1] for line in in_f}
        return label_mapping

    def write_predictions(self, predictions):               # 将预测结果写入文件，包括查询蛋白质、查找蛋白质、嵌入距离等信息。
        out_p = self.output_d / "eat_resultEC-ESM2-Attention.txt"
        with open(out_p, 'w+') as out_f:
            out_f.write(
                "Query-ID\tQuery-Label\tLookup-ID\tLookup-Label\tEmbedding distance\tNearest-Neighbor-Idx\n")
            out_f.write("\n".join(
                ["{}\t{}\t\t{}\t\t{}\t\t{:.4f}\t\t{}".format(query_id, query_label, lookup_id, lookup_label, eat_dist, nn_iter+1)
                 for query_id, query_label, lookup_id, lookup_label, eat_dist, nn_iter in predictions
                 ]))
        return None

    def pdist(self, lookup, queries, norm=2, use_double=False):     # 计算查找数据和查询数据之间的距离矩阵，这里使用欧氏距离
        lookup=lookup.unsqueeze(dim=0)
        queries=queries.unsqueeze(dim=0)            # 扩展 lookup 和 queries 张量的维度，二维扩成三维，因为 torch.cdist 函数要求输入的张量具有三维形状

        # double precision improves performance slightly but can be removed for speedy predictions (no significant difference in performance)
        # 双精度略微提高了性能，但可以删除以进行快速预测（性能没有显着差异）
        if use_double:
            lookup=lookup.double()
            queries=queries.double()                # 将 lookup 和 queries 张量的数据类型转换为双精度浮点型（double），这是为了提高精度

        try: # try to batch-compute pairwise-distance on GPU   尝试在 GPU 上批量计算成对距离
            pdist = torch.cdist(lookup, queries, p=norm).squeeze(dim=0)   # torch.cdist 函数用于计算两个张量之间的距离矩阵；squeeze(dim=0) 用于移除维度为1的维度，以获得正确的输出形状
        except RuntimeError as e:
            print("Encountered RuntimeError: {}".format(e))
            print("Trying single query inference on GPU.") #在GPU上尝试单个查询推理。
            try: # if OOM for batch-GPU, re-try single query pdist computation on GPU  如果batch-GPU出现OOM，重新在GPU上尝试单个
                pdist = torch.stack(
                    [torch.cdist(lookup, queries[0:1, q_idx], p=norm).squeeze(dim=0)
                     for q_idx in range(queries.shape[1])
                     ]
                ).squeeze(dim=-1).T

            except RuntimeError as e: # if OOM for single GPU, re-try single query on CPU  如果单GPU出现OOM，重新在CPU上进行单个查询
                print("Encountered RuntimeError: {}".format(e))
                print("Trying to move single query computation to CPU.") #试图将单个查询计算移动到CPU
                lookup=lookup.to("cpu")
                queries=queries.to("cpu")
                pdist = torch.stack(
                    [torch.cdist(lookup, queries[0:1, q_idx], p=norm).squeeze(dim=0)
                     for q_idx in range(queries.shape[1])
                     ]
                ).squeeze(dim=-1).T
                
        print(pdist.shape)
        return pdist                    # 无论是在 GPU 还是 CPU 上计算，距离矩阵将存储在 pdist 变量中，并返回该距离矩阵作为函数的输出。

    def get_NNs(self, threshold, random=False):                 #用于获取最近邻蛋白质，如果 random 参数为真，则会返回随机预测；否则，将返回真正的最近邻预测。
    #def get_NNs(self, threshold, random=True):
        start = time.time()
        p_dist = self.pdist(self.lookup_embs, self.query_embs)  # 计算查询蛋白质嵌入和参考蛋白质嵌入之间的距离       
                                                                # 计算两个矩阵之间欧氏距离的函数，返回一个包含距离值的张量（距离矩阵）。

        if random: # this is only needed for benchmarking against random background 这只需要在随机背景下进行基准测试
            print("Making RANDOM predictions!")
            nn_dists, nn_idxs = torch.topk(torch.rand_like(
                p_dist), self.num_NN, largest=False, dim=0)
        else: # infer nearest neighbor indices 推断最近邻指数
            nn_dists, nn_idxs = torch.topk(
                p_dist, self.num_NN, largest=False, dim=0)   # 从计算的距离矩阵中选择前self.num_NN个最小值，以得到最近邻的距离和索引。
            
        print("Computing NN took: {:.4f}[s]".format(time.time()-start))
        nn_dists, nn_idxs = nn_dists.to("cpu"), nn_idxs.to("cpu")

        # 迭代查询蛋白质
        predictions = list()        # 空列表，存储预测结果
        n_test = len(self.query_ids)
        for test_idx in range(n_test):  # for all test proteins 所有查询蛋白
            query_id = self.query_ids[test_idx]  # get id of test protein 获取当前查询蛋白的ID
            nn_idx = nn_idxs[:, test_idx]        # 获取当前查询蛋白的最近邻索引
            nn_dist = nn_dists[:, test_idx]      # 获取当前查询蛋白的最近邻距离

            for nn_iter, (nn_i, nn_d) in enumerate(zip(nn_idx, nn_dist)):
                # index of nearest neighbour (nn) in train set  训练集中最近邻(nn)的索引
                nn_i, nn_d = int(nn_i), float(nn_d)

                # if a threshold is passed, skip all proteins above this threshold  如果超过阈值，则跳过所有高于此阈值的蛋白质
                if threshold is not None and nn_d > threshold:
                    continue

                # get id of nn (infer annotation)
                lookup_id = self.lookup_ids[nn_i]
                lookup_label = self.lookupLabels[lookup_id]
                query_label = self.queryLabels[query_id]

                predictions.append(
                    (query_id, query_label, lookup_id, lookup_label, nn_d, nn_iter))
        end = time.time()
        print("Computing NN took: {:.4f}[s]".format(end-start))
        return predictions

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""  #创建并返回ArgumentParser对象

    # Instantiate the parser  实例化parser（设置相关参数）
    parser = argparse.ArgumentParser(description=(
        """    
                eat.py 使用embeddings间的欧几里得距离 来传递注解
                从查找文档到查询文档。
                输入（查找和查询文档）可以作为原始蛋白质串行文档（*.fasta）或使用您自己的预计算嵌入 (*.h5)。
                如果您只提供 FASTA 文档，则默认情况下会从 ProtT5（通用 EAT）生成嵌入。
                如果你想使用 ProtTucker(ProtT5) 来转移注释（对远程结构同系物有用），设置 'use_tucker' 为 1。
                如果您不提供单独的标签文档将 fasta 标头链接到注释（可选），则查找文档中蛋白质的 fasta 标头的 ID被解释为标签。
                例如，如果您传递一个 FASTA 文档，蛋白质标头将从查找查询。如果您传递 H5 文档，则会传输来自预先计算的嵌入的密钥。
                提供您自己的标签文档通常需要您实现自己的解析功能。
                默认情况下，仅推断最近邻。这可以使用 --num_NN 参数更改。                
                如果您还通过 --queryLabels 为查询传递标签，则可以计算 EAT 性能。
            """
    ))

    # 参数含义：
            # required：可选参数是否可以省略 (仅针对可选参数)。
            # type： 命令行参数应该被转换成的类型。
            # help：参数的帮助信息
            # default： 不指定参数时的默认值。

    # Required positional argument 需要的位置参数
    parser.add_argument('-l', '--lookup', required=True, type=str,
                        help='A path to your lookup file, stored either as fasta file (*.fasta) OR' +
                        'as pre-computed embeddings (H5-format; *.h5).')  #查找文件的路径，存储为fasta文件(.fasta)或作为预计算嵌入(h5格式;.h5)。

    # Optional positional argument 可选的位置参数
    parser.add_argument('-q', '--queries', required=True, type=str,
                        help='A path to your query file, stored either as fasta file (*.fasta) OR' +
                        'as pre-computed embeddings (H5-format; *.h5).')  #查询文件的路径，存储为fasta文件(.fasta)或作为预计算嵌入(h5格式;.h5)。

    # Optional positional argument
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='A path to folder storing EAT results.')   #存储EAT结果的文件夹路径。

    # Required positional argument
    parser.add_argument('-a', '--lookupLabels', required=False, type=str,
                        default=None,
                        help='A path to annotations for the proteins in the lookup file.' +
                        'Should be a CSV with 1st col. having protein IDs as stated in FASTA/H5 file and' +
                        '2nd col having labels.For example: P12345,Nucleus')  #查找文档中蛋白质注释的路径。 应该是带有FASTA/H5文档中所述的蛋白质ID的第一列的CSV，并且有标签的第二列。例如：P12345，细胞核

    # Optional positional argument
    parser.add_argument('-b', '--queryLabels', required=False, type=str,
                        default=None,
                        help='A path to annotations for the proteins in the query file. ' +
                        'Same format as --lookupLabels. Needed for EAT accuracy estimate.') #查询文档中蛋白质注释的路径。与 --lookupLabels 的格式相同。EAT准确性估计需要

    parser.add_argument('--use_tucker', type=int,
                        default=0,
                        help="Whether to use ProtTucker(ProtT5) to generate per-protein embeddings." +
                        " Default: 0 (no tucker).") #是否使用ProtTucker（ProtT5）生成每个蛋白质的嵌入。  默认值：0（no tucker）

    parser.add_argument('--num_NN', type=int,
                        default=1,
                        help="The number of nearest neighbors to retrieve via EAT." +
                        "Default: 1 (retrieve only THE nearest neighbor).") #通过 EAT 检索的最近邻居的数量。 默认值：1（仅检索最近的邻居）

    parser.add_argument('--threshold', type=float,
                        default=None,
                        help="The Euclidean distance threshold below which nearest neighbors are retrieved via EAT." +
                        "Default: None (retrieve THE nearest neighbor, irrespective of distance).") #欧几里得距离阈值，低于该阈值通过 EAT 检索最近的邻居。 默认值：无（检索最近的邻居，无论距离如何）

    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args(['--lookup','/home/featurize/work/protECdata/clustered_train_df.fasta','--queries','/home/featurize/work/EAT-main/data/example_data_subcell/price.fasta',
                              '--lookupLabels','/home/featurize/work/protECdata/EClist-0222train.txt','--queryLabels','/home/featurize/work/EAT-main/data/example_data_subcell/price.txt','--output','eat_results/',
                               '--use_tucker','1'])
    #使用parse_args(参数解析器)解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。

    lookup_p = Path(args.lookup)
    query_p = Path(args.queries)
    output_d = Path(args.output)

    lookupLabels_p = None if args.lookupLabels is None else Path(
        args.lookupLabels)
    queryLabels_p = None if args.queryLabels is None else Path(
        args.queryLabels)

    num_NN = int(args.num_NN)
    threshold = float(args.threshold) if args.threshold is not None else None
    assert num_NN > 0, print("Only positive number of nearest neighbors can be retrieved.") # 断言确保 num_NN > 0，如果不大于0，会输出错误信息。

    use_tucker = int(args.use_tucker)
    use_tucker = False if use_tucker == 0 else True
    
    start=time.time()
    eater = EAT(lookup_p, query_p, output_d,
                use_tucker, num_NN, lookupLabels_p, queryLabels_p)  # EAT 处理蛋白质嵌入数据，执行最近邻搜索并生成预测结果

    predictions = eater.get_NNs(threshold=threshold)                # 获取预测结果
    eater.write_predictions(predictions)                            # 保存预测结果
    end=time.time()

    print("Total time: {:.3f}[s] ({:.3f}[s]/protein)".format(
        end-start, (end-start)/len(eater.query_ids)))
    
    if queryLabels_p is not None:
        print("Found labels to queries. Computing EAT performance ...")
        evaluator = Evaluator(predictions)
        evaluator.compute_performance()                             # 该段代码检查是否提供了查询标签文件
                                                                    # 如果提供了，创建一个 Evaluator 对象，然后调用 compute_performance 方法来评估模型的性能
    return None

if __name__ == '__main__':
    main()