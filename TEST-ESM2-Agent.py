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

        self.embed_to_img = nn.Linear(input_dim, embed_dim * window * window)
        
        # AgentAttention
        self.agent_attention = AgentAttention(dim=embed_dim, num_heads=8, 
                                              qkv_bias=False, attn_drop=0., proj_drop=0.,
                                              agent_num=49, window=window)

        self.protTucker = nn.Sequential(
            nn.Linear(1280, 256),         
            nn.Tanh(),                     
            nn.Linear(256, 128),           
        )
        self.adjust_dim = nn.Linear(embed_dim * window * window, 256)

    def single_pass(self, X):
        X = X.float()
        return self.protTucker(X)

    def forward(self, X):                        # X (256, 3, 1280)
        batch_size = X.shape[0]

        X_embedded = self.embed_to_img(X.view(-1, X.size(-1)))
        X_reshaped = X_embedded.view(batch_size * 3, self.window * self.window, self.embed_dim)  #  [batch_size * 3, num_patches, embed_dim]
        

        attention_output = self.agent_attention(X_reshaped)
        
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
                                                      ) 

    def compute_performance(self):
        error_estimates = self.compute_err()
        for metric, (performance, bootstrap_err) in error_estimates.items(): 
            print("{}={:.3f} +/-{:.3f}".format(
                metric,
                performance,
                1.96*np.std(np.array(bootstrap_err), ddof=1)    
            ) 
            )

        return None

    def compute_err(self, n_bootstrap=1000): 
                                            
        n_total = len(self.Ys)  # total number of predictions 
        idx_list = range(n_total)

        Ys, Yhats = np.array(self.Ys), np.array(self.Yhats)
        acc = accuracy_score(Ys, Yhats)
        f1 = f1_score(Ys, Yhats, average="weighted") #average="weighted"  
        bAcc = balanced_accuracy_score(Ys, Yhats)
        cm = confusion_matrix(Ys, Yhats)
        print("Confusion Matrix:")
        print(cm)

        
        accs_btrap, f1s_btrap, bAccs_btrap = list(), list(), list() 
        # accs_btrap, f1s_btrap, bAccs_btrap, pres_btrap, recalls_btrap= list(), list(), list(),list(), list()

        n_skipped = 0
        for _ in range(n_bootstrap):
            rnd_subset = random.choices(idx_list, k=n_total) 
            # skip bootstrap iterations where predictions might hold labels not part of groundtruth 

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
        self.embedder, self.tokenizer = self.get_esm1b()       

    def get_esm1b(self):
        start=time.time()
        # Load your checkpoint here  
        # Currently, only the encoder-part of ProtT5 is loaded in half-precision  
        from transformers import AutoModel, AutoTokenizer
        print("Start loading ESM2...")
        model_name = "facebook/esm2_t33_650M_UR50D"
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        print("Finished loading {} in {:.1f}[s]".format(model_name,time.time()-start))
        return model, tokenizer
    
    def write_embedding_list(self,emb_p, ids,embeddings):                      
        embeddings=embeddings.detach().cpu().numpy().squeeze()                  
        with h5py.File(str(emb_p),"w") as hf:
            for idx, seq_id in enumerate(ids):
                hf.create_dataset(seq_id,data=embeddings[idx])
        return None
                
    def write_embeddings(self, emb_p, embds):                                 
        with h5py.File(str(emb_p), "w") as hf:
            for sequence_id, embedding in embds.items():
                # noinspection PyUnboundLocalVariable
                hf.create_dataset(sequence_id, data=embedding)
        return None
    
    def get_embeddings_batch(self, id2seq, max_residues=4000, max_seq_len=1000, max_batch=1):   
        print("Start generating embeddings for {} proteins.".format(len(id2seq)) +
              "This process might take a few minutes." +
              "Using batch-processing! If you run OOM/RuntimeError, you should use single-sequence embedding by setting max_batch=1.")
        start = time.time()
        ids = list()
        embeddings = list()
        batch = list()
        
        id2seq = sorted( id2seq.items(), key=lambda kv: len( id2seq[kv[0]] ), reverse=True ) 
       
        for seq_idx, (protein_id, original_seq) in enumerate(id2seq):              
            seq = original_seq.replace('U','X').replace('Z','X').replace('O','X')  
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((protein_id,seq,seq_len))      
            
            
            n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len   

            
            if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(id2seq) or seq_len>max_seq_len:
                protein_ids, seqs, seq_lens = zip(*batch)      
                batch = list()

                token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
      
                try:
                    with torch.no_grad():
                        # get embeddings extracted from last hidden state  
                        batch_emb = self.embedder(input_ids, attention_mask=attention_mask).last_hidden_state # [B, L, 1024]
                except RuntimeError as e :
                    print(e)
                    print("RuntimeError during embedding for {} (L={})".format(protein_id, seq_len))
                    continue
                
                for batch_idx, identifier in enumerate(protein_ids):
                    s_len = seq_lens[batch_idx]
                    emb = batch_emb[batch_idx,:s_len].mean(dim=0,keepdims=True)
                    ids.append(protein_ids[batch_idx])
                    embeddings.append(emb.detach())

        print("Creating per-protein embeddings took: {:.1f}[s]".format(time.time()-start))
        embeddings = torch.vstack(embeddings)          
        return ids, embeddings


# EAT: Embedding-based Annotation Transfer 
class EAT():
    def __init__(self, lookup_p, query_p, output_d, use_tucker, num_NN,
                 lookupLabels, queryLabels):

        self.output_d = output_d
        Path.mkdir(output_d, exist_ok=True) 
        
        self.num_NN = num_NN
        self.Embedder = None               
        
        self.lookup_ids, self.lookup_embs = self.read_inputs(lookup_p)     
        self.query_ids, self.query_embs = self.read_inputs(query_p)

        if use_tucker:  # create ProtTucker(ProtT5) embeddings
            
          
            self.lookup_embs = self.lookup_embs.to(torch.float)             
            self.query_embs = self.query_embs.to(torch.float)

            self.lookup_embs = self.tucker_embeddings(self.lookup_embs)     
            self.query_embs = self.tucker_embeddings(self.query_embs)

        self.lookupLabels = self.read_label_mapping(self.lookup_ids, lookupLabels)  
        self.queryLabels = self.read_label_mapping(self.query_ids, queryLabels)

    def tucker_embeddings(self, dataset):                   
        weights_p = self.output_d / "weights-EC(ESM2-agent).pt"

        # if no pre-trained model is available, yet --> download it  
        if not weights_p.exists():
            import urllib.request
            print("No existing model found. Start downloading pre-trained ESM2...")
            weights_link = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"           
            urllib.request.urlretrieve(weights_link, str(weights_p))

        print("Loading Tucker checkpoint from: {}".format(weights_p))
        state = torch.load(weights_p)['state_dict']            

        model = ProtTucker().to(device)                     
        model.load_state_dict(state)                   
        model=model.eval()                             

        start = time.time()
        dataset = model.single_pass(dataset)
        print("Tuckerin' took: {:.4f}[s]".format(time.time()-start))
        return dataset

    def read_inputs(self, input_p):                       
        # define path for storing embeddings 

        if not input_p.is_file():
            print("Neither input fasta, nor embedding H5 could be found for: {}".format(input_p))
            print("Files are expected to either end with .fasta or .h5")
            raise FileNotFoundError

        if input_p.name.endswith(".h5"): # if the embedding file already exists 
            return self.read_embeddings(input_p)
        
        elif input_p.name.endswith(".fasta"): # compute new embeddings if only FASTA available
            if self.Embedder is None: # avoid re-loading the pLM
                self.Embedder = Embedder()
            id2seq = self.read_fasta(input_p)                               
            
            ids, embeddings = self.Embedder.get_embeddings_batch(id2seq)    
                                                                            
            emb_p  = self.output_d / input_p.name.replace(".fasta", ".h5")  
            self.Embedder.write_embedding_list(emb_p, ids,embeddings)      
            return ids, embeddings
        else:
            print("The file you passed neither ended with .fasta nor .h5. " +
                  "Only those file formats are currently supported.")
            raise NotImplementedError

    def read_fasta(self, fasta_path):                      
       
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

    def read_embeddings(self, emb_p):                     
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

    def read_label_mapping(self, set_ids, label_p):         
        if label_p is None:
            return {set_id: None for set_id in set_ids}
  
        with open(label_p, 'r') as in_f:
            # protein-ID : label
            label_mapping = {line.strip().split(
                ' ')[0]: line.strip().split('EC:')[1] for line in in_f}
        return label_mapping

    def write_predictions(self, predictions):              
        out_p = self.output_d / "eat_resultEC-ESM2-Attention.txt"
        with open(out_p, 'w+') as out_f:
            out_f.write(
                "Query-ID\tQuery-Label\tLookup-ID\tLookup-Label\tEmbedding distance\tNearest-Neighbor-Idx\n")
            out_f.write("\n".join(
                ["{}\t{}\t\t{}\t\t{}\t\t{:.4f}\t\t{}".format(query_id, query_label, lookup_id, lookup_label, eat_dist, nn_iter+1)
                 for query_id, query_label, lookup_id, lookup_label, eat_dist, nn_iter in predictions
                 ]))
        return None

    def pdist(self, lookup, queries, norm=2, use_double=False):    
        lookup=lookup.unsqueeze(dim=0)
        queries=queries.unsqueeze(dim=0)            

        # double precision improves performance slightly but can be removed for speedy predictions (no significant difference in performance)
        if use_double:
            lookup=lookup.double()
            queries=queries.double()               

        try: # try to batch-compute pairwise-distance on GPU   
            pdist = torch.cdist(lookup, queries, p=norm).squeeze(dim=0)  
        except RuntimeError as e:
            print("Encountered RuntimeError: {}".format(e))
            print("Trying single query inference on GPU.") 
            try: # if OOM for batch-GPU, re-try single query pdist computation on GPU  
                pdist = torch.stack(
                    [torch.cdist(lookup, queries[0:1, q_idx], p=norm).squeeze(dim=0)
                     for q_idx in range(queries.shape[1])
                     ]
                ).squeeze(dim=-1).T

            except RuntimeError as e: # if OOM for single GPU, re-try single query on CPU  
                print("Encountered RuntimeError: {}".format(e))
                print("Trying to move single query computation to CPU.") 
                lookup=lookup.to("cpu")
                queries=queries.to("cpu")
                pdist = torch.stack(
                    [torch.cdist(lookup, queries[0:1, q_idx], p=norm).squeeze(dim=0)
                     for q_idx in range(queries.shape[1])
                     ]
                ).squeeze(dim=-1).T
                
        print(pdist.shape)
        return pdist                   

    def get_NNs(self, threshold, random=False):                
    #def get_NNs(self, threshold, random=True):
        start = time.time()
        p_dist = self.pdist(self.lookup_embs, self.query_embs)      
                                                               

        if random: # this is only needed for benchmarking against random background 
            print("Making RANDOM predictions!")
            nn_dists, nn_idxs = torch.topk(torch.rand_like(
                p_dist), self.num_NN, largest=False, dim=0)
        else: # infer nearest neighbor indices 
            nn_dists, nn_idxs = torch.topk(
                p_dist, self.num_NN, largest=False, dim=0)   
            
        print("Computing NN took: {:.4f}[s]".format(time.time()-start))
        nn_dists, nn_idxs = nn_dists.to("cpu"), nn_idxs.to("cpu")

       
        predictions = list()       
        n_test = len(self.query_ids)
        for test_idx in range(n_test): 
            query_id = self.query_ids[test_idx] 
            nn_idx = nn_idxs[:, test_idx]      
            nn_dist = nn_dists[:, test_idx]    

            for nn_iter, (nn_i, nn_d) in enumerate(zip(nn_idx, nn_dist)):
                # index of nearest neighbour (nn) in train set 
                nn_i, nn_d = int(nn_i), float(nn_d)

                # if a threshold is passed, skip all proteins above this threshold  
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
    # Instantiate the parser  
    parser = argparse.ArgumentParser(description=(
        """    
               eat.py uses the Euclidean distance between embeddings to pass annotations
               From finding documents to querying documents.
               The input (find and query documents) can be as raw protein serial documents (*.fasta) or using your own precomputed embeddings (*.h5).
               If you only provide FASTA documents, embeddings are generated from ProtT5 (generic EAT) by default.
               If you want to use ProtTucker(ProtT5) to transfer annotations (useful for remote structural homologs), set 'use_tucker' to 1.
               If you do not provide a separate tag document linking the fasta header to the annotation (optional), the ID of the fasta header for finding the protein in the document is interpreted as a tag.
               For example, if you pass a FASTA document, the protein header will be queried from the lookup. If you pass an H5 document, the key from the precomputed embedding is transmitted.
               Providing your own labeled documents usually requires you to implement your own parsing capabilities.
               By default, only the nearest neighbors are inferred. This can be changed using the --num_NN parameter.
               If you also pass labels for the query via --queryLabels, you can calculate EAT performance.
            """
    ))

    # Required positional argument 
    parser.add_argument('-l', '--lookup', required=True, type=str,
                        help='A path to your lookup file, stored either as fasta file (*.fasta) OR' +
                        'as pre-computed embeddings (H5-format; *.h5).')  

    # Optional positional argument 
    parser.add_argument('-q', '--queries', required=True, type=str,
                        help='A path to your query file, stored either as fasta file (*.fasta) OR' +
                        'as pre-computed embeddings (H5-format; *.h5).') 

    # Optional positional argument
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='A path to folder storing EAT results.')   

    # Required positional argument
    parser.add_argument('-a', '--lookupLabels', required=False, type=str,
                        default=None,
                        help='A path to annotations for the proteins in the lookup file.' +
                        'Should be a CSV with 1st col. having protein IDs as stated in FASTA/H5 file and' +
                        '2nd col having labels.For example: P12345,Nucleus') 

    # Optional positional argument
    parser.add_argument('-b', '--queryLabels', required=False, type=str,
                        default=None,
                        help='A path to annotations for the proteins in the query file. ' +
                        'Same format as --lookupLabels. Needed for EAT accuracy estimate.') 

    parser.add_argument('--use_tucker', type=int,
                        default=0,
                        help="Whether to use ProtTucker(ProtT5) to generate per-protein embeddings." +
                        " Default: 0 (no tucker).") 

    parser.add_argument('--num_NN', type=int,
                        default=1,
                        help="The number of nearest neighbors to retrieve via EAT." +
                        "Default: 1 (retrieve only THE nearest neighbor).")

    parser.add_argument('--threshold', type=float,
                        default=None,
                        help="The Euclidean distance threshold below which nearest neighbors are retrieved via EAT." +
                        "Default: None (retrieve THE nearest neighbor, irrespective of distance).") 

    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args(['--lookup','/home/featurize/work/protECdata/clustered_train_df.fasta','--queries','/home/featurize/work/EAT-main/data/example_data_subcell/price.fasta',
                              '--lookupLabels','/home/featurize/work/protECdata/EClist-0222train.txt','--queryLabels','/home/featurize/work/EAT-main/data/example_data_subcell/price.txt','--output','eat_results/',
                               '--use_tucker','1'])
 

    lookup_p = Path(args.lookup)
    query_p = Path(args.queries)
    output_d = Path(args.output)

    lookupLabels_p = None if args.lookupLabels is None else Path(
        args.lookupLabels)
    queryLabels_p = None if args.queryLabels is None else Path(
        args.queryLabels)

    num_NN = int(args.num_NN)
    threshold = float(args.threshold) if args.threshold is not None else None
    assert num_NN > 0, print("Only positive number of nearest neighbors can be retrieved.") 

    use_tucker = int(args.use_tucker)
    use_tucker = False if use_tucker == 0 else True
    
    start=time.time()
    eater = EAT(lookup_p, query_p, output_d,
                use_tucker, num_NN, lookupLabels_p, queryLabels_p)  

    predictions = eater.get_NNs(threshold=threshold)                
    eater.write_predictions(predictions)                          
    end=time.time()

    print("Total time: {:.3f}[s] ({:.3f}[s]/protein)".format(
        end-start, (end-start)/len(eater.query_ids)))
    
    if queryLabels_p is not None:
        print("Found labels to queries. Computing EAT performance ...")
        evaluator = Evaluator(predictions)
        evaluator.compute_performance()                             
                                                                   
    return None

if __name__ == '__main__':
    main()