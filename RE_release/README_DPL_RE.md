This document describes the data and source code of Deep Probabilistic Logic for Cross-Sentence N-ary Relation Extraction.

Our code is tested on python 2.7 with Theano.


### Experiments ###

To replicate our experiments, please go to "script" folder and run:

bash batch_run_lstm_dummy.sh (corresponds to Table 2, for each specific setting, please see comments in batch_run_lstm_dummy.sh)

This script will use data under data/drug_gene_var_dummy folder, with the training set comprising of instances in subfolders 0/1/2/3/4, the dev set in subfolder 5, same as in Peng et al. (TACL-17). The test set is in subfolder 6, which contains all instances from PubMed.

"dummy" means: We replace all drug/gene/mutation entities with dummy variables, thus preventing the model from memorizing the training instances.  

Indirect supervision is specified in "theano_src/indirect_supervision.py", which is used to build the factor graph.

To make predictions on a different set of instances, you can replace the data in subfolder 6 and re-run the script.

Results can be found under the "results" folder. See below.


### Code organization ###

data: contains the preprocessed data

     temp: 
         empty, some temporary files will be saved here when you run message passing
  
     drug_gene_var_dummy/: 
         main pubmed data used, by default, folder 0-4 used for training, 5 used as dev and 6 used for pudmed-scale wild prediction, folder 0-4 has same file structure, folder 5,6 only contain necessary files for predicting.
         for specific format for each file, please see data format section.
       0/
           data_graph: graph file from SPLAT, the same with TACL 2017 paper.
           sentences_2nd: label generated with distant supervision only, which is used in TACL 2017
           graph_arcs: dependency structure extracted from graph file

           sentences_3nd_f.1 sentences_3nd_f.2 sentences_3nd_f.3: label generated from different levels of indirect supervision, those labels are slightly different from sentences_2nd in the sense that they generated with more supervisions. 
           factorgraph_1.pkl factorgraph_2.pkl factorgraph_3.pkl: the saved factorgraph for each instance, which will be used in EM to avoid re-building the factorgraph in each iteration.
          
           sentences_3nd_f.1: soft labels generated from distant supervision + mention level prior; sentences_3nd_f.2: soft labels generated from distant supervision + mention level prior + relation level prior; sentences_3nd_f.3: soft labels generated from distant supervision + mention level prior + relation level prior + joint inference.
           factorgraph_1.pkl: factor graph generated from distant supervision + mention level prior; factorgraph_2.pkl: factor graph generated from distant supervision + mention level prior + relation level prior; factorgraph_3.pkl: factor graph generated from distant supervision + mention level prior + relation level prior + joint inference. 

           To generate those files from scratch, please see later section, and those different levels of prior corresponding to different configurations in Table 2.

       1/
       2/
       3/
       4/
       5/ 
          graph_arcs: dependency structure extracted from graph file
          sentences_2nd: sentence with human annotation, the same format with sentences_2nd file in folder 0-4
       6/
         graph_arcs: dependency structure extracted from graph file
         sentences_2nd: sentence with human annotation, the same format with sentences_2nd file in folder 0-4
         sentences_raw: raw sentences, which will be used for human annotations
 
      glove: glove embedding
      freq_uni.pkl: unigram frequency calculated from corpus, dict structure, e.g.,{"ER": xx, "pdf": xx}, the frequency information in this file is used as one prior. 
      gene_key.pkl: contains gene id information, dict structure, e.g. PDF, gene id: HGNC: 30012 Entrez Gene: 64146, this file is used for manually inspecting the prediction.
      entity_type.pkl: gene entities frequency calculated from corpus, dict structure, e.g., {'PTGES3': 4.522936943977144e-06, 'GAK': 5.025485493307938e-07}, the frequency information in this file is used as one prior.

scripts: contains the main shell script to run the training 
     
theano_src: contains the main python files and some shell scripts
    
     python files:
       neural_lib.py: core module for graphlstm 
       active_select_sample.py: dynamically select the samples, and adding it to the training
       data_process.py: process the data
       Relation_Extraction_Estep.py: E step, mainly to the message passing
       neural_architectures.py: core module for graphlstm
       lstm_RE.py: main script for training
       train_util.py: utils function for training
       message_passing.py: wrapper for message passing
       indirect_supervision.py: define various indirect supervisions, and build the factor graph given the input
       generate_factorgraph.py: wrapper to generate the factor graph
       utils.py: utility functions
       generate_factorgraph_chunks.py: wrapper to generate the factors
       KB.py: wrapper for knowledge base
       info_parser.py: parse all kind of information given instance and attribute query
       factorgraph.py: core module for message passing, add nodes to graph, delete nodes from graph etc 

     shell:

       cluster related submission code:
         submit_message_passing.sh: cluster submission code for message passing
         submit_generate_factorgraph.sh: cluster submission code for generating factorgraph
         submit.sh: shell for running the message passing file
         submit_generate_factorgraph2.sh: cluster submission code for generating factorgraph 2

       wrapper for python file:
         Active_select_sample.sh: shell for running the active sample selection 
         generate_factor.sh: shell for running generate_factorgraph.py
         Relation_Extraction_Estep.sh: shell for run Relation_Extraction_Estep.py, which will be called in training
         generate_factorgraph2.sh: shell for running generate_factorgraph_chunks.py 
         generate_factorgraph.sh: shell for running generate_factorgraph_chunks.py

       main shell to generate the factorgraph
         generate_allfactors.sh: shell for running the factor graph and message passing in the first iteration


annotations: contains some instances we annotated

results: contains the training log and predicted results

      Nary_param_and_predictions_olddata_dummy: xx.predictions: predicted probability for each instance in test set, format: id, \tab, p(x=1, i.e., response), to measure the precision&recall, we can sample from this file and annotate it (some annotations are given), trained model will also be saved in this folder.

      Nary_results_olddata_dummy: training log for each experiment.


#### Factor Graph ###

If you want to generate the factor graph ( data/drug_gene_var_dummy/0|1|2|3|4/factorgraph_1|2|3.pkl and data/drug_gene_var_dummy/0|1|2|3|4/sentences_3nd_f.1|2|3 ) from scratch, you can run following script (see comments in that shell for details):
bash theano_src/generate_allfactors.sh 

This will use various sets of indirect supervision to generate factorgraph_1|2|3.pkl and sentences_3nd_f.1|2|3 in the corresponding data folder, and the soft labels (marginal probabilities) for each instance will be saved in sentences_3nd_f.1|2|3. 

Note that generate_allfactors.sh will generate factor graph for each instance, which takes time. Our current code assumes a linux cluster with specific setup (see the submission scripts). If you want to run it in a different computing environment, you need to modify the E_step in the python code (LSTM_Re.py).

We use message passing for inference in the factor graph, which will generate temporary files in data/temp/XXXXXXXXXXXX, please delete them upon completion.


### Preprocessing & Data Format ###

We processed the source data into the format that is easier for our code to consume, which includes two files: "sentences_2nd" and "graph_arcs". The "sentences_2nd" file contains the dummmified input, in the same format as in TACL 2017 paper, except we changing the hard labels to soft ones:

e.g.
( a ) Workflow summarizes CTC based <ANNO_TYPE_gene> mutational analysis using patients’ blood samples , starting from thermoresponsive CTC purification of blood samples , via PCR amplifications and QC of CTC derived DNA , to Sanger sequencing targeting L858R <ANNO_TYPE_variant> <ANNO_TYPE_variant> point mutations in EGFR gene . ( b ) Three computed tomography ( CT ) scans of patient 6 taken at the timings of ( I ) heavy tumor burden before the <ANNO_TYPE_drug> <ANNO_TYPE_drug> treatment , ( II ) tumor shrinkage 3 months post-treatment , and ( III ) tumor relapse as a result of developing resistance to gefitinib .   73 74   7   40 39   0.32000002265 0.680000007153

0.32000002265 0.680000007153 is the probability for two classes (no-response, response).

the-original-sentences<TAB>indices-to-the-first-entity(drug)<TAB>indices-to-the-second-entity(gene/variant)[<TAB>indices-to-the-third-entity(variant)]<TAB>relation-label   

"sentences_3nd.f_xx" file contains the information with updated probability, and the format is the same with "sentences_2nd".

"sentences_raw" file contains the original sentences, we will need them when building the factor graph.

The "graph_arcs" file contains the information of the dependencies between the words, including time sequence adjacency, syntactic dependency, and discourse dependency. The format is:
dependencies-for-node-0<WHITESPACE>dependencieiis-for-node-1...
dependencies-for-node-n = dependency-0,,,dependency-1...
dependency-n = dependency-type::dependent-node

If you have additional question on Graph LSTM, please see README_GraphLSTM.md from Peng et al.


#### Citation ###
If you find this code is useful, please cite the following paper:

@inproceedings{wang2018deep,
  title={Deep Probabilistic Logic: A Unifying Framework for Indirect Supervision},
  author={Wang, Hai and Poon, Hoifung},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={1891--1902},
  year={2018}
}
