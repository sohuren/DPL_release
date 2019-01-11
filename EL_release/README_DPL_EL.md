This document describes the data and source code of Deep Probabilistic Logic for entity linking.

Our code is tested on python 2.7 with pytorch 0.3.0.post4
   
#### Experiments ###

To replicate our experiments, try the following command:

bash run_experiment_EL.sh

This script will run the following three experiments:

  # run the distant supervision baseline, with the length+Distant Supervision as the supervision: 
  bash run_experiment_DS.sh # table.5, DS setting
  # run the DPL with different level of supervision:
  bash run_experiment_DPL.sh 1 (+ data programming on mention level) # table.5, DS + DP setting
  bash run_experiment_DPL.sh 3 (+ joint inference) # table.5, DS + DP + JI setting

The training script will first load the pkl file, which may take a while. The training log & models of the above three experiments will be save in data/pubmed_parsed/gene_result/0/, with the corresponding training logs as follows:
hard_3_full.log
soft_featureset_1_3_learngraph_M_full.log
soft_featureset_3_3_learngraph_M_full.log

In those log files, after each iteration, you can find result like this:
Val set: Average loss: xx, Accuracy: xx/xx (xx%), precision: (0.xx), recall: (0.xx), f1: (0.xx)

After training, you can test on other data by running:
python run_rnn_test.py

Alternatively, you can supply additional arguments to run predictions on new instances. The format of the input text should be: sent + \tab + mention index  

Indirect supervision is defined in "code_data/indirect_supervision.py".


### Code Organization ###

The code & data are in code_data folder.
 
   -baseline_rnn.py: main function for train model
   -visualizer.py: generate the html file for better visualization
   -text_folder_basic.py: data loader wrapper for pytorch
   -network.py: neural module
   -dependency_converter.py: convert the dependency parser information
   -utils.py: utils file
   -data_loader.py: data loader
   -visualizer2.py: generate the html file for better visualization (different format)
   -visualizer3.py: generate the html file for better visualization (different format)
   -base_data_loader.py: data loader
   -indirect_supervision.py: define the indirect supervision which can be used for building factorgraph
   -mlp_data_loader.py: data loader
   -baseline_rnn_test.py: main function for test
   -message_passing.py: message passing
   -baseline_rnn_test_BioC.py: test on BioC (do not use here)
   -info_parser.py: parser for all information (depedency parser etc)
   -rnn_data_loader.py: data loader
   -factorgraph.py: core module for factor graph
   -text_folder_rnn.py: data processing wrapper


   ########  utils files, no need to run it unless you want to run everything from scratch  ###############
   -generate_dataset.py: generate the dataset given the basic string matching reuslt
   -Calculate_uni_freq.py: until files for calculate the entity frqeuency
   -Calculate_entitytype_freq.py: same as above
   -build_vocab.py: pre-fetch the word embedding from glove
   -generate_traindata.py: generate the train data

   ######################################################################################################## 

   scripts used to do the message passing on cluster (see comments in shell for details):

   -submit_mp.sh
   -submit_data.sh
   -submit.sh
   -submit_generatedata.sh
 
   -generate_data_soft.sh
   -generate_data_pubmed.sh
   -run_experiment_hard_full.sh
   -run_experiment_soft_M_full.sh
   -generate_data.sh
   -run_experiment_soft.sh


   folder ./data contains the data:
   -annotation: contains our annotation in different formats(html, txt and pkl)
   -gene_key.pkl: contains the Official Gene Card ID for each mention (used for checking), format is the same with that in Relation Extraction 

   -pubmed_parsed: 
    -pubmed_parsed_splits_dict_0.pkl: the pre-computed string matching result with ontology, and the parsing information for each sentence from stanford corenlp
     format: dict structure, and data[key1][key2] where key1 is the trial id and key2 is the "inclusion/exclusion", for example, data['6582443"']['inc/exc'] has the following content: 
     [{'text': ['antamanide', 'prevents', 'bradykinin-lnduced', 'filamin', 'translocation', 'by', 'inhibiting', 'extracellular', 'calcium', 'influx', '.'], 'tree': {u'sentences': [{u'parsetree': u'[Text=prevents CharacterOffsetBegin=11 CharacterOffsetEnd=19 PartOfSpeech=VBZ Lemma=prevent NamedEntityTag=O] [Text=bradykinin-lnduced CharacterOffsetBegin=20 CharacterOffsetEnd=38 PartOfSpeech=JJ Lemma=bradykinin-lnduced NamedEntityTag=O] [Text=filamin CharacterOffsetBegin=39 CharacterOffsetEnd=46 PartOfSpeech=NN Lemma=filamin NamedEntityTag=O] [Text=translocation CharacterOffsetBegin=47 CharacterOffsetEnd=60 PartOfSpeech=NN Lemma=translocation NamedEntityTag=O] [Text=by CharacterOffsetBegin=61 CharacterOffsetEnd=63 PartOfSpeech=IN Lemma=by NamedEntityTag=O] [Text=inhibiting CharacterOffsetBegin=64 CharacterOffsetEnd=74 PartOfSpeech=VBG Lemma=inhibit NamedEntityTag=O] [Text=extracellular CharacterOffsetBegin=75 CharacterOffsetEnd=88 PartOfSpeech=JJ Lemma=extracellular NamedEntityTag=O] [Text=calcium CharacterOffsetBegin=89 CharacterOffsetEnd=96 PartOfSpeech=NN Lemma=calcium NamedEntityTag=O] [Text=influx CharacterOffsetBegin=97 CharacterOffsetEnd=103 PartOfSpeech=NN Lemma=influx NamedEntityTag=O] [Text=. CharacterOffsetBegin=104 CharacterOffsetEnd=105 PartOfSpeech=. Lemma=. NamedEntityTag=O] (ROOT (S (NP (NN antamanide)) (VP (VBZ prevents) (NP (JJ bradykinin-lnduced) (NN filamin) (NN translocation)) (PP (IN by) (S (VP (VBG inhibiting) (NP (JJ extracellular) (NN calcium) (NN influx)))))) (. .)))', u'text': [u'antamanide', u'prevents', u'bradykinin-lnduced', u'filamin', u'translocation', u'by', u'inhibiting', u'extracellular', u'calcium', u'influx', u'.'], u'dependencies': [[u'root', u'ROOT', u'prevents'], [u'nsubj', u'prevents', u'antamanide'], [u'amod', u'translocation', u'bradykinin-lnduced'], [u'compound', u'translocation', u'filamin'], [u'dobj', u'prevents', u'translocation'], [u'mark', u'inhibiting', u'by'], [u'advcl:by', u'prevents', u'inhibiting'], [u'amod', u'influx', u'extracellular'], [u'compound', u'influx', u'calcium'], [u'dobj', u'inhibiting', u'influx'], [u'punct', u'prevents', u'.']], u'words': [[u'antamanide', {u'NamedEntityTag': u'O', u'CharacterOffsetEnd': u'10', u'CharacterOffsetBegin': u'0', u'PartOfSpeech': u'NN', u'Lemma': u'antamanide'}]]}]}, 'matched': {'pp': [], 'input_anatomy': [(3, 3)], 'adtte': [], 'aessp': [], 'food': [], 'humanity': [], 'disease': [], 'drug': [(3, 3), (8, 8)], 'publication_char': [], 'named_group': [], 'acronym': [], 'health_care': [], 'info_sci': [], 'gene': [], 'geo': [], 'bacteria': [], 'dis_occ': []}}] 

    this structure contains parsing information from corenlp and string matching result with ontology, e.g., 'drug': [(3, 3), (8, 8)] specify this sentence has two drug mentions: 3th token and 8th token

    -vocab_gene.pkl: the vocab used in embedding layer, generated from "build_vocab.py"
    -embedding_vec_gene.pkl: the pre-trained embeedding from glove, generated from "build_vocab.py"
    -freq_uni.pkl: the frequency for each gene, format is the same with that in Relation Extraction

    -gene_0: contains the files saved the pre-computed factorgraph
       -0/hard: distant supervision(DS) result 
       -0/soft_featureset_1: DS + data programming at entity level
       -0/soft_featureset_2: DS + data programming at cross mention level
       -0/soft_featureset_3: DS + data programming at cross sentence level
       here _1, _2, _3 represents different level of supervision

       all these folders have the same structure: it contains train_xx, test_xx, validation_anno_full.pkl       
       "xx" means the length, e.g., train_3.pkl: means all gene with more than 3 chars
       validation_anno_full.pkl: is our annotation file, which we will use to measure the performance.
       to generate those files from scratch, please see later section.
 
    -gene_results: the trained model & training log & html for displaying the result

 
### Factor Graph ###

To generate the factor graph files (data/gene_0/soft_featureset_1|2|3/train|test.pkl etc) from scratch, please run:
bash generate_data_pubmed.sh

Note that this code was run on a cluster with specific setup. Please modify the submission scripts for your environment.

The script can be used for cross validation, in which case, $i, $j represent the folds. By default, we are not doing that, so they are just set to 0. 

$k represent the version of indirect supervision:
  k=1: + data programming on mention level
  k=2, + joint inference over mention inside each sentence
  k=3, + joint inference over mention cross sentences


### Citation ####
If you find this code is useful, please cite the following paper:

@inproceedings{wang2018deep,
  title={Deep Probabilistic Logic: A Unifying Framework for Indirect Supervision},
  author={Wang, Hai and Poon, Hoifung},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={1891--1902},
  year={2018}
}

