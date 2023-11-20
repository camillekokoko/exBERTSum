
# [exBERTSum] Leveraging Large Language Model with Domain Adaption: Enhancing Community Directories with Deep Learning Text Summarization for Machine-Readable Cataloging (MARC) Standard Description Notes

This project is a part of the Master of AI and Machine Learning Research.  Other projects can be found at the [main GitHub repo](https://github.com/camillekokoko/exBERT). Presentation can be found at the [youtube link](https://www.youtube.com/watch?v=srKsCCE_l8Y)

#### -- Project Status: [Active, On-Hold, Completed]

## Project Intro/Objective
The purpose of this project is to build a system utilizing Natural Language Processing (NLP) to generate informative summaries from community's directories in South Australia, accompanied by machine-readable cataloging in adherence to the MARC standard's description notes (Library of Congress 2023).

### Partner
* [SACommunity](https://sacommunity.org/)

### Methods Used
* Generative AI
* Finetune Large Language Models 
* Pretrain Transformer BERT
* Natural Language Processing
* Data Visualization
* etc.

### Technologies
* Python
* Pandas, jupyter
* Pytorch
* Docker
* MySql
* YAML
* JSON
* etc. 

## Project Description (to completed)
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Getting Started (to be completed)

1. Clone this repo 
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. Command Scripts are being kept [here]()

## Pre-train an exBERTSum model 
In command line:

    python train.py \
    -task ext -mode train \
    -bert_data_path ../bert_data/ \
    -ext_dropout 0.1 \
    -model_path ../models/bertbase \
    -lr 2e-3 \
    -visible_gpus 0 \
    -report_every 1000 \
    -save_checkpoint_steps 1000 \
    -batch_size 32 \
    -train_steps 1000 \
    -accum_count 2 \
    -log_file ../logs/ext_bertbaseCs2.log \
    -use_interval true \
    -warmup_steps 10000 \
    -max_pos 512 \
    -exbert True \
    -finetune False \
    -config2 ./bert_config_ex_s2.json \
    -checkpoint_path ./models_Presumm/Cs2_Best_stat_dic_exBERTe2_b32_lr0.0001.pth

## Validation/Test an exBERTSum model 
In command line:

    python train.py \
     -task ext \
     -mode test \
     -batch_size 32 \
     -test_batch_size 32 \
     -bert_data_path ../bert_data_story_files/ \
     -model_path ../models/bertbase \
     -test_from ../models/bertbase/bert_config_ex_Cs2.json_0.002_32_exbert_model_step_1000.pt
     -log_file ../logs/test_ext_bertbaseCs2_testtest.log \
     -result_path ../results/ext_bertbase \
     -sep_optim true \
     -use_interval true \
     -visible_gpus -1 \
     -max_pos 512 \
     -max_length 128 \
     -alpha 0.95 \
     -min_length 50 \
     -finetune_bert True \
     -exbert True \
     -config2 ./bert_config_ex_s2.json 


Results on SA Community Data: (to completed)
<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
 <tr>
    <td class="tg-0pky">Baseline</td>
    <td class="tg-0pky">64.05</td>
    <td class="tg-0pky">60.09</td>
    <td class="tg-0pky">61.11</td>
  </tr>
  <tr>
    <td class="tg-0pky">OS1</td>
    <td class="tg-0pky">65.59</td>
    <td class="tg-0pky">61.24</td>
    <td class="tg-0pky">63.92</td>
  </tr>
  <tr>
    <td class="tg-0pky">OS2</td>
    <td class="tg-0pky">66.10</td>
    <td class="tg-0pky">62.00</td>
    <td class="tg-0pky">64.32</td>
  </tr>
  <tr>
    <td class="tg-0pky">OS3</td>
    <td class="tg-0pky">65.02</td>
    <td class="tg-0pky">60.46</td>
    <td class="tg-0pky">63.17</td>
  </tr>
      <tr>
    <td class="tg-0pky">OS4</td>
    <td class="tg-0pky">65.85</td>
    <td class="tg-0pky">61.65</td>
    <td class="tg-0pky">64.18</td>
  </tr>
      <tr>
    <td class="tg-0pky">OS5</td>
    <td class="tg-0pky">64.33</td>
    <td class="tg-0pky">59.49</td>
    <td class="tg-0pky">62.38</td>
      </tr>
</table>


## Sentence Representation:
In command line:
        
        export CLASSPATH=/Users/camilleko/PreSumm/stanford-corenlp-4.5.5/stanford-corenlp-4.5.5.jar


        python preprocess.py -mode tokenize -raw_path ../raw_data/story_files/ -save_path ../merged_story_files_tokenized

        python preprocess.py -mode format_to_lines -raw_path ../merged_story_files_tokenized -save_path ../json_data_story_files/ -n_cpus 1 -use_bert_basic_tokenizer false -map_path ../urls


        python preprocess.py -mode format_to_bert -raw_path ../json_data_story_files/ -save_path ../bert_data_story_files/  -lower -n_cpus 1 -log_file ../logs/preprocess.log


        export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES 

        python preprocess.py -mode format_to_bert -raw_path ../json_data_1 -save_path ../bert_data_1  -lower -n_cpus 1 -log_file ../logs/preprocess.log


## Reference

**[Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**
**[github](https://github.com/cgmhaicenter/exBERT](https://github.com/nlpyang/PreSumm/tree/master)**





