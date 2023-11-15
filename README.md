
# [exBERTSum] Leveraging Large Language Model with Domain Adaption: Enhancing Community Directories with Deep Learning Text Summarization for Machine-Readable Cataloging (MARC) Standard Description Notes

This project is a part of the Master of AI and Machine Learning Research.  Other projects can be found at the [main GitHub repo](https://github.com/camillekokoko/exBERT). Presentation can be found at the [youtube link](https://www.youtube.com/watch?v=6KChrujZ3_4)

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


## Reference

**[Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)**
**[github](https://github.com/cgmhaicenter/exBERT](https://github.com/nlpyang/PreSumm/tree/master)**





Results on CNN/DailyMail (20/8/2019):


<table class="tg">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="4">Extractive</td>
  </tr>
  <tr>
    <td class="tg-0pky">TransformerExt</td>
    <td class="tg-0pky">40.90</td>
    <td class="tg-0pky">18.02</td>
    <td class="tg-0pky">37.17</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt</td>
    <td class="tg-0pky">43.23</td>
    <td class="tg-0pky">20.24</td>
    <td class="tg-0pky">39.63</td>
  </tr>
  <tr>
    <td class="tg-0pky">BertSumExt (large)</td>
    <td class="tg-0pky">43.85</td>
    <td class="tg-0pky">20.34</td>
    <td class="tg-0pky">39.90</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="4">Abstractive</td>
  </tr>
  <tr>
    <td class="tg-0lax">TransformerAbs</td>
    <td class="tg-0lax">40.21</td>
    <td class="tg-0lax">17.76</td>
    <td class="tg-0lax">37.09</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumAbs</td>
    <td class="tg-0lax">41.72</td>
    <td class="tg-0lax">19.39</td>
    <td class="tg-0lax">38.76</td>
  </tr>
  <tr>
    <td class="tg-0lax">BertSumExtAbs</td>
    <td class="tg-0lax">42.13</td>
    <td class="tg-0lax">19.60</td>
    <td class="tg-0lax">39.18</td>
  </tr>
</table>



#### Step 1 Download Stories
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files in one directory (e.g. `../raw_stories`)

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory. 

####  Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


####  Step 4. Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -n_cpus 1 -use_bert_basic_tokenizer false -map_path MAP_PATH
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

####  Step 5. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 1 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)


