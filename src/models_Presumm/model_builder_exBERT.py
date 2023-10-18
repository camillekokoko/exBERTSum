import copy

import torch
import torch.nn as nn
from models_Presumm import exBERT
# from exBERT import BertModel, BertConfig # DO I NEED TO RECREATE FOR EXBERT????????? --> transformers-modeling_bert | transformers-configuration | transformers-tokenization
from exBERT import BertModelNew, BertConfigNew
from torch.nn.init import xavier_uniform_

# from models.decoder import TransformerDecoder
from models_Presumm.encoder import Classifier, ExtTransformerEncoder
from models_Presumm.optimizers import Optimizer

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

# class Bert(nn.Module):
#     def __init__(self, large, temp_dir, finetune=True):
#         super(Bert, self).__init__() 
#         if(large):
#             self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
#         else:
#             self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
#         self.finetune = finetune
#         print('print out ExtSummarizer', self)

    
class exBert_model(nn.Module):
    def __init__(self, args, config_2, checkpoint_path, finetune):
        super(exBert_model, self).__init__()
        self.args = args
        config_2 = args.config2  #'./bert_config_ex_s3.json'
        checkpoint_path = args.checkpoint_path #='./models_Presumm/Best_stat_dic_exBERTe2_b16_lr1e-05.pth'
        finetune = args.finetune_bert #default=True
        bert_config_1 = BertConfigNew.from_json_file('./bert_config.json')
        bert_config_2 = BertConfigNew.from_json_file(config_2)
        self.finetune = finetune
        print('finetune', self.finetune)
        
         
        self.model = BertModelNew(bert_config_1, bert_config_2)
        self.checkpoint = torch.load(checkpoint_path)
        
        

        # Freeze the BERT layers if not finetuning
        if not self.finetune:
            for param in self.model.parameters():
                param.requires_grad = False 

        # for param in self.model.bert.bert.parameters(): 
        #     param.requires_grad = False 

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, secondargument = self.model(x, segs, attention_mask=mask)
            # print('@@@@@ topvec', len(top_vec))
            # print('@@@@@ secondargument', len(secondargument))
        else:
            self.eval()
            with torch.no_grad():
                top_vec, secondargument = self.model(x, segs, attention_mask=mask)
                # print('no finetune @@@@@ topvec', len(top_vec))
                # print('no finetune @@@@@ secondargument', len(secondargument))
        return top_vec


class ExtSummarizer_exBERT(nn.Module):   #i change this
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer_exBERT, self).__init__() #i change this
        self.args = args
        self.device = device

        # Initialize the ExBert model
        self.exbert = exBert_model( args,
            config_2=args.config2,
            checkpoint_path=args.checkpoint_path,
            # config_file='src/bert_config.json',
            # checkpoint_path='path.pth',
            finetune=args.finetune_bert
        )
        self.ext_layer = ExtTransformerEncoder(self.exbert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        # if (args.encoder == 'baseline'): # default = bert
        #     bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
        #                              num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
        #     self.bert.model = BertModel(bert_config)
        #     self.ext_layer = Classifier(self.bert.model.config.hidden_size)
        # if(args.max_pos>512):
        #     my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
        #     my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
        #     self.bert.model.embeddings.position_embeddings = my_pos_embeddings
 
        model_state_dict = self.exbert.state_dict()

        def remove_prefix(key):
            if key.startswith("model."):
                return key[len("model."):]
            if key.startswith("bert."):
                return key[len("bert."):]
            if key.startswith("exbert."):
                return key[len("exbert."):]
            if key.startswith("exbert.model."):
                return key[len("exbert.model."):]
            return key

        for checkpoint_key in self.exbert.checkpoint.keys():
            model_key = remove_prefix(checkpoint_key)
            print('model_key', model_key)
            if model_key in model_state_dict:
                model_state_dict[model_key] = self.exbert.checkpoint[checkpoint_key]

        self.exbert.load_state_dict(model_state_dict)
        
        model_state_dict = self.exbert.state_dict()
        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                        

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        print('----->check:src', src)
        print('----->check:segs', segs)
        print('----->check:mask_src', mask_src )
        
        top_vec = self.exbert(src, segs, mask_src)
        for i in range(len(top_vec)):
            print(f'shape{i}',top_vec[i] )
        print('EXTTTTT len top_vec', len(top_vec)) # debug
        print('EXTTTTT[0]len top_vec ', len(top_vec[0]))#.size()) # debug
        
        # print('EXTTTTT[1] len top_vec ', len(top_vec[1]))#.size()) # debug
        print('type of top_vec', type(top_vec)) # debug
        print('type of[0] top_vec', type(top_vec[0])) # debug
        # print('type of[1] top_vec', type(top_vec[1])) # debug
        
        print('len[0] [0] top_vec size ', len(top_vec[0][0])) # debug
        # print('len[1] [0] top_vec size ', len(top_vec[1][0])) # debug
        


        top_vec_max, _ = torch.max(torch.stack(top_vec), dim=0)
        print('max pooling', top_vec_max)
        top_vec_mean =torch.mean(torch.stack(top_vec), dim=0) 
        print('mean pooling', top_vec_mean)


        sents_vec = top_vec_max[torch.arange(top_vec_max.size(0)).unsqueeze(1), clss]


        print('sents_vec', sents_vec)
        print('sents_vec shape', sents_vec.shape)
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


# class AbsSummarizer(nn.Module):
#     def __init__(self, args, device, checkpoint=None, bert_from_extractive=None): 
#         super(AbsSummarizer, self).__init__()
#         self.args = args
#         self.device = device
#         self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
#         if bert_from_extractive is not None:
#             self.bert.model.load_state_dict(
#                 dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)
#         if (args.encoder == 'baseline'):
#             bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
#                                      num_hidden_layers=args.enc_layers, num_attention_heads=8,
#                                      intermediate_size=args.enc_ff_size,
#                                      hidden_dropout_prob=args.enc_dropout,
#                                      attention_probs_dropout_prob=args.enc_dropout)
#             self.bert.model = BertModel(bert_config)
#         if(args.max_pos>512):
#             my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
#             my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
#             my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
#             self.bert.model.embeddings.position_embeddings = my_pos_embeddings
#         self.vocab_size = self.bert.model.config.vocab_size
#         tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#         if (self.args.share_emb):
#             tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#         self.decoder = TransformerDecoder(
#             self.args.dec_layers,
#             self.args.dec_hidden_size, heads=self.args.dec_heads,
#             d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
#         self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
#         self.generator[0].weight = self.decoder.embeddings.weight
#         if checkpoint is not None:
#             self.load_state_dict(checkpoint['model'], strict=True)
#         else:
#             for module in self.decoder.modules():
#                 if isinstance(module, (nn.Linear, nn.Embedding)):
#                     module.weight.data.normal_(mean=0.0, std=0.02)
#                 elif isinstance(module, nn.LayerNorm):
#                     module.bias.data.zero_()
#                     module.weight.data.fill_(1.0)
#                 if isinstance(module, nn.Linear) and module.bias is not None:
#                     module.bias.data.zero_()
#             for p in self.generator.parameters():
#                 if p.dim() > 1:
#                     xavier_uniform_(p)
#                 else:
#                     p.data.zero_()
#             if(args.use_bert_emb):
#                 tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
#                 tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
#                 self.decoder.embeddings = tgt_embeddings
#                 self.generator[0].weight = self.decoder.embeddings.weight
#         self.to(device)
#     def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
#         top_vec = self.bert(src, segs, mask_src)
#         dec_state = self.decoder.init_decoder_state(src, top_vec)
#         decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
#         return decoder_outputs, None
