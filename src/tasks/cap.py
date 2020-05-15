# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.cap_model import CAPModel
from tasks.cap_data import CAPDataset, CAPTorchDataset, CAPEvaluator

import torch.nn.functional as F
from torch.distributions import Categorical
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BertForMaskedLM


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = CAPDataset(splits)
    tset = CAPTorchDataset(dset)
    evaluator = CAPEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class CAP:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = CAPModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit,out,input_ids,_ = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                # loss = self.bce_loss(logit, target)
                # loss = loss * logit.size(1)
                loss = self.CrossEntropyLoss(out.view(-1, 30000),input_ids.view(-1))

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, loss.item())

            #if self.valid_tuple is not None:  # Do Validation
                #valid_score = self.evaluate(eval_tuple)
                #if valid_score > best_valid:
                    #best_valid = valid_score
                    #self.save("BEST")

            #    log_str += "Epoch %d: Valid %0.2f\n" % (epoch, loss.item())

            print(log_str, end='')
            #print(sent)
            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        captions = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit,_,_,output_sents = self.model(feats, boxes, sent)
                #print(len(output_sents))
                #print(feats.shape)
                # dist = Categorical(logits=F.log_softmax(logit[0], dim=-1))
                # pred_idxs = dist.sample().tolist()
                # label = BertTokenizer.convert_ids_to_tokens(pred_idxs)
                score, label = logit.max(1)
                #dset.convert_ids_to_tokens
                #for qid, l in zip(ques_id, label.cpu().numpy()):
                    # ans = dset.label2ans[l]
                    #ans = l
                    #quesid2ans[qid.item()] = ans
                for qid, caption in zip(ques_id, output_sents):
                    words = tokenizer.convert_ids_to_tokens(caption)
                    captions[qid.item()]=[" ".join(words)]
                    #print(type(qid))
                    #print(type(caption))
        if dump is not None:
            evaluator.dump_result(captions, dump)
        return captions

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    cap = CAP()

    # Load CAP model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        cap.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            cap.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = cap.evaluate(
                get_data_tuple('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', cap.train_tuple.dataset.splits)
        if cap.valid_tuple is not None:
            print('Splits in Valid data:', cap.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (cap.oracle_score(cap.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        cap.train(cap.train_tuple, cap.valid_tuple)


