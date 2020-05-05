""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

# recall与rank以epoch为单位交替进行训练
import argparse
import glob
import json
import logging
from torch import nn
import os
import random
import heapq
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import operator
import os
from EntityRecallRank import EntityRecallRank
from DataSetProcessor import DataSetProcessor
# from .DataSetProcessor import DataSetProcessor
from utils import LinkingExample
import math
import multiprocessing
from torch.nn import CrossEntropyLoss
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    FlaubertConfig,
    RobertaConfig,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
    get_linear_schedule_with_warmup,
)
# from transformers import glue_compute_metrics as compute_metrics
# from transformers import glue_convert_examples_to_features as convert_examples_to_features
# from transformers import glue_output_modes as output_modes
# from transformers import glue_processors as processors
from utils import convert_rank_examples_to_features, convert_recall_examples_to_features



try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# 记住需要修改top64

# 以epoch为单位对recall和rank进行训练


def train(args, train_dataset, model, tokenizer, entity_set, mention_set, config):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_recall_batch_size = args.per_gpu_train_recall_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_recall_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_recall_steps) + 1
    else:
        # len(train_dataloader)返回的过一遍整个数据集对应的batch数目，t_total表示训练需要的step
        t_total = len(train_dataloader) // args.gradient_accumulation_recall_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # 添加LayerNorm.bias
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    # 对模型参数设置分组学习策略，每组以字典的形式给出,优化器将recall和rank阶段的参数全部装入
    # 因此无需对两个阶段分别设立优化器
    # todo: 看两个阶段的loss量级是否相同，将loss,两阶段的准确率加入到tensorboard
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 使用学习率预热，训练时先从小的学习率开始训练
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("****** Running recall training ******")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_recall_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_recall_batch_size
        * args.gradient_accumulation_recall_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_recall_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_recall_step = 0
    global_rank_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_recall_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_recall_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_recall_loss, logging_recall_loss = 0.0, 0.0
    tr_rank_loss, logging_rank_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    training_epoch = 1
    for _ in train_iterator:  # 遍历epoch
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):  # 遍历batch

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)  # 将训练数据部署到GPU上面

            # 需要迭代两次进行，分别执行召回和排序
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            # todo:要进行复位
            inputs["recall"] = True
            inputs["rank"] = False
            outputs = model(**inputs)
            # torch.cuda.empty_cache()
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_recall_steps > 1:
                loss = loss / args.gradient_accumulation_recall_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_recall_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_recall_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_recall_step += 1
                if args.local_rank in [-1, 0] and args.logging_recall_steps > 0 and global_recall_step % args.logging_recall_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, recall=True)
                        for key, value in results.items():
                            eval_key = "eval_recall{}".format(key)
                            logs[eval_key] = value

                    recall_loss_scalar = (tr_recall_loss - logging_recall_loss) / args.logging_recall_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["recall_loss"] = recall_loss_scalar
                    logging_recall_loss = tr_recall_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_recall_step)
                    # print(json.dumps({**logs, **{"step": global_step}}))

        preds = None
        labels = None
        mention_ids = None
        for step, batch in enumerate(epoch_iterator):  # 重新遍历batch

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)  # 将训练数据部署到GPU上面

            # 需要迭代两次进行，分别执行召回和排序
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                # todo:要进行复位
                inputs["recall"] = True
                inputs["rank"] = False
                outputs = model(**inputs)
                # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                logits = outputs[1]  # shape(batch_size, entity_num)
                if preds is None and labels is None:
                    preds = logits.detach().cpu().numpy()
                    labels = inputs["labels"].detach().cpu().numpy()
                    mention_ids = batch[4].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
                    mention_ids = np.append(mention_ids, batch[4].detach().cpu().numpy(), axis=0)
        # # preds = preds.detach().cpu().numpy()
        # # preds = preds.tolist()
        #
        index = np.argsort(-preds, axis=1)
        # 截取top64
        index = index[:, :64]
        # # assert logits[0][index[0]] > logits[0][index[1]]
        # # todo:logits[index[0]]>logits[index[1]]
        # # print("index")
        # # print(index)
        # # 判断top64里面是否包含mention实际对应的label，没有则把头部的entity替换
        # # 保证top64里面第一个为gold label
        for id in range(index.shape[0]):
            if labels[id] not in index[id, :]:
                index[id, 0] = labels[id]
            else:
                for num2, s in enumerate(index[id, :]):
                    if s == labels[id]:
                        index[id, 0] = index[id, 0] + index[id, num2]
                        index[id, num2] = index[id, 0] - index[id, num2]
                        index[id, 0] = index[id, 0] - index[id, num2]
                        break
        #     # print("index1")
        #     # print(index)
        #
        # # entity ranking
        examples = []
        candidate_set = index
        candidate_set = candidate_set.tolist()
        mention_ids = mention_ids.tolist()
        # logger.info("candidate_set shape{},{}".format(len(candidate_set), len(candidate_set[0])))
        # logger.info("mention_id length".format(len(mention_ids)))
        for num in range(len(candidate_set)):
            for num1, candidate in enumerate(candidate_set[num]):
                entity_text = entity_set[candidate]
                mention_text = mention_set[mention_ids[num]]
                if num1 == 0:
                    examples.append(LinkingExample(guid='data', text_a=mention_text, text_b=entity_text,
                                                   label='1', mention_id=mention_ids[num]))
                else:
                    examples.append(LinkingExample(guid='data', text_a=mention_text, text_b=entity_text,
                                                   label='0', mention_id=mention_ids[num]))
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_.bin".format(
                "rank",
                "data",
                training_epoch,
            ),
        )
        features = []
        print("total num of examples{}".format(len(examples)))
        recurrent_step = len(examples)//10000
        pool = multiprocessing.Pool(processes=15)  # 创建4个进程
        result = []
        for i in range(recurrent_step):
            result.append(pool.apply_async(convert_rank_examples_to_features, (examples[10000*i:10000*(i+1)], tokenizer,args.max_seq_length,"rank",None,"classification", False,tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],0,True,cached_features_file, 10000*(i+1)
        )))
        pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
        pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        for res in result:
            features = features + res.get()
        if (len(examples)-recurrent_step*10000) > 0:
            features = features + convert_rank_examples_to_features(
                examples[10000*recurrent_step:],
                tokenizer,
                label_list=None,
                task="rank",
                max_length=args.max_seq_length,
                output_mode="classification",
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                cached_features_file=cached_features_file,
            )
        training_epoch += 1


        # if args.local_rank == 0 and not evaluate:
        #     torch.distributed.barrier()
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset,创建tensor时默认是不可求导的，即requires_grad属性为false
        output_mode = "classification"
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # token_type_ids和segment_ids是一个意思
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        # Each sample will be retrieved by indexing tensors along the first dimension
        # tensors that have the same size of the first dimension, 第一个维度也就是样本的数量
        # 定义新的dataset必须要继承dataset类，实现__getitem__和__len__()方法
        RankDataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        train_rank_batch_size = args.per_gpu_train_rank_batch_size * max(1, args.n_gpu)
        train_rank_sampler = SequentialSampler(RankDataset) if args.local_rank == -1 else DistributedSampler(
            train_dataset)
        train_rank_dataloader = DataLoader(RankDataset, sampler=train_rank_sampler,
                                      batch_size=train_rank_batch_size)
        # 计算总共所需的step
        t_total_rank = len(train_rank_dataloader)//args.gradient_accumulation_rank_steps * args.num_train_epochs
        rank_epoch_iterator = tqdm(train_rank_dataloader, desc="Rank Iteration", disable=args.local_rank not in [-1, 0])
        # rank_pooled_output = torch.empty(0, requires_grad=True)
        for rank_step, rank_batch in enumerate(rank_epoch_iterator):
            model.train()
            rank_batch = tuple(t.to(args.device) for t in rank_batch)
            rank_inputs = {"input_ids": rank_batch[0], "attention_mask": rank_batch[1], "labels": rank_batch[3]}
            rank_inputs["token_type_ids"] = rank_batch[2]
            rank_inputs["rank"] = True
            rank_inputs["recall"] = False
            rank_outputs = model(**rank_inputs)
            rank_loss = rank_outputs[0]
            if args.n_gpu > 1:
                rank_loss = rank_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_rank_steps > 1:
                rank_loss = rank_loss / args.gradient_accumulation_rank_steps

            if args.fp16:
                with amp.scale_loss(rank_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                rank_loss.backward()
            tr_rank_loss += rank_loss.item()
            if (rank_step + 1) % args.gradient_accumulation_rank_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_rank_step += 1

                if args.local_rank in [-1, 0] and args.logging_rank_steps > 0 and global_rank_step % args.logging_rank_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, recall=False)
                        for key, value in results.items():
                            eval_key = "eval_rank{}".format(key)
                            logs[eval_key] = value

                    rank_loss_scalar = (tr_rank_loss - logging_rank_loss) / args.logging_rank_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["rank_loss"] = rank_loss_scalar
                    logging_rank_loss = tr_rank_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_rank_step)
                    # print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_recall_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_recall_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # #如果我们有一个分布式模型，只保存封装的模型
                    # #它包装在PyTorch DistributedDataParallel或DataParallel中
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
        #             for key, value in logs.items():
        #                 with open(os.path.join(output_dir, "results.txt")) as f:
        #                     f.write(str(key, value))
        #
        #     if args.max_steps > 0 and global_step > args.max_steps:
        #         epoch_iterator.close()
        #         break
        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_rank_step, tr_rank_loss / global_rank_step

import csv


# todo: 把evaluate的参数进行调整
# recall 为True，验证recall的准确率
def evaluate(args, model, tokenizer, recall, prefix=" "):
    # 建立从mention_id到label_id的集合
    lable_list = []
    # with open('/home/puzhao_xie/entity-linking-task/share-bert/models/dev.tsv', encoding='utf-8-sig') as f:
    # todo:把dev文件的读取放到load_and_cache函数里面
    with open(args.data_dir+'/dev.tsv', encoding='utf-8-sig') as f:
        for line in list(csv.reader(f, delimiter="\t", quotechar=None)):
            lable_list.append(line[0])
    # rightnum = 0
    total_num = 0
    recall_num = 0
    right_rank_num = 0
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_output_dir = args.output_dir

    results = {}
    eval_dataset, entity_set, mention_set = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_recall_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_recall_loss = 0.0
    nb_eval_recall_steps = 0
    eval_rank_loss = 0.0
    nb_eval_rank_steps = 0
    # preds = None
    # out_label_ids = None
    preds_recall = None
    labels = None
    mention_ids = None
    preds_rank = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = batch[2]
            inputs["recall"] = True
            inputs["rank"] = False
            outputs = model(**inputs)
            # torch.cuda.empty_cache()
            tmp_eval_loss, logits = outputs[:2]
            # print("tmp_eval_loss")
            # print(tmp_eval_loss)
            if args.n_gpu > 0:
                tmp_eval_loss = tmp_eval_loss.mean()
            eval_recall_loss += tmp_eval_loss.item()
            if preds_recall is None and labels is None:
                preds_recall = logits.detach().cpu().numpy()
                labels = inputs["labels"].detach().cpu().numpy()
                mention_ids = batch[4].detach().cpu().numpy()
            else:
                preds_recall = np.append(preds_recall, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
                mention_ids = np.append(mention_ids, batch[4].detach().cpu().numpy(), axis=0)

        nb_eval_recall_steps += 1
    # _, index = torch.sort(preds_recall, dim=1, descending=True)
    index = np.argsort(-preds_recall, axis=1)
    index = index[:, :64]
    # print("未修改的indextop10 mention")
    # print(index[:10, :])
    # print("mention_id")
    # print(mention_ids[:10])
    # print("label")
    # print(labels[:10])
    if recall is True:
        for id in range(index.shape[0]):
            if labels[id] not in index[id, :]:
                total_num += 1
            else:
                recall_num += 1
                total_num += 1
        # a = recall_num/total_num
    # todo:做evaluation的时候是否可以对candidate set 做替换
    else:
        for id in range(index.shape[0]):
            if labels[id] not in index[id, :]:
                total_num += 1
                index[id, 0] = labels[id]
            else:
                recall_num += 1
                total_num += 1
                for num2, s in enumerate(index[id, :]):
                    if s == labels[id]:
                        index[id, 0] = index[id, 0] + index[id, num2]
                        index[id, num2] = index[id, 0] - index[id, num2]
                        index[id, 0] = index[id, 0] - index[id, num2]
                        break
        # print("修改后的index top10 对应的mention")
        # print(index[:10, :])

        examples = []
        candidate_set = index
        candidate_set = candidate_set.tolist()
        mention_ids = mention_ids.tolist()
        for num in range(len(candidate_set)):
            for num1, candidate in enumerate(candidate_set[num]):
                entity_text = entity_set[candidate]
                mention_text = mention_set[mention_ids[num]]
                if num1 == 0:
                    # print("mention_text")
                    # print(mention_text)
                    examples.append(LinkingExample(guid='data', text_a=mention_text, text_b=entity_text,
                                                   label='1', mention_id=mention_ids[num]))
                else:
                    examples.append(LinkingExample(guid='data', text_a=mention_text, text_b=entity_text,
                                                   label='0', mention_id=mention_ids[num]))
        # print(examples[0].text_a)
        features = []
        print("total num of examples{}".format(len(examples)))
        recurrent_step = len(examples) // 10000
        pool = multiprocessing.Pool(processes=10)  # 创建4个进程
        result = []
        for i in range(recurrent_step):
            # print("hhh")
            # print(recurrent_step)
            # print(len(examples[10000 * i:10000 * (i + 1)]))
            result.append(pool.apply_async(convert_rank_examples_to_features, (
            examples[10000 * i:10000 * (i + 1)], tokenizer, args.max_seq_length, "rank", None,
            "classification", False, tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], 0, True,
            None, 10000 * (i + 1)
            )))
        pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
        pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
        for res in result:
            print("len features{}".format(len(res.get())))
            features = features + res.get()
        # print("features num{}".format(len(features)))
        if (len(examples) - recurrent_step * 10000) > 0:
            features = features + convert_rank_examples_to_features(
                examples[10000 * recurrent_step:],
                tokenizer,
                label_list=None,
                task="rank",
                max_length=args.max_seq_length,
                output_mode="classification",
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                cached_features_file=None,
            )
        # features = convert_rank_examples_to_features(
        #     examples,
        #     tokenizer,
        #     label_list=None,
        #     task="rank",
        #     max_length=args.max_seq_length,
        #     output_mode="classification",
        #     pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        #     pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        #     pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        # )
        print("features num{}".format(len(features)))
        output_mode = "classification"
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # token_type_ids和segment_ids是一个意思
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        RankDataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        eval_rank_batch_size = args.per_gpu_eval_rank_batch_size * max(1, args.n_gpu)
        eval_rank_sampler = SequentialSampler(RankDataset)
        eval_rank_dataloader = DataLoader(RankDataset, sampler=eval_rank_sampler,
                                           batch_size=eval_rank_batch_size)
        rank_epoch_iterator = tqdm(eval_rank_dataloader, desc="Rank eval Iteration",)
        for rank_step, rank_batch in enumerate(rank_epoch_iterator):
            model.eval()
            rank_batch = tuple(t.to(args.device) for t in rank_batch)
            with torch.no_grad():
                rank_inputs = {"input_ids": rank_batch[0], "attention_mask": rank_batch[1], "labels": rank_batch[3]}
                rank_inputs["token_type_ids"] = rank_batch[2]
                rank_inputs["rank"] = True
                rank_inputs["recall"] = False
                rank_outputs = model(**rank_inputs)
                rank_eval_loss = rank_outputs[0]
                logits = rank_outputs[1]
                if args.n_gpu > 0:
                    rank_eval_loss = rank_eval_loss.mean()
                eval_rank_loss += rank_eval_loss.item()
            nb_eval_rank_steps += 1

        # nb_eval_steps += 1
            if preds_rank is None:
                preds_rank = logits.detach().cpu().numpy()
                # out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds_rank = np.append(preds_rank, logits.detach().cpu().numpy(), axis=0)
                # out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        # nb_eval_steps += 1
        # print("preds_list")
        # print(preds_rank.shape)
        preds_list = preds_rank[:, 1].tolist()
        label_set = []
        for num in mention_ids:
            label_set.append(lable_list[num])
        # print("preds_list")
        # print(len(preds_list))
        # print("length of candidate set")
        # print(len(candidate_set))
        # print(label_set)
        # print(candidate_set)
        for i in range(len(candidate_set)):
            # 截取64个候选实体
            pred_each = preds_list[i*64:(i+1)*64]
            # max_num_index_list = map(pred_each.index, heapq.nlargest(1, pred_each))
            max_index = pred_each.index(max(pred_each))
            if max_index == 0:
                right_rank_num = right_rank_num + 1
            # if label_set[i] == candidate_set[i][max_index]:
            #     right_rank_num = right_rank_num + 1
    # total_num = total_num + len(label_set)
    # print('*****************************************')
    #     print('rightnum'+str(right_rank_num))
    # print("totalnum"+str(total_num))
    # print('*****************************************')
    # print('rightnum'+str(rightnum))
    # print('totalnum'+str(len(lable_list)))
    # print(preds_list)
    # print(preds)
    # print(out_label_ids)
    # logger.info('length={}'.format(str(len(preds_list))))
    # eval_loss = eval_loss / nb_eval_steps
    # if args.output_mode == "classification":
    #     preds = np.argmax(preds, axis=1)  # 返回每一行最大值对应的索引
    # elif args.output_mode == "regression":
    #     preds = np.squeeze(preds)
    # result = compute_metrics("mrpc", preds, out_label_ids)
    if not recall:
        results["recall_accuracy"] = recall_num/total_num
        results["rank_accuracy"] = right_rank_num/total_num
        results["rank_loss"] = eval_rank_loss/nb_eval_rank_steps
        results["recall_loss"] = eval_recall_loss/nb_eval_recall_steps
    if recall:
        results["recall_accuracy"] = recall_num / total_num
        results["recall_loss"] = eval_recall_loss / nb_eval_recall_steps
    # output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results {} *****".format(prefix))
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))

    return results

# 读取training set 以及


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = DataSetProcessor()
    output_mode = "classification"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            "linking",
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        # if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        #     # HACK(label indices are swapped in RoBERTa pretrained model)
        #     label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (  # 返回的是example列表
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        # 如果seq_length大于max_seq_length,在处理一对句子时，先从较长的句子开始截取,直到
        # 句子对的总长小于max_seq_length
        features = convert_recall_examples_to_features(
            examples,
            tokenizer,
            task="recall",
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset,创建tensor时默认是不可求导的，即requires_grad属性为false
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    # token_type_ids和segment_ids是一个意思
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    all_mention_id = torch.tensor([f.mention_id for f in features], dtype=torch.long)

    # Each sample will be retrieved by indexing tensors along the first dimension
    # tensors that have the same size of the first dimension, 第一个维度也就是样本的数量
    # 定义新的dataset必须要继承dataset类，实现__getitem__和__len__()方法
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_mention_id)
    entityset = []
    documents = {}
    # with open('/home/puzhao_xie/entity-linking-task/zero-shot-dataset/Wikia/zeshel/documents/military.json', 'r') as f:
    with open(args.data_dir+'/military.json', 'r') as f:
        while True:
            line = f.readline()
            if line:
                entity_dict = json.loads(line)
                entity_dict = dict(entity_dict)
                entityset.append(entity_dict['text'])
                documents[entity_dict['document_id']] = entity_dict
            else:
                break
    mentions = []
    if evaluate:
        # with open('/home/puzhao_xie/entity-linking-task/zero-shot-dataset/Wikia/zeshel/mentions/heldout_train_unseen.json') as f:
        with open(args.data_dir+'/heldout_train_unseen.json') as f:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    mention_dict = json.loads(line)
                    if operator.eq(mention_dict['corpus'], 'military'):  # 提取military这个domain对应的mention
                        mentions.append(mention_dict)
                else:
                    break
    if not evaluate:
        # with open('/home/puzhao_xie/entity-linking-task/zero-shot-dataset/Wikia/zeshel/mentions/train.json') as f:
        with open(args.data_dir+'/train.json') as f:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    mention_dict = json.loads(line)
                    if operator.eq(mention_dict['corpus'], 'military'):  # 提取military这个domain对应的mention
                        mentions.append(mention_dict)
                else:
                    break

    mentionset = []
    for i, mention in enumerate(mentions):
        mention_context = create_context_from_document(mention, documents, 32)
        mentionset.append(mention_context)

    return dataset, entityset, mentionset


def load_entity_embedding(args):
    entity_embed = torch.empty(0)
    if os.path.exists(args.entity_dir):
        with open(args.entity_dir, 'rb') as f:
            entity_embed = np.load(f)
            # 设置entity embedding的属性为可导
            entity_embed = torch.from_numpy(entity_embed)
    # logger.info('finished reading entity embedding of shape{}'.format(entity_embed.shape))
    return entity_embed


def create_context_from_document(mention, documents, max_seq_length):
    context_document_id = mention['context_document_id']
    start_index = mention['start_index']
    end_index = mention['end_index']
    context_document = documents[context_document_id]['text']
    context_tokens = context_document.split()
    mention_context = get_context_tokens(context_tokens, start_index, end_index, max_seq_length)
    mention_context = ' '.join(mention_context)
    result = mention_context
    return result


def get_context_tokens(context_tokens, start_index, end_index, max_seq_length):
    start_pos = start_index - max_seq_length
    if start_pos < 0:
        start_pos = 0
    prefix = context_tokens[start_pos: start_index]
    suffix = context_tokens[end_index + 1: end_index + max_seq_length + 1]
    mention = context_tokens[start_index: end_index + 1]
    remaining_tokens = max_seq_length - len(mention)
    half_remaining_tokens = int(math.ceil(1.0*remaining_tokens/2))
    # 首先确定prefix的长度
    mention_context = []
    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
    elif len(prefix) < half_remaining_tokens:
        prefix_len = len(prefix)

    if prefix_len > len(prefix):  # 这里针对的是第二个elif条件，即(remaining_tokens - len(suffix))>len(prefix)
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    # mention_start = len(prefix)
    # mention_end = mention_start + len(mention) - 1
    mention_context = mention_context[:max_seq_length]  # 将mention超出的部分截取掉
    # print(len(mention_context))
    # print(mention)

    return mention_context


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--entity_dir",
        default=None,
        type=str,
        required=True,
        help="the input entity numpy array dir",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    # parser.add_argument(
    #     "--task_name",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="The name of the task to train selected in the list: ",
    # )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    # todo:在train的时候进行evaluate
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_recall_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training recall stage.",
    )
    parser.add_argument(
        "--per_gpu_train_rank_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training rankstage.",
    )
    parser.add_argument(
        "--per_gpu_eval_recall_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_gpu_eval_rank_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_rank_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a rank backward/update pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_recall_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a recall backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(  # 如果设置了num_train_epochs，就无需再设置max_steps
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--entity_num",
        default=-1,
        type=int,
        required=True,
        help="number of entities in a domain",
    )
    parser.add_argument("--num_share_layers", type=int, default=12, help="number of shared layers between rank and recall bert")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_rank_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--logging_recall_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    # args.no_cuda = False
    if args.local_rank == -1 or args.no_cuda:  # local_rank的默认值为-1
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    processor = DataSetProcessor()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:  # 同步所有进程
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, _, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        num_entity=args.entity_num,
        finetuning_task='mrpc',  # 本文选择mrpc: MrpcProcessor
        num_hidden_layers=args.num_share_layers,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # model_class = EntityRecallRank(config)
    model_class = EntityRecallRank
    model = model_class.from_pretrained(
        args.model_name_or_path,  # 从model_name_or_path加载模型
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # todo:根据args.do_train来判断
    if args.do_train:
        entity_embedding = load_entity_embedding(args)
        model.recall_classifier.weight = nn.Parameter(entity_embedding)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)  # 将模型布置到GPU上

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, entity_set, mention_set = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, entity_set, mention_set, config)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, recall=False, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()