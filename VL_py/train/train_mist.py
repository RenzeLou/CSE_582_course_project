import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
import os
import json
import evaluate
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import wandb
from IPython.core.debugger import Pdb
import numpy as np
dbg = Pdb()
from prettytable import PrettyTable

def eval(model, val_loader, a2v, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    results = {}

    yhat = []
    y = []
    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        for i, batch in enumerate(val_loader):
            answer_id, answer, video, question, question_clip = (
                batch["answer_id"],
                batch["answer"],
                (batch["video"][0].cuda(), batch["video"][1].cuda()),
                batch["question"].cuda(),
                batch['question_clip'].cuda()
            )
            video_len = batch["video_len"]
            question_mask = (question > 0).float()
            video_mask = get_mask(video_len, video[1].size(1)).cuda()
            count += answer_id.size(0)

            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    # video_mask=video_mask,
                    question_clip=question_clip
                )
                topk = torch.topk(predicts, dim=1, k=1).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id

                predicted = torch.max(predicts, dim=1).indices.cpu()
                # print('all', predicted, answer_id)
                # print('label0', predicted[~answer_id], answer_id[~answer_id])
                # print('label1', predicted[answer_id], answer_id[answer_id])
                # input()
                yhat.extend(predicted.tolist())
                y.extend(answer_id.tolist())
                # print(yhat, y)

                for bs, qid in enumerate(batch['question_id']):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}

            # else:
            #     fusion_proj, answer_proj = model(
            #         video,
            #         question,
            #         text_mask=question_mask,
            #         # video_mask=video_mask,
            #         answer=answer.cuda(),
            #         question_clip=question_clip
            #     )
            #     fusion_proj = fusion_proj.unsqueeze(2)
            #     predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
            #     predicted = torch.max(predicts, dim=1).indices.cpu()
            #     metrics["acc"] += (predicted == answer_id).sum().item()
               
            #     for bs, qid in enumerate(batch['question_id']):
            #         results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}


    step = "val" if not test else "test"
    # print(acc_metric)

    f1 = np.append(f1_score(y_pred=yhat,y_true=y,zero_division=1, average='macro'), f1_score(y_pred=yhat,y_true=y,zero_division=1, average=None))
    precision =  np.append(precision_score(y_pred=yhat,y_true=y,zero_division=1, average='macro'), precision_score(y_pred=yhat,y_true=y,zero_division=1, average=None))
    recall =  np.append(recall_score(y_pred=yhat,y_true=y,zero_division=1, average='macro'), recall_score(y_pred=yhat,y_true=y,zero_division=1, average=None))
    
    metrics['overall_accuracy'] = accuracy_score(y_pred=yhat, y_true=y)
    metrics['overall_f1'] = f1[0]
    metrics['label_0_f1'] = f1[1]
    metrics['label_1_f1'] = f1[2]
    metrics['overall_recall'] = recall[0]
    metrics['label_0_recall'] = recall[1]
    metrics['label_1_recall'] = recall[2]
    metrics['overall_precision'] = precision[0]
    metrics['label_0_precision'] = precision[1]
    metrics['label_1_precision'] = precision[2]
    
    #metrics table
    t = PrettyTable(['label', 'accuracy', 'recall', 'precision', 'f1'])
    t.add_row(['overall', metrics['overall_accuracy'], metrics['overall_recall'], metrics['overall_precision'], metrics['overall_f1']])
    t.add_row(['label 0', '-', metrics['label_0_recall'], metrics['label_0_precision'], metrics['label_0_f1']])
    t.add_row(['label 1', '-', metrics['label_1_recall'], metrics['label_1_precision'], metrics['label_1_f1']])
    t.float_format = ".2f"
    logging.info(t)
    acc = metrics['overall_accuracy']
    
    json.dump(results, open(os.path.join(args.save_dir, f"val-{acc:.5%}.json"), "w"))
    

    return metrics


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, val_loader=None, best_val_acc=None, best_epoch=None):
    
    wandb.init(name=args.name)
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    
    for i, batch in enumerate(train_loader):
        # print(batch)
        # input()
        answer_id, answer, video, question, question_clip = (
            batch["answer_id"],
            batch["answer"],
            (batch["video"][0].cuda(), batch["video"][1].cuda()),
            batch["question"].cuda(),
            batch['question_clip'].cuda()
        )
        # print(answer)
        video_len = batch["video_len"]
        question_mask = (question > 0).float()
        # video_mask = (
        #     get_mask(video_len, video[1].size(1)).cuda() if args.max_feats > 0 else None
        # )
        N = answer_id.size(0)
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video,
                question,
                text_mask=question_mask,
                # video_mask=video_mask,
                question_clip=question_clip
            )
        else:
            fusion_proj, answer_proj = model(
                video,
                question,
                text_mask=question_mask,
                # video_mask=video_mask,
                answer=answer.cuda(),
                question_clip=question_clip
            )
            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        if args.dataset == "ivqa":
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            vqa_loss = criterion(predicts, answer_id.cuda())
            predicted = torch.max(predicts, dim=1).indices.cpu()
            running_acc.update((predicted == answer_id).sum().item() / N, N)

        if args.mlm_prob:
            inputs = batch["question"]
            inputs, labels = mask_tokens(
                inputs, model.module.bert.bert_tokenizer, mlm_probability=0.15
            )
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                mode="mlm",
            )
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        else:
            loss = vqa_loss

        if torch.isnan(loss):
            print(batch['question_id'], batch['video_id'], loss)
            dbg.set_trace()
        # dbg.set_trace()
        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)
        
        if ((i + 1) % (len(train_loader) // args.freq_display ) == 0):
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, Training MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}"
                )
            wandb.log({"training_acc": running_acc.avg, 'loss':running_vqa_loss.avg} )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()
            
            
        if val_loader is not None and ((i + 1) % (len(train_loader) // args.freq_display ) == 0):
            metrics = eval(model, val_loader, a2v, args, test=False)
            wandb.log(metrics)
            val_acc = metrics['accuracy'] 
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
            # else:
            #     torch.save(
            #         model.state_dict(), os.path.join(args.save_dir, f"model-{epoch}.pth")
            #     )

    return best_val_acc, best_epoch
