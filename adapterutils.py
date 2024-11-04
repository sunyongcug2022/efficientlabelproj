import os
import random
import yaml
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

# from adapter.dpmf.clip import clip
import clip

from datautils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = "cpu"

# DualAdapter/extract_few_shot_features.py
# 从few shot数据集中抽取tain特征, cache_keys和cache_value信息
def extract_few_shot_features(
        cfg,
        clip_model,
        train_loader_cache):  # train_loader_cache is few shot trainloader
    cache_keys = []
    cache_values = []
    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg['augment_epoch']):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx,
                                                    cfg['augment_epoch']))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.to(device)
                image_features = clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    target = target.to(device)
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(
                0))  # cache_keys are the few-shot train feautres

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0).to(
        torch.int64)).half()  # cache_values are the few-shot train labels
    torch.save(cache_keys,
               cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    torch.save(cache_values,
               cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return train_features, cache_keys, cache_values


# 采用CLIP模型从loader中抽取视觉特征信息
def extract_features_from_loader(cfg, split, clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.to(device), target.to(device)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
    torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels


# 根据Positive语言提示器抽取分类特征，或称之为语言文本表征
def extract_text_features(cfg, classnames, clip_model, template):
    # text embedding for dual texts
    with torch.no_grad():
        clip_weights = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            if cfg['rsdataname'] != 'siri-wuhu':
                classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            texts_token = clip.tokenize(template_texts,
                                        truncate=True).to(device)
            # prompt ensemble for Eurosat
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    torch.save(clip_weights, cfg['cache_dir'] + "/text_weights_template.pt")

    return clip_weights


# 根据cupl方法抽取positive prompts的语言文本表征text embeddings
def extract_cupl_text_features(cfg, classnames, prompt_path, clip_model,
                               template):
    f = open(prompt_path)
    prompts = json.load(f)

    # text embedding for cupl texts
    with torch.no_grad():
        cupl_clip_weights = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            if cfg['rsdataname'] != 'siri-wuhu':
                classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = cupl_texts + template_texts

            texts_token = clip.tokenize(texts, truncate=True).to(device)
            # prompt ensemble for Eurosat
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            cupl_clip_weights.append(class_embedding)

        cupl_clip_weights = torch.stack(cupl_clip_weights, dim=1).to(device)
    torch.save(cupl_clip_weights, cfg['cache_dir'] + "/text_weights_cupl.pt")

    return cupl_clip_weights


# 根据cupl方法抽取negative prompts的语言文本表征text embeddings
def extract_text_features_negative(cfg, classnames, prompt_path, clip_model,
                                   template):
    f = open(prompt_path)
    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        clip_weights_all = []
        for i, classname in enumerate(classnames):
            # Tokenize the prompts
            if cfg['rsdataname'] != 'siri-wuhu':
                classname = classname.replace('_', ' ')
            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts

            texts_token = clip.tokenize(texts, truncate=True).to(device)
            # prompt ensemble for eurosat
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    torch.save(clip_weights,
               cfg['cache_dir'] + "/text_weights_negative_template.pt")
    print(clip_weights.shape)
    return clip_weights

# utils.py
def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_template.pt"
    clip_weights_template = torch.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_cupl.pt"
    clip_weights_cupl = torch.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_negative_template.pt"
    clip_weights_negative = torch.load(save_path)
    return clip_weights_template, clip_weights_cupl, clip_weights_negative

def load_few_shot_feature(cfg):
    cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return cache_keys, cache_values # cache_keys are features, cache_values are labels


def load_val_test_feature(cfg, split):
    features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
    labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def get_clip_model_feat_dims(model_backbone):
    # feature dimensions for each model
    feat_dims = {
        'RN50': 1024,
        'ViT-B/16': 512,
        'RN50x16': 768,
        'RN101': 512,
        'ViT-L/14': 768,
        'ViT-B/32': 512
    }
    return feat_dims[model_backbone]


def search_hp_ensemble(cfg,
                       cache_keys_resnet,
                       cache_values_resnet,
                       val_features_resnet,
                       cache_keys_vit,
                       cache_values_vit,
                       val_features_vit,
                       val_labels,
                       clip_weights_resnet, 
                       clip_weights_vit,
                       adapter=None):
    if cfg['search_hp'] == True:
        beta_list = [
            i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1
            for i in range(cfg['search_step'][0])
        ]
        alpha_list = [
            i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1
            for i in range(cfg['search_step'][1])
        ]
        gamma_list = [
            i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1
            for i in range(cfg['search_step'][1])
        ]

        best_acc = 0
        best_beta, best_alpha, best_gamma = 0, 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                for gamma in gamma_list:
                    if adapter:
                        affinity = adapter(features)
                    else:
                        # Affinity and cache logits for ResNet
                        affinity_resnet = val_features_resnet @ cache_keys_resnet
                        # Affinity and cache logits for ViT
                        affinity_vit = val_features_vit @ cache_keys_vit

                    cache_logits_resnet = ((-1) *
                                           (beta - beta * affinity_resnet)
                                           ).exp() @ cache_values_resnet
                    cache_logits_vit = (
                        (-1) *
                        (beta - beta * affinity_vit)).exp() @ cache_values_vit

                    # Combine ResNet and ViT cache logits
                    cache_logits = gamma * cache_logits_resnet + cache_logits_vit  # gamma
                    # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                    vit_clip_logits = 100. * val_features_vit @ clip_weights_vit
                    tip_logits = vit_clip_logits + cache_logits * alpha
                    acc = cls_acc(tip_logits, val_labels)

                    if acc > best_acc:
                        print(
                            "New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}"
                            .format(beta, alpha, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                        best_gamma = gamma

        print(
            "\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha, best_gamma
