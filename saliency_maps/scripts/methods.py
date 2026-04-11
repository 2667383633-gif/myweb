from scripts.iba import IBAInterpreter, Estimator
import numpy as np
import torch

# Feature Map is the output of a certain layer given X

def extract_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True)
        feature = states['hidden_states'][layer_idx + 1]  # +1 because the first output is embedding
        return feature


def extract_text_feature_map(model, layer_idx, x):
    with torch.no_grad():
        states = model(x, output_hidden_states=True)
        feature = states[0]
        return feature


# Extract BERT Layer

def extract_bert_layer(model, layer_idx):
    desired_layer = ''
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == 'layers' or n == 'resblocks':
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        desired_layer = s2
                        return desired_layer


# Get an estimator for the compression term

def get_compression_estimator(var, layer, features):
    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features)
    estimator.S = var * np.ones(features.shape)
    estimator.N = 1
    estimator.layer = layer
    return estimator



def text_heatmap_iba(text_t, image_t, model, layer_idx, beta, var, lr=1, train_steps=10, progbar=True):
    features = extract_text_feature_map(model.text_model, layer_idx, text_t)
    layer = extract_bert_layer(model.text_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(model, compression_estimator, beta=beta, lr=lr, steps=train_steps, progbar=progbar)
    return reader.text_heatmap(text_t, image_t)



def vision_heatmap_iba(
    text_t,
    image_t,
    model,
    layer_idx,
    beta,
    var,
    lr=1,
    train_steps=10,
    ensemble=False,
    progbar=True,
    precomputed_text_feature=False,
):
    """
    Args:
        text_t:
            - 鏃фā寮�: token ids, shape (1, L)
            - 鏂版ā寮�: 棰勮绠楁枃鏈壒寰�, shape (1, D)
        precomputed_text_feature:
            True 鏃�, 涓嬫父鐩存帴鎶� text_t 褰撴垚鏂囨湰鐗瑰緛浣跨敤, 涓嶅啀璋冪敤 model.get_text_features
    """
    features = extract_feature_map(model.vision_model, layer_idx, image_t)
    layer = extract_bert_layer(model.vision_model, layer_idx)
    compression_estimator = get_compression_estimator(var, layer, features)
    reader = IBAInterpreter(
        model,
        compression_estimator,
        beta=beta,
        lr=lr,
        steps=train_steps,
        progbar=progbar,
        ensemble=ensemble,
    )
    return reader.vision_heatmap(
        text_t,
        image_t,
        precomputed_text_feature=precomputed_text_feature,
    )