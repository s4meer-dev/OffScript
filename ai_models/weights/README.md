---
license: apache-2.0
base_model: motheecreator/Deepfake-audio-detection
tags:
- generated_from_trainer
datasets:
- audiofolder
metrics:
- accuracy
model-index:
- name: Deepfake-audio-detection-V2
  results:
  - task:
      name: Audio Classification
      type: audio-classification
    dataset:
      name: audiofolder
      type: audiofolder
      config: default
      split: train
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9972843305874898
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Deepfake-audio-detection-V2

This model is a fine-tuned version of [motheecreator/Deepfake-audio-detection](https://huggingface.co/motheecreator/Deepfake-audio-detection) on the audiofolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0141
- Accuracy: 0.9973

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.0503        | 1.0   | 1381 | 0.0514          | 0.9858   |
| 0.0327        | 2.0   | 2762 | 0.0174          | 0.9956   |
| 0.0064        | 3.0   | 4143 | 0.0221          | 0.9950   |
| 0.0003        | 4.0   | 5524 | 0.0174          | 0.9965   |
| 0.0115        | 5.0   | 6905 | 0.0141          | 0.9973   |


### Framework versions

- Transformers 4.41.2
- Pytorch 2.1.2
- Datasets 2.19.2
- Tokenizers 0.19.1
