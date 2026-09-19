---
license: apache-2.0
base_model: mo-thecreator/wav2vec2-base-finetuned
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: wav2vec2-base-finetuned-finetuned
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wav2vec2-base-finetuned-finetuned

This model is a fine-tuned version of [mo-thecreator/wav2vec2-base-finetuned](https://huggingface.co/mo-thecreator/wav2vec2-base-finetuned) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0829
- Accuracy: 0.9882

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Accuracy | Validation Loss |
|:-------------:|:-----:|:----:|:--------:|:---------------:|
| 0.1448        | 1.0   | 1900 | 0.9601   | 0.1447          |
| 0.0673        | 2.0   | 3800 | 0.9824   | 0.0817          |
| 0.0178        | 3.0   | 5700 | 0.9796   | 0.1054          |
| 0.0002        | 4.0   | 7600 | 0.9824   | 0.1074          |
| 0.0108        | 5.0   | 9500 | 0.9882  |  0.0829          |


### Framework versions

- Transformers 4.39.3
- Pytorch 2.1.2
- Datasets 2.18.0
- Tokenizers 0.15.2

### Contributors

- Abdalla312
- mo-thecreator