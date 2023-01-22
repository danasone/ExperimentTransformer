# ExperimentTransformer

This repository contains the implementation and tests of [Linear attention (self-attention depends on N instead of N^2 time but can't work in zero-shot)](https://arxiv.org/pdf/2006.16236.pdf) and [8bit optimization](https://arxiv.org/pdf/2010.13382.pdf) (32 bit to 8 bit ~ memory boost by 4 times with capability of zero-shot inference and using pretrained weights). All method give same results.

Available config changes:
```python -m train --config configs/<>```
- attention "full" (vanilla) / "linear"
- precision 32/8 with appropriate Adam8bit
- task of GLUE "cola" which is classofocation whether it is a grammatical English sentence with accuracy metric or "sst2" (sentiment classification) with MCC metric
Some experiments are shown in [lab.ipynb](https://github.com/danasone/ExperimentTransformer/lab.ipynb)

Speed test on default weights transformer
| Method | Time (s) |
| --- | --- |
| vanilla | 0.29 |
| linear | 0.26 |
| 8bit | 0.19 |
| linear + 8bit | 0.15 |

Also you can finetune or zero-shot inference pretrain deberta in 8 bit by adding argument ```--deberta-path <path to folder>``` in [transformers](https://github.com/huggingface/transformers) format
