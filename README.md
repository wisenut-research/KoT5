# KoT5
한국어 T5 모델

##Introdution
[text-to-text-transfer-transformer](https://arxiv.org/abs/1910.10683) \
![](./imgs/t5_dataflow.png)


##Model Version
```
t5==0.6.4
mesh-tensorflow==0.1.17
```

##Pre-trained Checkpoints
| model | layers | hidden_size | parameter | tensorflow | pytorch
| --- | --- | --- | --- | --- | --- |
| T5-Small | 6 | 512 | 60M | [link](https://storage.googleapis.com/nlp_bucket-1/KoT5_models/small.zip) | [link](https://storage.googleapis.com/nlp_bucket-1/KoT5_models/small_hf.zip)
| T5-Base | 12 | 768 | 220M | [link](https://storage.googleapis.com/nlp_bucket-1/KoT5_models/base.zip) | [link](https://storage.googleapis.com/nlp_bucket-1/KoT5_models/base_hf.zip)

##Usage
[Tensorflow](./kot5/README.md) \
[Pytorch](./kot5_hf/README.md)

##Experiment
|                       | **NSMC**<br/>(acc) | **KorSTS**<br/>(spearman) | **Summarization**<br/>(rouge f1) |
| :-------------------- | :----------------: | :--------------------: | :----------------: | 
| KoGPT[[1]](https://github.com/kakaobrain/kogpt)                |       89.59        |         87.92          |       -        | 
| KoGPT2-base[[2]](https://github.com/SKT-AI/KoGPT2)      |       89.03        |         86.65          |      -        |
| KoBART-base[[3]](https://github.com/SKT-AI/KoBART)               |       90.06        |         87.70          |       51.5, 35.1, 41.5        |
| **KoT5-small**       |       **88.50**        |         **-**          |       **50.79, 34.25, 42.40**        |
| **KoT5-base**    |       **90.66**        |         **-**         |       **52.50, 35.87, 43.47**      | 





##Citation
모델을 연구용으로 사용하는 경우 아래와 같이 인용해주세요.
```
@misc{kakaobrain2021kogpt,
  title         = {KoT5: Wisenut Research Korean Text-To-Text Transfer Transformer},
  author        = {Bongsu Kim and Saebyeok Lee},
  year          = {2021},
  howpublished  = {\url{https://github.com/wisenut-research/KoT5}},
}
```

##License
**KoT5**는 [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 라이선스 하에 공개되어 있습니다. \
모델을 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 [LICENSE](./LICENSE) 파일에서 확인하실 수 있습니다.


[paper]: https://arxiv.org/abs/1910.10683
[released_checkpoints]: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md
[beam]: https://beam.apache.org
[c4]: https://www.tensorflow.org/datasets/catalog/c4
[cc]: https://commoncrawl.org
[dataflow]: https://cloud.google.com/dataflow/
[gcs]: https://www.tensorflow.org/datasets/gcs
[gcd]: https://cloud.google.com/dataflow/
[gin]: https://github.com/google/gin-config
[mtft]: https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow/transformer
[tfds]: https://www.tensorflow.org/datasets
[tfds_beam]: https://www.tensorflow.org/datasets/beam_datasets
