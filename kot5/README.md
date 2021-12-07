# KoT5(Tensorflow)
반드시 사전학습에 사용된 모델 버전(t5==0.64, mesh_tensorflow==0.1.17)를 설치해야함

## Setup

```
conda install cudnn==8.2.1
pip install -r requirements.txt
```


## Usage

### Options

`--module_import`: task파일(데이터로드, 전처리, 매트릭 함수 등 포함) 경로 \
`--model_dir`: tensorflow 버전 t5 모델 경로 \
`--gin_param="MIXTURE_NAME`: task 파일에서 등록한 task 명 \
`--gin_param="utils.run.mesh_shape`: GPU 사용하는 경우 배치 수 \
`--gin_param="utils.run.mesh_devices` : 등록할 GPU Device \
`--gin_param="utils.run.batch_size`: 배치당 토큰 수 \
`--gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica`: 마이크로 배치당 토큰 수 \
`--gin_file` : config파일(gin 파일)경로


### Train example

```
python -m t5.models.mesh_transformer_main \
--module_import="kot5.tasks.nsmc_task" \
--model_dir="resources\sample_model" \
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
--gin_param="utils.run.mesh_devices = ['gpu:0']" \
--gin_param="MIXTURE_NAME = 'nsmc'" \
--gin_param="utils.run.train_steps=733000" \
--gin_file="t5/models/gin/dataset.gin" \
--gin_param="SentencePieceVocabulary.extra_ids=100" \
--gin_file="resources/sample_model/operative_config.gin" \
--gin_file="t5/models/gin/sequence_lengths/cnn_dailymail_v002.gin" \
--gin_file="t5/models/gin/learning_rate_schedules/rsqrt_no_ramp_down.gin" \
--gin_file="t5/models/gin/models/t5.1.0.small.gin" \
--gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
--gin_param="utils.run.batch_size=('tokens_per_batch', 2560)" \
--gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 512" \
--gin_param="utils.run.save_checkpoints_steps=100"
```

### Evaluate example

```
python -m t5.models.mesh_transformer_main \
--module_import="kot5.tasks.nsmc_task" \
--model_dir="resources\sample_model" \
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
--gin_param="utils.run.mesh_devices = ['gpu:0']" \
--gin_param="MIXTURE_NAME = 'nsmc'" \
--gin_param="SentencePieceVocabulary.extra_ids=100" \
--gin_file="resources/sample_model/operative_config.gin" \
--gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
--gin_param="utils.run.batch_size=('tokens_per_batch', 2560)" \
--gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 512" \
--gin_param="eval_checkpoint_step = 'all'" \
--gin_file="t5/models/gin/eval.gin" \
--gin_file="t5/models/gin/beam_search.gin" \
--gin_file="t5/models/gin/sequence_lengths/cnn_dailymail_v002.gin" \
--gin_file="t5/models/gin/learning_rate_schedules/rsqrt_no_ramp_down.gin" \
```

### Predict example

```
python -m t5.models.mesh_transformer_main \
--module_import="kot5.tasks.nsmc_task" \
--model_dir="resources\sample_model" \
--gin_file="infer.gin" \
--gin_file="sample_decode.gin" \
--gin_file="gin\models\t5.1.0.small.gin" \
--gin_param="input_filename = 'predict_dir\inputs.txt'" \
--gin_param="output_filename = 'predict_dir\outputs.txt'"
```