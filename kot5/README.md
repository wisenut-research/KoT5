# KoT5

##Setup
```
conda install cudnn==8.2.1
pip install -r requirements.txt
```


##Usage



###Train example
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

###Evaluate example
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

###Predict example
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