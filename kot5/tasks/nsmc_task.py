import functools
import t5.data
from t5.data import postprocessors as t5_postprocessors
from t5.evaluation import metrics as t5_metrics
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary
import tensorflow as tf
import t5
from t5.data.utils import TextLineTask
from t5.data import preprocessors
from t5.data.utils import Feature

TaskRegistry = t5.data.TaskRegistry

file_list = ['resources/sample_data/ratings_train.txt']
file_list2 = ['resources/sample_data/ratings_test.txt']

corpus_path = {
    "train": file_list,
    "validation" : file_list2
}


#dataset loader
def task_dataset_fn(split, shuffle_files=False):
  del shuffle_files

  ds = tf.data.TextLineDataset(corpus_path[split])
  print(ds)

  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", "", ""], #columns 수
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = ds.map(lambda *ex: dict(zip(["id", "document", "label"], ex)))
  return ds

# append prefix
def task_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""

        return {
            "inputs": tf.strings.join(["nsmc 문장1 : ", normalize_text(ex["document"])
                                       ]),
            "targets": ex["label"]
        }

    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


##vocab path
vocab_model_path = 'resources/vocab/sentencepiece.model'
vocab = SentencePieceVocabulary(vocab_model_path, extra_ids=100)
print("Vocab has a size of %d\n" %vocab.vocab_size)

t5.data.TaskRegistry.remove("nsmc")


t5.data.TaskRegistry.add(
    "nsmc",
    dataset_fn = task_dataset_fn,
    splits=["train", "validation"],
    text_preprocessor=task_preprocessor,
    output_features=t5.data.Feature(vocabulary=vocab, add_eos=True),
    metric_fns=[t5.evaluation.metrics.accuracy] #metric
)

