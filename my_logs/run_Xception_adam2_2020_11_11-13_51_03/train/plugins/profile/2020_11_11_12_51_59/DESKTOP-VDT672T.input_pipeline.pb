	�D��$P�@�D��$P�@!�D��$P�@	���[�;�?���[�;�?!���[�;�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�D��$P�@�A`��""@A2U0*�<�@Y��~j�4$@*	�����O%A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�0����@!�7�^��X@)�0����@1�7�^��X@:Preprocessing2F
Iterator::Modele�`TR�@!��E����?)�3��7x@1�ҵW`(�?:Preprocessing2P
Iterator::Model::Prefetch��|?5^�?!`�˔
�?)��|?5^�?1`�˔
�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap�m4�7��@!8�BUR�X@)�߾�3�?1F�	���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9���[�;�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�A`��""@�A`��""@!�A`��""@      ��!       "      ��!       *      ��!       2	2U0*�<�@2U0*�<�@!2U0*�<�@:      ��!       B      ��!       J	��~j�4$@��~j�4$@!��~j�4$@R      ��!       Z	��~j�4$@��~j�4$@!��~j�4$@JCPU_ONLYY���[�;�?b 