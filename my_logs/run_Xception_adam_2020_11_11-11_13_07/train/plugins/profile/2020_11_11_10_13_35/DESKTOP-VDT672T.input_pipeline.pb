	M�O�b�@M�O�b�@!M�O�b�@	�J�M��?�J�M��?!�J�M��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M�O�b�@ �~�:p�?Aŏ1�Y�@Y��K7	 @*	    � A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generatord;�O�@!��*�_�X@)d;�O�@1��*�_�X@:Preprocessing2F
Iterator::Model��+e�?!'K�I�^�?)333333�?1�Z�duY|?:Preprocessing2P
Iterator::Model::PrefetchA��ǘ��?!:w�\j�`?)A��ǘ��?1:w�\j�`?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap6<�R�@!�U	m�X@)/n��r?1r)@�U�J?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�J�M��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 �~�:p�? �~�:p�?! �~�:p�?      ��!       "      ��!       *      ��!       2	ŏ1�Y�@ŏ1�Y�@!ŏ1�Y�@:      ��!       B      ��!       J	��K7	 @��K7	 @!��K7	 @R      ��!       Z	��K7	 @��K7	 @!��K7	 @JCPU_ONLYY�J�M��?b 