  *	???????@2?
]Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2b??AL@!\??g?X@)??????@1?M???9G@:Preprocessing2?
sIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::TensorSlice??????!??8r??<@)?????1??8r??<@:Preprocessing2?
fIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle??[????@!+w&_J@)ė?"?n??1t?????7@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetcht?v?4E??!??<g{Q??)t?v?4E??1??<g{Q??:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism ?4???!?aX???)B#ظ?]??1j??I????:Preprocessing2?
TIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl????R@!yJ?k??X@)???|?Rw?1?u8?&Ӹ?:Preprocessing2Y
"Iterator::Model::PrivateThreadPoole??Q??!?V4????)?(??/?r?1J?v???:Preprocessing2F
Iterator::Model?鲘?|??!???????)l%t??Yq?1P?c?w??:Preprocessing2?
PIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCache???~?S@!>?حb?X@)?<??S?Z?1B[\?!d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.