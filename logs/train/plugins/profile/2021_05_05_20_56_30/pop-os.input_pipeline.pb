  *	`??"{7?@2?
]Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2Q?|a?@!??3|L?X@)?$\?#x	@1?{[??D@:Preprocessing2?
fIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle?????O?@!T??M@)?E{??N@1?OɡBh@@:Preprocessing2?
sIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::TensorSlice?ٓ??<??!Y?ޔ<9@)ٓ??<??1Y?ޔ<9@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism?O?mp??!́c!??)?|%????1#?yƛ]??:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch?"?J %??!ᜉ u???)?"?J %??1ᜉ u???:Preprocessing2?
TIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl?c> й@!=???M?X@)_&???}?1?4?g??:Preprocessing2F
Iterator::Model??>???!?*?ʍ??)f?(?7s?1???Ү?:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?릔?J??!ӮfÛ???)߿yq??m?1-N?5???:Preprocessing2?
PIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::MemoryCache;9Cqǻ@!??Uj??X@)iUMu_?1??Ƅ?j??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.