??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.0-dev202105042v1.12.1-56004-g99ec2f1e6878??
?
"word2_vec/w2v_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??(*3
shared_name$"word2_vec/w2v_embedding/embeddings
?
6word2_vec/w2v_embedding/embeddings/Read/ReadVariableOpReadVariableOp"word2_vec/w2v_embedding/embeddings* 
_output_shapes
:
??(*
dtype0
?
word2_vec/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??(*/
shared_name word2_vec/embedding/embeddings
?
2word2_vec/embedding/embeddings/Read/ReadVariableOpReadVariableOpword2_vec/embedding/embeddings* 
_output_shapes
:
??(*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
)Adam/word2_vec/w2v_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??(*:
shared_name+)Adam/word2_vec/w2v_embedding/embeddings/m
?
=Adam/word2_vec/w2v_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp)Adam/word2_vec/w2v_embedding/embeddings/m* 
_output_shapes
:
??(*
dtype0
?
%Adam/word2_vec/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??(*6
shared_name'%Adam/word2_vec/embedding/embeddings/m
?
9Adam/word2_vec/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp%Adam/word2_vec/embedding/embeddings/m* 
_output_shapes
:
??(*
dtype0
?
)Adam/word2_vec/w2v_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??(*:
shared_name+)Adam/word2_vec/w2v_embedding/embeddings/v
?
=Adam/word2_vec/w2v_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp)Adam/word2_vec/w2v_embedding/embeddings/v* 
_output_shapes
:
??(*
dtype0
?
%Adam/word2_vec/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??(*6
shared_name'%Adam/word2_vec/embedding/embeddings/v
?
9Adam/word2_vec/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp%Adam/word2_vec/embedding/embeddings/v* 
_output_shapes
:
??(*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
target_embedding
context_embedding
dots
flatten
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
d
iter

beta_1

beta_2
	 decay
!learning_ratemFmGvHvI

0
1
 

0
1
?
"layer_metrics
trainable_variables
regularization_losses
#layer_regularization_losses
$metrics

%layers
&non_trainable_variables
	variables
 
nl
VARIABLE_VALUE"word2_vec/w2v_embedding/embeddings6target_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
'layer_metrics
regularization_losses
trainable_variables
(layer_regularization_losses
)metrics

*layers
+non_trainable_variables
	variables
ki
VARIABLE_VALUEword2_vec/embedding/embeddings7context_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
,layer_metrics
regularization_losses
trainable_variables
-layer_regularization_losses
.metrics

/layers
0non_trainable_variables
	variables
 
 
 
?
1layer_metrics
regularization_losses
trainable_variables
2layer_regularization_losses
3metrics

4layers
5non_trainable_variables
	variables
 
 
 
?
6layer_metrics
regularization_losses
trainable_variables
7layer_regularization_losses
8metrics

9layers
:non_trainable_variables
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

;0
<1

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	=total
	>count
?	variables
@	keras_api
D
	Atotal
	Bcount
C
_fn_kwargs
D	variables
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

D	variables
??
VARIABLE_VALUE)Adam/word2_vec/w2v_embedding/embeddings/mRtarget_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adam/word2_vec/embedding/embeddings/mScontext_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/word2_vec/w2v_embedding/embeddings/vRtarget_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adam/word2_vec/embedding/embeddings/vScontext_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_input_2Placeholder*+
_output_shapes
:?????????*
dtype0	* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2"word2_vec/w2v_embedding/embeddingsword2_vec/embedding/embeddings*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_11801661
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6word2_vec/w2v_embedding/embeddings/Read/ReadVariableOp2word2_vec/embedding/embeddings/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp=Adam/word2_vec/w2v_embedding/embeddings/m/Read/ReadVariableOp9Adam/word2_vec/embedding/embeddings/m/Read/ReadVariableOp=Adam/word2_vec/w2v_embedding/embeddings/v/Read/ReadVariableOp9Adam/word2_vec/embedding/embeddings/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_11801815
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"word2_vec/w2v_embedding/embeddingsword2_vec/embedding/embeddings	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1)Adam/word2_vec/w2v_embedding/embeddings/m%Adam/word2_vec/embedding/embeddings/m)Adam/word2_vec/w2v_embedding/embeddings/v%Adam/word2_vec/embedding/embeddings/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_11801870??
?

?
G__inference_embedding_layer_call_and_return_conditional_losses_11801552

inputs	-
embedding_lookup_11801546:
??(
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_11801546inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*,
_class"
 loc:@embedding_lookup/11801546*/
_output_shapes
:?????????(*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/11801546*/
_output_shapes
:?????????(2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:?????????(2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
k
A__inference_dot_layer_call_and_return_conditional_losses_11801592

inputs
inputs_1
identityu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputs_1transpose/perm:output:0*
T0*+
_output_shapes
:?????????(2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2]
stack/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2	
stack/1?
stackPackstrided_slice_1:output:0stack/1:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2
stack}
ReshapeReshapeinputsstack:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Reshape?
MatMulBatchMatMulV2Reshape:output:0transpose:y:0*
T0*4
_output_shapes"
 :??????????????????2
MatMulQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3?
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_4\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2strided_slice_3:output:0strided_slice:output:0strided_slice_4:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat}
	Reshape_1ReshapeMatMul:output:0concat:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_1n
IdentityIdentityReshape_1:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????(:?????????(:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
,__inference_embedding_layer_call_fn_11801693

inputs	
unknown:
??(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_118015522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_11801600

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_word2_vec_layer_call_fn_11801614
input_1
input_2	
unknown:
??(
	unknown_0:
??(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_word2_vec_layer_call_and_return_conditional_losses_118016032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
&__inference_signature_wrapper_11801661
input_1
input_2	
unknown:
??(
	unknown_0:
??(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_118015252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
G__inference_word2_vec_layer_call_and_return_conditional_losses_11801603
input_1
input_2	*
w2v_embedding_11801540:
??(&
embedding_11801553:
??(
identity??!embedding/StatefulPartitionedCall?%w2v_embedding/StatefulPartitionedCall?
%w2v_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1w2v_embedding_11801540*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_w2v_embedding_layer_call_and_return_conditional_losses_118015392'
%w2v_embedding/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_11801553*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_118015522#
!embedding/StatefulPartitionedCall?
dot/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0.w2v_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dot_layer_call_and_return_conditional_losses_118015922
dot/PartitionedCall?
flatten/PartitionedCallPartitionedCalldot/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_118016002
flatten/PartitionedCall{
IdentityIdentity flatten/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall&^w2v_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2N
%w2v_embedding/StatefulPartitionedCall%w2v_embedding/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
K__inference_w2v_embedding_layer_call_and_return_conditional_losses_11801539

inputs-
embedding_lookup_11801533:
??(
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_11801533inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/11801533*+
_output_shapes
:?????????(*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/11801533*+
_output_shapes
:?????????(2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????(2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????(2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_layer_call_fn_11801746

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_118016002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?F
?
#__inference__wrapped_model_11801525
input_1
input_2	E
1word2_vec_w2v_embedding_embedding_lookup_11801481:
??(A
-word2_vec_embedding_embedding_lookup_11801486:
??(
identity??$word2_vec/embedding/embedding_lookup?(word2_vec/w2v_embedding/embedding_lookup?
(word2_vec/w2v_embedding/embedding_lookupResourceGather1word2_vec_w2v_embedding_embedding_lookup_11801481input_1",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@word2_vec/w2v_embedding/embedding_lookup/11801481*+
_output_shapes
:?????????(*
dtype02*
(word2_vec/w2v_embedding/embedding_lookup?
1word2_vec/w2v_embedding/embedding_lookup/IdentityIdentity1word2_vec/w2v_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@word2_vec/w2v_embedding/embedding_lookup/11801481*+
_output_shapes
:?????????(23
1word2_vec/w2v_embedding/embedding_lookup/Identity?
3word2_vec/w2v_embedding/embedding_lookup/Identity_1Identity:word2_vec/w2v_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????(25
3word2_vec/w2v_embedding/embedding_lookup/Identity_1?
$word2_vec/embedding/embedding_lookupResourceGather-word2_vec_embedding_embedding_lookup_11801486input_2",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*@
_class6
42loc:@word2_vec/embedding/embedding_lookup/11801486*/
_output_shapes
:?????????(*
dtype02&
$word2_vec/embedding/embedding_lookup?
-word2_vec/embedding/embedding_lookup/IdentityIdentity-word2_vec/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@word2_vec/embedding/embedding_lookup/11801486*/
_output_shapes
:?????????(2/
-word2_vec/embedding/embedding_lookup/Identity?
/word2_vec/embedding/embedding_lookup/Identity_1Identity6word2_vec/embedding/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:?????????(21
/word2_vec/embedding/embedding_lookup/Identity_1?
word2_vec/dot/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
word2_vec/dot/transpose/perm?
word2_vec/dot/transpose	Transpose<word2_vec/w2v_embedding/embedding_lookup/Identity_1:output:0%word2_vec/dot/transpose/perm:output:0*
T0*+
_output_shapes
:?????????(2
word2_vec/dot/transpose?
word2_vec/dot/ShapeShape8word2_vec/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
word2_vec/dot/Shape?
!word2_vec/dot/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!word2_vec/dot/strided_slice/stack?
#word2_vec/dot/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#word2_vec/dot/strided_slice/stack_1?
#word2_vec/dot/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#word2_vec/dot/strided_slice/stack_2?
word2_vec/dot/strided_sliceStridedSliceword2_vec/dot/Shape:output:0*word2_vec/dot/strided_slice/stack:output:0,word2_vec/dot/strided_slice/stack_1:output:0,word2_vec/dot/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
word2_vec/dot/strided_slice?
#word2_vec/dot/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#word2_vec/dot/strided_slice_1/stack?
%word2_vec/dot/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%word2_vec/dot/strided_slice_1/stack_1?
%word2_vec/dot/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%word2_vec/dot/strided_slice_1/stack_2?
word2_vec/dot/strided_slice_1StridedSliceword2_vec/dot/Shape:output:0,word2_vec/dot/strided_slice_1/stack:output:0.word2_vec/dot/strided_slice_1/stack_1:output:0.word2_vec/dot/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
word2_vec/dot/strided_slice_1?
#word2_vec/dot/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#word2_vec/dot/strided_slice_2/stack?
%word2_vec/dot/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%word2_vec/dot/strided_slice_2/stack_1?
%word2_vec/dot/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%word2_vec/dot/strided_slice_2/stack_2?
word2_vec/dot/strided_slice_2StridedSliceword2_vec/dot/Shape:output:0,word2_vec/dot/strided_slice_2/stack:output:0.word2_vec/dot/strided_slice_2/stack_1:output:0.word2_vec/dot/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
word2_vec/dot/strided_slice_2y
word2_vec/dot/stack/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
word2_vec/dot/stack/1?
word2_vec/dot/stackPack&word2_vec/dot/strided_slice_1:output:0word2_vec/dot/stack/1:output:0&word2_vec/dot/strided_slice_2:output:0*
N*
T0*
_output_shapes
:2
word2_vec/dot/stack?
word2_vec/dot/ReshapeReshape8word2_vec/embedding/embedding_lookup/Identity_1:output:0word2_vec/dot/stack:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
word2_vec/dot/Reshape?
word2_vec/dot/MatMulBatchMatMulV2word2_vec/dot/Reshape:output:0word2_vec/dot/transpose:y:0*
T0*4
_output_shapes"
 :??????????????????2
word2_vec/dot/MatMul{
word2_vec/dot/Shape_1Shapeword2_vec/dot/MatMul:output:0*
T0*
_output_shapes
:2
word2_vec/dot/Shape_1?
#word2_vec/dot/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#word2_vec/dot/strided_slice_3/stack?
%word2_vec/dot/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%word2_vec/dot/strided_slice_3/stack_1?
%word2_vec/dot/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%word2_vec/dot/strided_slice_3/stack_2?
word2_vec/dot/strided_slice_3StridedSliceword2_vec/dot/Shape_1:output:0,word2_vec/dot/strided_slice_3/stack:output:0.word2_vec/dot/strided_slice_3/stack_1:output:0.word2_vec/dot/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
word2_vec/dot/strided_slice_3?
#word2_vec/dot/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2%
#word2_vec/dot/strided_slice_4/stack?
%word2_vec/dot/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%word2_vec/dot/strided_slice_4/stack_1?
%word2_vec/dot/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%word2_vec/dot/strided_slice_4/stack_2?
word2_vec/dot/strided_slice_4StridedSliceword2_vec/dot/Shape_1:output:0,word2_vec/dot/strided_slice_4/stack:output:0.word2_vec/dot/strided_slice_4/stack_1:output:0.word2_vec/dot/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
word2_vec/dot/strided_slice_4x
word2_vec/dot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
word2_vec/dot/concat/axis?
word2_vec/dot/concatConcatV2&word2_vec/dot/strided_slice_3:output:0$word2_vec/dot/strided_slice:output:0&word2_vec/dot/strided_slice_4:output:0"word2_vec/dot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
word2_vec/dot/concat?
word2_vec/dot/Reshape_1Reshapeword2_vec/dot/MatMul:output:0word2_vec/dot/concat:output:0*
T0*/
_output_shapes
:?????????2
word2_vec/dot/Reshape_1?
word2_vec/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
word2_vec/flatten/Const?
word2_vec/flatten/ReshapeReshape word2_vec/dot/Reshape_1:output:0 word2_vec/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
word2_vec/flatten/Reshape}
IdentityIdentity"word2_vec/flatten/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp%^word2_vec/embedding/embedding_lookup)^word2_vec/w2v_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : 2L
$word2_vec/embedding/embedding_lookup$word2_vec/embedding/embedding_lookup2T
(word2_vec/w2v_embedding/embedding_lookup(word2_vec/w2v_embedding/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_2
?D
?	
$__inference__traced_restore_11801870
file_prefixG
3assignvariableop_word2_vec_w2v_embedding_embeddings:
??(E
1assignvariableop_1_word2_vec_embedding_embeddings:
??(&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: Q
=assignvariableop_11_adam_word2_vec_w2v_embedding_embeddings_m:
??(M
9assignvariableop_12_adam_word2_vec_embedding_embeddings_m:
??(Q
=assignvariableop_13_adam_word2_vec_w2v_embedding_embeddings_v:
??(M
9assignvariableop_14_adam_word2_vec_embedding_embeddings_v:
??(
identity_16??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6target_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB7context_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRtarget_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBScontext_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRtarget_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBScontext_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp3assignvariableop_word2_vec_w2v_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp1assignvariableop_1_word2_vec_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp=assignvariableop_11_adam_word2_vec_w2v_embedding_embeddings_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_adam_word2_vec_embedding_embeddings_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp=assignvariableop_13_adam_word2_vec_w2v_embedding_embeddings_vIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp9assignvariableop_14_adam_word2_vec_embedding_embeddings_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_15f
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_16?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_16Identity_16:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
0__inference_w2v_embedding_layer_call_fn_11801677

inputs
unknown:
??(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_w2v_embedding_layer_call_and_return_conditional_losses_118015392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
R
&__inference_dot_layer_call_fn_11801735
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dot_layer_call_and_return_conditional_losses_118015922
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????(:?????????(:Y U
/
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????(
"
_user_specified_name
inputs/1
?

?
K__inference_w2v_embedding_layer_call_and_return_conditional_losses_11801670

inputs-
embedding_lookup_11801664:
??(
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_11801664inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/11801664*+
_output_shapes
:?????????(*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/11801664*+
_output_shapes
:?????????(2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????(2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????(2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_layer_call_and_return_conditional_losses_11801686

inputs	-
embedding_lookup_11801680:
??(
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_11801680inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*,
_class"
 loc:@embedding_lookup/11801680*/
_output_shapes
:?????????(*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/11801680*/
_output_shapes
:?????????(2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:?????????(2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*/
_output_shapes
:?????????(2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
m
A__inference_dot_layer_call_and_return_conditional_losses_11801729
inputs_0
inputs_1
identityu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputs_1transpose/perm:output:0*
T0*+
_output_shapes
:?????????(2
	transposeF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2]
stack/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2	
stack/1?
stackPackstrided_slice_1:output:0stack/1:output:0strided_slice_2:output:0*
N*
T0*
_output_shapes
:2
stack
ReshapeReshapeinputs_0stack:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Reshape?
MatMulBatchMatMulV2Reshape:output:0transpose:y:0*
T0*4
_output_shapes"
 :??????????????????2
MatMulQ
Shape_1ShapeMatMul:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3?
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceShape_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_4\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis?
concatConcatV2strided_slice_3:output:0strided_slice:output:0strided_slice_4:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat}
	Reshape_1ReshapeMatMul:output:0concat:output:0*
T0*/
_output_shapes
:?????????2
	Reshape_1n
IdentityIdentityReshape_1:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????(:?????????(:Y U
/
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????(
"
_user_specified_name
inputs/1
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_11801741

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
!__inference__traced_save_11801815
file_prefixA
=savev2_word2_vec_w2v_embedding_embeddings_read_readvariableop=
9savev2_word2_vec_embedding_embeddings_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopH
Dsavev2_adam_word2_vec_w2v_embedding_embeddings_m_read_readvariableopD
@savev2_adam_word2_vec_embedding_embeddings_m_read_readvariableopH
Dsavev2_adam_word2_vec_w2v_embedding_embeddings_v_read_readvariableopD
@savev2_adam_word2_vec_embedding_embeddings_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6target_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB7context_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRtarget_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBScontext_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRtarget_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBScontext_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_word2_vec_w2v_embedding_embeddings_read_readvariableop9savev2_word2_vec_embedding_embeddings_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopDsavev2_adam_word2_vec_w2v_embedding_embeddings_m_read_readvariableop@savev2_adam_word2_vec_embedding_embeddings_m_read_readvariableopDsavev2_adam_word2_vec_w2v_embedding_embeddings_v_read_readvariableop@savev2_adam_word2_vec_embedding_embeddings_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*q
_input_shapes`
^: :
??(:
??(: : : : : : : : : :
??(:
??(:
??(:
??(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??(:&"
 
_output_shapes
:
??(:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??(:&"
 
_output_shapes
:
??(:&"
 
_output_shapes
:
??(:&"
 
_output_shapes
:
??(:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
?
input_24
serving_default_input_2:0	?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?p
?
target_embedding
context_embedding
dots
flatten
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
*J&call_and_return_all_conditional_losses
K__call__
L_default_save_signature"?

_tf_keras_model?
{"name": "word2_vec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Word2Vec", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "__tuple__", "items": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1024, 1]}, "int32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1024, 4, 1]}, "int64", "input_2"]}]}, "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Word2Vec"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": true, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"?
_tf_keras_layer?{"name": "w2v_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "w2v_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 20000, "output_dim": 40, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 1]}}
?

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"?
_tf_keras_layer?{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "input_dim": 20000, "output_dim": 40, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 4}, "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 4, 1]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"?
_tf_keras_layer?{"name": "dot", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dot", "config": {"name": "dot", "trainable": true, "dtype": "float32", "axes": {"class_name": "__tuple__", "items": [3, 2]}, "normalize": false}, "shared_object_id": 6, "build_input_shape": [{"class_name": "TensorShape", "items": [1024, 4, 1, 40]}, {"class_name": "TensorShape", "items": [1024, 1, 40]}]}
?
regularization_losses
trainable_variables
	variables
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 8}}
w
iter

beta_1

beta_2
	 decay
!learning_ratemFmGvHvI"
	optimizer
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
"layer_metrics
trainable_variables
regularization_losses
#layer_regularization_losses
$metrics

%layers
&non_trainable_variables
	variables
K__call__
L_default_save_signature
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
,
Userving_default"
signature_map
6:4
??(2"word2_vec/w2v_embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
'layer_metrics
regularization_losses
trainable_variables
(layer_regularization_losses
)metrics

*layers
+non_trainable_variables
	variables
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
2:0
??(2word2_vec/embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
,layer_metrics
regularization_losses
trainable_variables
-layer_regularization_losses
.metrics

/layers
0non_trainable_variables
	variables
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1layer_metrics
regularization_losses
trainable_variables
2layer_regularization_losses
3metrics

4layers
5non_trainable_variables
	variables
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6layer_metrics
regularization_losses
trainable_variables
7layer_regularization_losses
8metrics

9layers
:non_trainable_variables
	variables
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	=total
	>count
?	variables
@	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 9}
?
	Atotal
	Bcount
C
_fn_kwargs
D	variables
E	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
:  (2total
:  (2count
.
=0
>1"
trackable_list_wrapper
-
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
;:9
??(2)Adam/word2_vec/w2v_embedding/embeddings/m
7:5
??(2%Adam/word2_vec/embedding/embeddings/m
;:9
??(2)Adam/word2_vec/w2v_embedding/embeddings/v
7:5
??(2%Adam/word2_vec/embedding/embeddings/v
?2?
G__inference_word2_vec_layer_call_and_return_conditional_losses_11801603?
???
FullArgSpec
args?
jself
jpair
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *R?O
M?J
!?
input_1?????????
%?"
input_2?????????	
?2?
,__inference_word2_vec_layer_call_fn_11801614?
???
FullArgSpec
args?
jself
jpair
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *R?O
M?J
!?
input_1?????????
%?"
input_2?????????	
?2?
#__inference__wrapped_model_11801525?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *R?O
M?J
!?
input_1?????????
%?"
input_2?????????	
?2?
K__inference_w2v_embedding_layer_call_and_return_conditional_losses_11801670?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_w2v_embedding_layer_call_fn_11801677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_layer_call_and_return_conditional_losses_11801686?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_layer_call_fn_11801693?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dot_layer_call_and_return_conditional_losses_11801729?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dot_layer_call_fn_11801735?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_layer_call_and_return_conditional_losses_11801741?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_layer_call_fn_11801746?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_11801661input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_11801525?\?Y
R?O
M?J
!?
input_1?????????
%?"
input_2?????????	
? "3?0
.
output_1"?
output_1??????????
A__inference_dot_layer_call_and_return_conditional_losses_11801729?f?c
\?Y
W?T
*?'
inputs/0?????????(
&?#
inputs/1?????????(
? "-?*
#? 
0?????????
? ?
&__inference_dot_layer_call_fn_11801735?f?c
\?Y
W?T
*?'
inputs/0?????????(
&?#
inputs/1?????????(
? " ???????????
G__inference_embedding_layer_call_and_return_conditional_losses_11801686g3?0
)?&
$?!
inputs?????????	
? "-?*
#? 
0?????????(
? ?
,__inference_embedding_layer_call_fn_11801693Z3?0
)?&
$?!
inputs?????????	
? " ??????????(?
E__inference_flatten_layer_call_and_return_conditional_losses_11801741`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????
? ?
*__inference_flatten_layer_call_fn_11801746S7?4
-?*
(?%
inputs?????????
? "???????????
&__inference_signature_wrapper_11801661?m?j
? 
c?`
,
input_1!?
input_1?????????
0
input_2%?"
input_2?????????	"3?0
.
output_1"?
output_1??????????
K__inference_w2v_embedding_layer_call_and_return_conditional_losses_11801670_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????(
? ?
0__inference_w2v_embedding_layer_call_fn_11801677R/?,
%?"
 ?
inputs?????????
? "??????????(?
G__inference_word2_vec_layer_call_and_return_conditional_losses_11801603?\?Y
R?O
M?J
!?
input_1?????????
%?"
input_2?????????	
? "%?"
?
0?????????
? ?
,__inference_word2_vec_layer_call_fn_11801614|\?Y
R?O
M?J
!?
input_1?????????
%?"
input_2?????????	
? "??????????