??8
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
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
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
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
2
StopGradient

input"T
output"T"	
Ttype
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
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
 ?"serve*2.6.0-dev202105042v1.12.1-56004-g99ec2f1e6878??3
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
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:(*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
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
?
-transformer_block/multi_head_att/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*>
shared_name/-transformer_block/multi_head_att/dense/kernel
?
Atransformer_block/multi_head_att/dense/kernel/Read/ReadVariableOpReadVariableOp-transformer_block/multi_head_att/dense/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_1/kernel
?
Ctransformer_block/multi_head_att/dense_1/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_1/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_2/kernel
?
Ctransformer_block/multi_head_att/dense_2/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_2/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_3/kernel
?
Ctransformer_block/multi_head_att/dense_3/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_3/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_4/kernel
?
Ctransformer_block/multi_head_att/dense_4/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_4/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_5/kernel
?
Ctransformer_block/multi_head_att/dense_5/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_5/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_6/kernel
?
Ctransformer_block/multi_head_att/dense_6/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_6/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_7/kernel
?
Ctransformer_block/multi_head_att/dense_7/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_7/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*@
shared_name1/transformer_block/multi_head_att/dense_8/kernel
?
Ctransformer_block/multi_head_att/dense_8/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_8/kernel*
_output_shapes

:((*
dtype0
?
/transformer_block/multi_head_att/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x(*@
shared_name1/transformer_block/multi_head_att/dense_9/kernel
?
Ctransformer_block/multi_head_att/dense_9/kernel/Read/ReadVariableOpReadVariableOp/transformer_block/multi_head_att/dense_9/kernel*
_output_shapes

:x(*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( * 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:( *
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
: *
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: (* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

: (*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:(*
dtype0
?
+transformer_block/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*<
shared_name-+transformer_block/layer_normalization/gamma
?
?transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+transformer_block/layer_normalization/gamma*
_output_shapes
:(*
dtype0
?
*transformer_block/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*;
shared_name,*transformer_block/layer_normalization/beta
?
>transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*transformer_block/layer_normalization/beta*
_output_shapes
:(*
dtype0
?
-transformer_block/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*>
shared_name/-transformer_block/layer_normalization_1/gamma
?
Atransformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-transformer_block/layer_normalization_1/gamma*
_output_shapes
:(*
dtype0
?
,transformer_block/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*=
shared_name.,transformer_block/layer_normalization_1/beta
?
@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,transformer_block/layer_normalization_1/beta*
_output_shapes
:(*
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
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
?
4Adam/transformer_block/multi_head_att/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*E
shared_name64Adam/transformer_block/multi_head_att/dense/kernel/m
?
HAdam/transformer_block/multi_head_att/dense/kernel/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/multi_head_att/dense/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_1/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_1/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_2/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_2/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_3/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_3/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_4/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_4/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_5/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_5/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_6/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_6/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_7/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_7/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_8/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_8/kernel/m*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x(*G
shared_name86Adam/transformer_block/multi_head_att/dense_9/kernel/m
?
JAdam/transformer_block/multi_head_att/dense_9/kernel/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_9/kernel/m*
_output_shapes

:x(*
dtype0
?
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *'
shared_nameAdam/dense_10/kernel/m
?
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:( *
dtype0
?
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: (*'
shared_nameAdam/dense_11/kernel/m
?
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

: (*
dtype0
?
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/dense_11/bias/m
y
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes
:(*
dtype0
?
2Adam/transformer_block/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*C
shared_name42Adam/transformer_block/layer_normalization/gamma/m
?
FAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/m*
_output_shapes
:(*
dtype0
?
1Adam/transformer_block/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*B
shared_name31Adam/transformer_block/layer_normalization/beta/m
?
EAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/m*
_output_shapes
:(*
dtype0
?
4Adam/transformer_block/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/m
?
HAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/m*
_output_shapes
:(*
dtype0
?
3Adam/transformer_block/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/m
?
GAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/m*
_output_shapes
:(*
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
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0
?
4Adam/transformer_block/multi_head_att/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*E
shared_name64Adam/transformer_block/multi_head_att/dense/kernel/v
?
HAdam/transformer_block/multi_head_att/dense/kernel/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/multi_head_att/dense/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_1/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_1/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_2/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_2/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_3/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_3/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_4/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_4/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_5/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_5/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_6/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_6/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_7/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_7/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*G
shared_name86Adam/transformer_block/multi_head_att/dense_8/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_8/kernel/v*
_output_shapes

:((*
dtype0
?
6Adam/transformer_block/multi_head_att/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x(*G
shared_name86Adam/transformer_block/multi_head_att/dense_9/kernel/v
?
JAdam/transformer_block/multi_head_att/dense_9/kernel/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block/multi_head_att/dense_9/kernel/v*
_output_shapes

:x(*
dtype0
?
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *'
shared_nameAdam/dense_10/kernel/v
?
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:( *
dtype0
?
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: (*'
shared_nameAdam/dense_11/kernel/v
?
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

: (*
dtype0
?
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*%
shared_nameAdam/dense_11/bias/v
y
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes
:(*
dtype0
?
2Adam/transformer_block/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*C
shared_name42Adam/transformer_block/layer_normalization/gamma/v
?
FAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/v*
_output_shapes
:(*
dtype0
?
1Adam/transformer_block/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*B
shared_name31Adam/transformer_block/layer_normalization/beta/v
?
EAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/v*
_output_shapes
:(*
dtype0
?
4Adam/transformer_block/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/v
?
HAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/v*
_output_shapes
:(*
dtype0
?
3Adam/transformer_block/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/v
?
GAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/v*
_output_shapes
:(*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Н
valueŝB?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
?

embeddings
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratem?'m?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?v?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?
?
0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
'19
(20
121
222
 
?
0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
'19
(20
121
222
?
Nnon_trainable_variables

trainable_variables

Olayers
regularization_losses
Pmetrics
Qlayer_regularization_losses
	variables
Rlayer_metrics
 
rp
VARIABLE_VALUE"word2_vec/w2v_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
 
?
trainable_variables
	variables

Slayers
regularization_losses
Tmetrics
Ulayer_regularization_losses
Vnon_trainable_variables
Wlayer_metrics
?
	Xquery
Ykey
	Zvalue
[	linOutput
\trainable_variables
]	variables
^regularization_losses
_	keras_api
?
`layer_with_weights-0
`layer-0
alayer_with_weights-1
alayer-1
btrainable_variables
cregularization_losses
d	variables
e	keras_api
q
faxis
	Jgamma
Kbeta
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
q
kaxis
	Lgamma
Mbeta
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
R
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
R
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
 
?
trainable_variables
	variables

xlayers
regularization_losses
ymetrics
zlayer_regularization_losses
{non_trainable_variables
|layer_metrics
 
 
 
?
trainable_variables
 	variables

}layers
!regularization_losses
~metrics
layer_regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
#trainable_variables
$	variables
?layers
%regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
)trainable_variables
*	variables
?layers
+regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
-trainable_variables
.	variables
?layers
/regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
3trainable_variables
4	variables
?layers
5regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
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
sq
VARIABLE_VALUE-transformer_block/multi_head_att/dense/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_2/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_3/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_4/kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_5/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_6/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_7/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE/transformer_block/multi_head_att/dense_8/kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block/multi_head_att/dense_9/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_10/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_10/bias1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_11/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_11/bias1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+transformer_block/layer_normalization/gamma1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*transformer_block/layer_normalization/beta1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-transformer_block/layer_normalization_1/gamma1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,transformer_block/layer_normalization_1/beta1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

?0
?1
 
 
 
 
 
 
 

?0
?1
?2

?0
?1
?2

?0
?1
?2
b

Ekernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
F
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
 
?
\trainable_variables
]	variables
?layers
^regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
l

Fkernel
Gbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Hkernel
Ibias
?trainable_variables
?	variables
?regularization_losses
?	keras_api

F0
G1
H2
I3
 

F0
G1
H2
I3
?
?non_trainable_variables
btrainable_variables
?layers
cregularization_losses
?metrics
 ?layer_regularization_losses
d	variables
?layer_metrics
 

J0
K1

J0
K1
 
?
gtrainable_variables
h	variables
?layers
iregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
 

L0
M1

L0
M1
 
?
ltrainable_variables
m	variables
?layers
nregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
ptrainable_variables
q	variables
?layers
rregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
 
 
 
?
ttrainable_variables
u	variables
?layers
vregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
*
0
1
2
3
4
5
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
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
b

<kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

=kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

>kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

?kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

@kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

Akernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

Bkernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

Ckernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
b

Dkernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api

E0

E0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
O
?0
?1
?2
?3
?4
?5
?6
?7
?8
[9
 
 
 
 

F0
G1

F0
G1
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

H0
I1

H0
I1
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
 

`0
a1
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
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables

<0

<0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

=0

=0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

>0

>0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

?0

?0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

@0

@0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

A0

A0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

B0

B0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

C0

C0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics

D0

D0
 
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
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
??
VARIABLE_VALUE)Adam/word2_vec/w2v_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/transformer_block/multi_head_att/dense/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_2/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_3/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_4/kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_5/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_6/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_7/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_8/kernel/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_9/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_10/kernel/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_10/bias/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_11/kernel/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_11/bias/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)Adam/word2_vec/w2v_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/transformer_block/multi_head_att/dense/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_2/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_3/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_4/kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_5/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_6/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_7/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_8/kernel/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block/multi_head_att/dense_9/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_10/kernel/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_10/bias/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_11/kernel/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_11/bias/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1"word2_vec/w2v_embedding/embeddings-transformer_block/multi_head_att/dense/kernel/transformer_block/multi_head_att/dense_3/kernel/transformer_block/multi_head_att/dense_6/kernel/transformer_block/multi_head_att/dense_1/kernel/transformer_block/multi_head_att/dense_4/kernel/transformer_block/multi_head_att/dense_7/kernel/transformer_block/multi_head_att/dense_2/kernel/transformer_block/multi_head_att/dense_5/kernel/transformer_block/multi_head_att/dense_8/kernel/transformer_block/multi_head_att/dense_9/kernel+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/betadense_10/kerneldense_10/biasdense_11/kerneldense_11/bias-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betadense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_6727
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6word2_vec/w2v_embedding/embeddings/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpAtransformer_block/multi_head_att/dense/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_1/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_2/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_3/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_4/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_5/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_6/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_7/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_8/kernel/Read/ReadVariableOpCtransformer_block/multi_head_att/dense_9/kernel/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp?transformer_block/layer_normalization/gamma/Read/ReadVariableOp>transformer_block/layer_normalization/beta/Read/ReadVariableOpAtransformer_block/layer_normalization_1/gamma/Read/ReadVariableOp@transformer_block/layer_normalization_1/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp=Adam/word2_vec/w2v_embedding/embeddings/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOpHAdam/transformer_block/multi_head_att/dense/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_1/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_2/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_3/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_4/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_5/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_6/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_7/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_8/kernel/m/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_9/kernel/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOp=Adam/word2_vec/w2v_embedding/embeddings/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpHAdam/transformer_block/multi_head_att/dense/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_1/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_2/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_3/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_4/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_5/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_6/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_7/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_8/kernel/v/Read/ReadVariableOpJAdam/transformer_block/multi_head_att/dense_9/kernel/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpConst*[
TinT
R2P	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_9006
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"word2_vec/w2v_embedding/embeddingsdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate-transformer_block/multi_head_att/dense/kernel/transformer_block/multi_head_att/dense_1/kernel/transformer_block/multi_head_att/dense_2/kernel/transformer_block/multi_head_att/dense_3/kernel/transformer_block/multi_head_att/dense_4/kernel/transformer_block/multi_head_att/dense_5/kernel/transformer_block/multi_head_att/dense_6/kernel/transformer_block/multi_head_att/dense_7/kernel/transformer_block/multi_head_att/dense_8/kernel/transformer_block/multi_head_att/dense_9/kerneldense_10/kerneldense_10/biasdense_11/kerneldense_11/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betatotalcounttotal_1count_1)Adam/word2_vec/w2v_embedding/embeddings/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/m4Adam/transformer_block/multi_head_att/dense/kernel/m6Adam/transformer_block/multi_head_att/dense_1/kernel/m6Adam/transformer_block/multi_head_att/dense_2/kernel/m6Adam/transformer_block/multi_head_att/dense_3/kernel/m6Adam/transformer_block/multi_head_att/dense_4/kernel/m6Adam/transformer_block/multi_head_att/dense_5/kernel/m6Adam/transformer_block/multi_head_att/dense_6/kernel/m6Adam/transformer_block/multi_head_att/dense_7/kernel/m6Adam/transformer_block/multi_head_att/dense_8/kernel/m6Adam/transformer_block/multi_head_att/dense_9/kernel/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/m2Adam/transformer_block/layer_normalization/gamma/m1Adam/transformer_block/layer_normalization/beta/m4Adam/transformer_block/layer_normalization_1/gamma/m3Adam/transformer_block/layer_normalization_1/beta/m)Adam/word2_vec/w2v_embedding/embeddings/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v4Adam/transformer_block/multi_head_att/dense/kernel/v6Adam/transformer_block/multi_head_att/dense_1/kernel/v6Adam/transformer_block/multi_head_att/dense_2/kernel/v6Adam/transformer_block/multi_head_att/dense_3/kernel/v6Adam/transformer_block/multi_head_att/dense_4/kernel/v6Adam/transformer_block/multi_head_att/dense_5/kernel/v6Adam/transformer_block/multi_head_att/dense_6/kernel/v6Adam/transformer_block/multi_head_att/dense_7/kernel/v6Adam/transformer_block/multi_head_att/dense_8/kernel/v6Adam/transformer_block/multi_head_att/dense_9/kernel/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/v2Adam/transformer_block/layer_normalization/gamma/v1Adam/transformer_block/layer_normalization/beta/v4Adam/transformer_block/layer_normalization_1/gamma/v3Adam/transformer_block/layer_normalization_1/beta/v*Z
TinS
Q2O*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_9250??0
?
?
)__inference_sequential_layer_call_fn_5181
dense_10_input
unknown:( 
	unknown_0: 
	unknown_1: (
	unknown_2:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_51702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:??????????(
(
_user_specified_namedense_10_input
??
?
?__inference_model_layer_call_and_return_conditional_losses_7501

inputs7
#w2v_embedding_embedding_lookup_7104:
??(Z
Htransformer_block_multi_head_att_dense_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_3_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_6_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_1_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_4_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_7_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_2_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_5_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_8_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_9_tensordot_readvariableop_resource:x(Y
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:(U
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:(Y
Gtransformer_block_sequential_dense_10_tensordot_readvariableop_resource:( S
Etransformer_block_sequential_dense_10_biasadd_readvariableop_resource: Y
Gtransformer_block_sequential_dense_11_tensordot_readvariableop_resource: (S
Etransformer_block_sequential_dense_11_biasadd_readvariableop_resource:([
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:(W
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:(9
'dense_12_matmul_readvariableop_resource:(6
(dense_12_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:6
(dense_13_biasadd_readvariableop_resource:
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?>transformer_block/layer_normalization/batchnorm/ReadVariableOp?Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp??transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp?<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp?<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp?w2v_embedding/embedding_lookupz
w2v_embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????2
w2v_embedding/Cast?
w2v_embedding/embedding_lookupResourceGather#w2v_embedding_embedding_lookup_7104w2v_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@w2v_embedding/embedding_lookup/7104*,
_output_shapes
:??????????(*
dtype02 
w2v_embedding/embedding_lookup?
'w2v_embedding/embedding_lookup/IdentityIdentity'w2v_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@w2v_embedding/embedding_lookup/7104*,
_output_shapes
:??????????(2)
'w2v_embedding/embedding_lookup/Identity?
)w2v_embedding/embedding_lookup/Identity_1Identity0w2v_embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????(2+
)w2v_embedding/embedding_lookup/Identity_1?
&transformer_block/multi_head_att/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2(
&transformer_block/multi_head_att/Shape?
4transformer_block/multi_head_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4transformer_block/multi_head_att/strided_slice/stack?
6transformer_block/multi_head_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6transformer_block/multi_head_att/strided_slice/stack_1?
6transformer_block/multi_head_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6transformer_block/multi_head_att/strided_slice/stack_2?
.transformer_block/multi_head_att/strided_sliceStridedSlice/transformer_block/multi_head_att/Shape:output:0=transformer_block/multi_head_att/strided_slice/stack:output:0?transformer_block/multi_head_att/strided_slice/stack_1:output:0?transformer_block/multi_head_att/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.transformer_block/multi_head_att/strided_slice?
?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOpReadVariableOpHtransformer_block_multi_head_att_dense_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02A
?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?
5transformer_block/multi_head_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5transformer_block/multi_head_att/dense/Tensordot/axes?
5transformer_block/multi_head_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5transformer_block/multi_head_att/dense/Tensordot/free?
6transformer_block/multi_head_att/dense/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:28
6transformer_block/multi_head_att/dense/Tensordot/Shape?
>transformer_block/multi_head_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense/Tensordot/GatherV2/axis?
9transformer_block/multi_head_att/dense/Tensordot/GatherV2GatherV2?transformer_block/multi_head_att/dense/Tensordot/Shape:output:0>transformer_block/multi_head_att/dense/Tensordot/free:output:0Gtransformer_block/multi_head_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense/Tensordot/GatherV2?
@transformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axis?
;transformer_block/multi_head_att/dense/Tensordot/GatherV2_1GatherV2?transformer_block/multi_head_att/dense/Tensordot/Shape:output:0>transformer_block/multi_head_att/dense/Tensordot/axes:output:0Itransformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense/Tensordot/GatherV2_1?
6transformer_block/multi_head_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6transformer_block/multi_head_att/dense/Tensordot/Const?
5transformer_block/multi_head_att/dense/Tensordot/ProdProdBtransformer_block/multi_head_att/dense/Tensordot/GatherV2:output:0?transformer_block/multi_head_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5transformer_block/multi_head_att/dense/Tensordot/Prod?
8transformer_block/multi_head_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense/Tensordot/Const_1?
7transformer_block/multi_head_att/dense/Tensordot/Prod_1ProdDtransformer_block/multi_head_att/dense/Tensordot/GatherV2_1:output:0Atransformer_block/multi_head_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense/Tensordot/Prod_1?
<transformer_block/multi_head_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/multi_head_att/dense/Tensordot/concat/axis?
7transformer_block/multi_head_att/dense/Tensordot/concatConcatV2>transformer_block/multi_head_att/dense/Tensordot/free:output:0>transformer_block/multi_head_att/dense/Tensordot/axes:output:0Etransformer_block/multi_head_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7transformer_block/multi_head_att/dense/Tensordot/concat?
6transformer_block/multi_head_att/dense/Tensordot/stackPack>transformer_block/multi_head_att/dense/Tensordot/Prod:output:0@transformer_block/multi_head_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6transformer_block/multi_head_att/dense/Tensordot/stack?
:transformer_block/multi_head_att/dense/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0@transformer_block/multi_head_att/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2<
:transformer_block/multi_head_att/dense/Tensordot/transpose?
8transformer_block/multi_head_att/dense/Tensordot/ReshapeReshape>transformer_block/multi_head_att/dense/Tensordot/transpose:y:0?transformer_block/multi_head_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8transformer_block/multi_head_att/dense/Tensordot/Reshape?
7transformer_block/multi_head_att/dense/Tensordot/MatMulMatMulAtransformer_block/multi_head_att/dense/Tensordot/Reshape:output:0Gtransformer_block/multi_head_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(29
7transformer_block/multi_head_att/dense/Tensordot/MatMul?
8transformer_block/multi_head_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2:
8transformer_block/multi_head_att/dense/Tensordot/Const_2?
>transformer_block/multi_head_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense/Tensordot/concat_1/axis?
9transformer_block/multi_head_att/dense/Tensordot/concat_1ConcatV2Btransformer_block/multi_head_att/dense/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense/Tensordot/Const_2:output:0Gtransformer_block/multi_head_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense/Tensordot/concat_1?
0transformer_block/multi_head_att/dense/TensordotReshapeAtransformer_block/multi_head_att/dense/Tensordot/MatMul:product:0Btransformer_block/multi_head_att/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(22
0transformer_block/multi_head_att/dense/Tensordot?
Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_3/Tensordot/axes?
7transformer_block/multi_head_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_3/Tensordot/free?
8transformer_block/multi_head_att/dense_3/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_3/Tensordot/Shape?
@transformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_3/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_3/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_3/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_3/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_3/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_3/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_3/Tensordot/Const?
7transformer_block/multi_head_att/dense_3/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_3/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_3/Tensordot/Prod?
:transformer_block/multi_head_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_3/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_3/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_3/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_3/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_3/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_3/Tensordot/free:output:0@transformer_block/multi_head_att/dense_3/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_3/Tensordot/concat?
8transformer_block/multi_head_att/dense_3/Tensordot/stackPack@transformer_block/multi_head_att/dense_3/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_3/Tensordot/stack?
<transformer_block/multi_head_att/dense_3/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_3/Tensordot/transpose?
:transformer_block/multi_head_att/dense_3/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_3/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_3/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_3/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_3/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_3/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_3/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_3/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_3/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_3/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_3/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_3/TensordotReshapeCtransformer_block/multi_head_att/dense_3/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_3/Tensordot?
Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_6/Tensordot/axes?
7transformer_block/multi_head_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_6/Tensordot/free?
8transformer_block/multi_head_att/dense_6/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_6/Tensordot/Shape?
@transformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_6/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_6/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_6/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_6/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_6/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_6/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_6/Tensordot/Const?
7transformer_block/multi_head_att/dense_6/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_6/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_6/Tensordot/Prod?
:transformer_block/multi_head_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_6/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_6/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_6/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_6/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_6/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_6/Tensordot/free:output:0@transformer_block/multi_head_att/dense_6/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_6/Tensordot/concat?
8transformer_block/multi_head_att/dense_6/Tensordot/stackPack@transformer_block/multi_head_att/dense_6/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_6/Tensordot/stack?
<transformer_block/multi_head_att/dense_6/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_6/Tensordot/transpose?
:transformer_block/multi_head_att/dense_6/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_6/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_6/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_6/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_6/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_6/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_6/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_6/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_6/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_6/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_6/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_6/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_6/TensordotReshapeCtransformer_block/multi_head_att/dense_6/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_6/Tensordot?
'transformer_block/multi_head_att/MatMulBatchMatMulV29transformer_block/multi_head_att/dense/Tensordot:output:0;transformer_block/multi_head_att/dense_3/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2)
'transformer_block/multi_head_att/MatMul?
*transformer_block/multi_head_att/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2,
*transformer_block/multi_head_att/truediv/y?
(transformer_block/multi_head_att/truedivRealDiv0transformer_block/multi_head_att/MatMul:output:03transformer_block/multi_head_att/truediv/y:output:0*
T0*-
_output_shapes
:???????????2*
(transformer_block/multi_head_att/truediv?
(transformer_block/multi_head_att/SoftmaxSoftmax,transformer_block/multi_head_att/truediv:z:0*
T0*-
_output_shapes
:???????????2*
(transformer_block/multi_head_att/Softmax?
)transformer_block/multi_head_att/MatMul_1BatchMatMulV22transformer_block/multi_head_att/Softmax:softmax:0;transformer_block/multi_head_att/dense_6/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2+
)transformer_block/multi_head_att/MatMul_1?
Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_1/Tensordot/axes?
7transformer_block/multi_head_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_1/Tensordot/free?
8transformer_block/multi_head_att/dense_1/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_1/Tensordot/Shape?
@transformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_1/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_1/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_1/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_1/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_1/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_1/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_1/Tensordot/Const?
7transformer_block/multi_head_att/dense_1/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_1/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_1/Tensordot/Prod?
:transformer_block/multi_head_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_1/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_1/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_1/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_1/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_1/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_1/Tensordot/free:output:0@transformer_block/multi_head_att/dense_1/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_1/Tensordot/concat?
8transformer_block/multi_head_att/dense_1/Tensordot/stackPack@transformer_block/multi_head_att/dense_1/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_1/Tensordot/stack?
<transformer_block/multi_head_att/dense_1/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_1/Tensordot/transpose?
:transformer_block/multi_head_att/dense_1/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_1/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_1/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_1/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_1/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_1/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_1/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_1/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_1/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_1/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_1/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_1/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_1/TensordotReshapeCtransformer_block/multi_head_att/dense_1/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_1/Tensordot?
Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_4/Tensordot/axes?
7transformer_block/multi_head_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_4/Tensordot/free?
8transformer_block/multi_head_att/dense_4/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_4/Tensordot/Shape?
@transformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_4/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_4/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_4/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_4/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_4/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_4/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_4/Tensordot/Const?
7transformer_block/multi_head_att/dense_4/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_4/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_4/Tensordot/Prod?
:transformer_block/multi_head_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_4/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_4/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_4/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_4/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_4/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_4/Tensordot/free:output:0@transformer_block/multi_head_att/dense_4/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_4/Tensordot/concat?
8transformer_block/multi_head_att/dense_4/Tensordot/stackPack@transformer_block/multi_head_att/dense_4/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_4/Tensordot/stack?
<transformer_block/multi_head_att/dense_4/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_4/Tensordot/transpose?
:transformer_block/multi_head_att/dense_4/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_4/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_4/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_4/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_4/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_4/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_4/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_4/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_4/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_4/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_4/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_4/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_4/TensordotReshapeCtransformer_block/multi_head_att/dense_4/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_4/Tensordot?
Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_7_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_7/Tensordot/axes?
7transformer_block/multi_head_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_7/Tensordot/free?
8transformer_block/multi_head_att/dense_7/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_7/Tensordot/Shape?
@transformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_7/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_7/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_7/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_7/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_7/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_7/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_7/Tensordot/Const?
7transformer_block/multi_head_att/dense_7/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_7/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_7/Tensordot/Prod?
:transformer_block/multi_head_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_7/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_7/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_7/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_7/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_7/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_7/Tensordot/free:output:0@transformer_block/multi_head_att/dense_7/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_7/Tensordot/concat?
8transformer_block/multi_head_att/dense_7/Tensordot/stackPack@transformer_block/multi_head_att/dense_7/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_7/Tensordot/stack?
<transformer_block/multi_head_att/dense_7/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_7/Tensordot/transpose?
:transformer_block/multi_head_att/dense_7/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_7/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_7/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_7/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_7/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_7/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_7/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_7/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_7/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_7/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_7/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_7/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_7/TensordotReshapeCtransformer_block/multi_head_att/dense_7/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_7/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_7/Tensordot?
)transformer_block/multi_head_att/MatMul_2BatchMatMulV2;transformer_block/multi_head_att/dense_1/Tensordot:output:0;transformer_block/multi_head_att/dense_4/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2+
)transformer_block/multi_head_att/MatMul_2?
,transformer_block/multi_head_att/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2.
,transformer_block/multi_head_att/truediv_1/y?
*transformer_block/multi_head_att/truediv_1RealDiv2transformer_block/multi_head_att/MatMul_2:output:05transformer_block/multi_head_att/truediv_1/y:output:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/truediv_1?
*transformer_block/multi_head_att/Softmax_1Softmax.transformer_block/multi_head_att/truediv_1:z:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/Softmax_1?
)transformer_block/multi_head_att/MatMul_3BatchMatMulV24transformer_block/multi_head_att/Softmax_1:softmax:0;transformer_block/multi_head_att/dense_7/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2+
)transformer_block/multi_head_att/MatMul_3?
Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_2/Tensordot/axes?
7transformer_block/multi_head_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_2/Tensordot/free?
8transformer_block/multi_head_att/dense_2/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_2/Tensordot/Shape?
@transformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_2/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_2/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_2/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_2/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_2/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_2/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_2/Tensordot/Const?
7transformer_block/multi_head_att/dense_2/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_2/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_2/Tensordot/Prod?
:transformer_block/multi_head_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_2/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_2/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_2/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_2/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_2/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_2/Tensordot/free:output:0@transformer_block/multi_head_att/dense_2/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_2/Tensordot/concat?
8transformer_block/multi_head_att/dense_2/Tensordot/stackPack@transformer_block/multi_head_att/dense_2/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_2/Tensordot/stack?
<transformer_block/multi_head_att/dense_2/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_2/Tensordot/transpose?
:transformer_block/multi_head_att/dense_2/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_2/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_2/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_2/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_2/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_2/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_2/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_2/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_2/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_2/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_2/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_2/TensordotReshapeCtransformer_block/multi_head_att/dense_2/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_2/Tensordot?
Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_5_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_5/Tensordot/axes?
7transformer_block/multi_head_att/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_5/Tensordot/free?
8transformer_block/multi_head_att/dense_5/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_5/Tensordot/Shape?
@transformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_5/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_5/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_5/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_5/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_5/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_5/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_5/Tensordot/Const?
7transformer_block/multi_head_att/dense_5/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_5/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_5/Tensordot/Prod?
:transformer_block/multi_head_att/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_5/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_5/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_5/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_5/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_5/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_5/Tensordot/free:output:0@transformer_block/multi_head_att/dense_5/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_5/Tensordot/concat?
8transformer_block/multi_head_att/dense_5/Tensordot/stackPack@transformer_block/multi_head_att/dense_5/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_5/Tensordot/stack?
<transformer_block/multi_head_att/dense_5/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_5/Tensordot/transpose?
:transformer_block/multi_head_att/dense_5/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_5/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_5/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_5/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_5/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_5/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_5/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_5/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_5/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_5/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_5/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_5/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_5/TensordotReshapeCtransformer_block/multi_head_att/dense_5/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_5/Tensordot?
Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_8/Tensordot/axes?
7transformer_block/multi_head_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_8/Tensordot/free?
8transformer_block/multi_head_att/dense_8/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_8/Tensordot/Shape?
@transformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_8/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_8/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_8/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_8/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_8/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_8/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_8/Tensordot/Const?
7transformer_block/multi_head_att/dense_8/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_8/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_8/Tensordot/Prod?
:transformer_block/multi_head_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_8/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_8/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_8/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_8/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_8/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_8/Tensordot/free:output:0@transformer_block/multi_head_att/dense_8/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_8/Tensordot/concat?
8transformer_block/multi_head_att/dense_8/Tensordot/stackPack@transformer_block/multi_head_att/dense_8/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_8/Tensordot/stack?
<transformer_block/multi_head_att/dense_8/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_8/Tensordot/transpose?
:transformer_block/multi_head_att/dense_8/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_8/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_8/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_8/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_8/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_8/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_8/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_8/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_8/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_8/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_8/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_8/TensordotReshapeCtransformer_block/multi_head_att/dense_8/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_8/Tensordot?
)transformer_block/multi_head_att/MatMul_4BatchMatMulV2;transformer_block/multi_head_att/dense_2/Tensordot:output:0;transformer_block/multi_head_att/dense_5/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2+
)transformer_block/multi_head_att/MatMul_4?
,transformer_block/multi_head_att/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2.
,transformer_block/multi_head_att/truediv_2/y?
*transformer_block/multi_head_att/truediv_2RealDiv2transformer_block/multi_head_att/MatMul_4:output:05transformer_block/multi_head_att/truediv_2/y:output:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/truediv_2?
*transformer_block/multi_head_att/Softmax_2Softmax.transformer_block/multi_head_att/truediv_2:z:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/Softmax_2?
)transformer_block/multi_head_att/MatMul_5BatchMatMulV24transformer_block/multi_head_att/Softmax_2:softmax:0;transformer_block/multi_head_att/dense_8/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2+
)transformer_block/multi_head_att/MatMul_5?
,transformer_block/multi_head_att/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,transformer_block/multi_head_att/concat/axis?
'transformer_block/multi_head_att/concatConcatV22transformer_block/multi_head_att/MatMul_1:output:02transformer_block/multi_head_att/MatMul_3:output:02transformer_block/multi_head_att/MatMul_5:output:05transformer_block/multi_head_att/concat/axis:output:0*
N*
T0*,
_output_shapes
:??????????x2)
'transformer_block/multi_head_att/concat?
Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

:x(*
dtype02C
Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_9/Tensordot/axes?
7transformer_block/multi_head_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_9/Tensordot/free?
8transformer_block/multi_head_att/dense_9/Tensordot/ShapeShape0transformer_block/multi_head_att/concat:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_9/Tensordot/Shape?
@transformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_9/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_9/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_9/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_9/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_9/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_9/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_9/Tensordot/Const?
7transformer_block/multi_head_att/dense_9/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_9/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_9/Tensordot/Prod?
:transformer_block/multi_head_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_9/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_9/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_9/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_9/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_9/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_9/Tensordot/free:output:0@transformer_block/multi_head_att/dense_9/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_9/Tensordot/concat?
8transformer_block/multi_head_att/dense_9/Tensordot/stackPack@transformer_block/multi_head_att/dense_9/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_9/Tensordot/stack?
<transformer_block/multi_head_att/dense_9/Tensordot/transpose	Transpose0transformer_block/multi_head_att/concat:output:0Btransformer_block/multi_head_att/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????x2>
<transformer_block/multi_head_att/dense_9/Tensordot/transpose?
:transformer_block/multi_head_att/dense_9/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_9/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_9/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_9/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_9/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_9/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_9/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_9/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_9/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_9/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_9/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_9/TensordotReshapeCtransformer_block/multi_head_att/dense_9/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_9/Tensordot?
'transformer_block/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2)
'transformer_block/dropout/dropout/Const?
%transformer_block/dropout/dropout/MulMul;transformer_block/multi_head_att/dense_9/Tensordot:output:00transformer_block/dropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????(2'
%transformer_block/dropout/dropout/Mul?
'transformer_block/dropout/dropout/ShapeShape;transformer_block/multi_head_att/dense_9/Tensordot:output:0*
T0*
_output_shapes
:2)
'transformer_block/dropout/dropout/Shape?
>transformer_block/dropout/dropout/random_uniform/RandomUniformRandomUniform0transformer_block/dropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????(*
dtype02@
>transformer_block/dropout/dropout/random_uniform/RandomUniform?
0transformer_block/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>22
0transformer_block/dropout/dropout/GreaterEqual/y?
.transformer_block/dropout/dropout/GreaterEqualGreaterEqualGtransformer_block/dropout/dropout/random_uniform/RandomUniform:output:09transformer_block/dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????(20
.transformer_block/dropout/dropout/GreaterEqual?
&transformer_block/dropout/dropout/CastCast2transformer_block/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????(2(
&transformer_block/dropout/dropout/Cast?
'transformer_block/dropout/dropout/Mul_1Mul)transformer_block/dropout/dropout/Mul:z:0*transformer_block/dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????(2)
'transformer_block/dropout/dropout/Mul_1?
transformer_block/addAddV22w2v_embedding/embedding_lookup/Identity_1:output:0+transformer_block/dropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????(2
transformer_block/add?
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2F
Dtransformer_block/layer_normalization/moments/mean/reduction_indices?
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(24
2transformer_block/layer_normalization/moments/mean?
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:??????????2<
:transformer_block/layer_normalization/moments/StopGradient?
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2A
?transformer_block/layer_normalization/moments/SquaredDifference?
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block/layer_normalization/moments/variance/reduction_indices?
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(28
6transformer_block/layer_normalization/moments/variance?
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?527
5transformer_block/layer_normalization/batchnorm/add/y?
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????25
3transformer_block/layer_normalization/batchnorm/add?
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????27
5transformer_block/layer_normalization/batchnorm/Rsqrt?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02D
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(25
3transformer_block/layer_normalization/batchnorm/mul?
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization/batchnorm/mul_1?
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization/batchnorm/mul_2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02@
>transformer_block/layer_normalization/batchnorm/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(25
3transformer_block/layer_normalization/batchnorm/sub?
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization/batchnorm/add_1?
>transformer_block/sequential/dense_10/Tensordot/ReadVariableOpReadVariableOpGtransformer_block_sequential_dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02@
>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp?
4transformer_block/sequential/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:26
4transformer_block/sequential/dense_10/Tensordot/axes?
4transformer_block/sequential/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       26
4transformer_block/sequential/dense_10/Tensordot/free?
5transformer_block/sequential/dense_10/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_10/Tensordot/Shape?
=transformer_block/sequential/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_10/Tensordot/GatherV2/axis?
8transformer_block/sequential/dense_10/Tensordot/GatherV2GatherV2>transformer_block/sequential/dense_10/Tensordot/Shape:output:0=transformer_block/sequential/dense_10/Tensordot/free:output:0Ftransformer_block/sequential/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2:
8transformer_block/sequential/dense_10/Tensordot/GatherV2?
?transformer_block/sequential/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block/sequential/dense_10/Tensordot/GatherV2_1/axis?
:transformer_block/sequential/dense_10/Tensordot/GatherV2_1GatherV2>transformer_block/sequential/dense_10/Tensordot/Shape:output:0=transformer_block/sequential/dense_10/Tensordot/axes:output:0Htransformer_block/sequential/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2<
:transformer_block/sequential/dense_10/Tensordot/GatherV2_1?
5transformer_block/sequential/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 27
5transformer_block/sequential/dense_10/Tensordot/Const?
4transformer_block/sequential/dense_10/Tensordot/ProdProdAtransformer_block/sequential/dense_10/Tensordot/GatherV2:output:0>transformer_block/sequential/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 26
4transformer_block/sequential/dense_10/Tensordot/Prod?
7transformer_block/sequential/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7transformer_block/sequential/dense_10/Tensordot/Const_1?
6transformer_block/sequential/dense_10/Tensordot/Prod_1ProdCtransformer_block/sequential/dense_10/Tensordot/GatherV2_1:output:0@transformer_block/sequential/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 28
6transformer_block/sequential/dense_10/Tensordot/Prod_1?
;transformer_block/sequential/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;transformer_block/sequential/dense_10/Tensordot/concat/axis?
6transformer_block/sequential/dense_10/Tensordot/concatConcatV2=transformer_block/sequential/dense_10/Tensordot/free:output:0=transformer_block/sequential/dense_10/Tensordot/axes:output:0Dtransformer_block/sequential/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:28
6transformer_block/sequential/dense_10/Tensordot/concat?
5transformer_block/sequential/dense_10/Tensordot/stackPack=transformer_block/sequential/dense_10/Tensordot/Prod:output:0?transformer_block/sequential/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_10/Tensordot/stack?
9transformer_block/sequential/dense_10/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0?transformer_block/sequential/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2;
9transformer_block/sequential/dense_10/Tensordot/transpose?
7transformer_block/sequential/dense_10/Tensordot/ReshapeReshape=transformer_block/sequential/dense_10/Tensordot/transpose:y:0>transformer_block/sequential/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????29
7transformer_block/sequential/dense_10/Tensordot/Reshape?
6transformer_block/sequential/dense_10/Tensordot/MatMulMatMul@transformer_block/sequential/dense_10/Tensordot/Reshape:output:0Ftransformer_block/sequential/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 28
6transformer_block/sequential/dense_10/Tensordot/MatMul?
7transformer_block/sequential/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 29
7transformer_block/sequential/dense_10/Tensordot/Const_2?
=transformer_block/sequential/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_10/Tensordot/concat_1/axis?
8transformer_block/sequential/dense_10/Tensordot/concat_1ConcatV2Atransformer_block/sequential/dense_10/Tensordot/GatherV2:output:0@transformer_block/sequential/dense_10/Tensordot/Const_2:output:0Ftransformer_block/sequential/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/sequential/dense_10/Tensordot/concat_1?
/transformer_block/sequential/dense_10/TensordotReshape@transformer_block/sequential/dense_10/Tensordot/MatMul:product:0Atransformer_block/sequential/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 21
/transformer_block/sequential/dense_10/Tensordot?
<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOpEtransformer_block_sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?
-transformer_block/sequential/dense_10/BiasAddBiasAdd8transformer_block/sequential/dense_10/Tensordot:output:0Dtransformer_block/sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2/
-transformer_block/sequential/dense_10/BiasAdd?
*transformer_block/sequential/dense_10/ReluRelu6transformer_block/sequential/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2,
*transformer_block/sequential/dense_10/Relu?
>transformer_block/sequential/dense_11/Tensordot/ReadVariableOpReadVariableOpGtransformer_block_sequential_dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02@
>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp?
4transformer_block/sequential/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:26
4transformer_block/sequential/dense_11/Tensordot/axes?
4transformer_block/sequential/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       26
4transformer_block/sequential/dense_11/Tensordot/free?
5transformer_block/sequential/dense_11/Tensordot/ShapeShape8transformer_block/sequential/dense_10/Relu:activations:0*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_11/Tensordot/Shape?
=transformer_block/sequential/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_11/Tensordot/GatherV2/axis?
8transformer_block/sequential/dense_11/Tensordot/GatherV2GatherV2>transformer_block/sequential/dense_11/Tensordot/Shape:output:0=transformer_block/sequential/dense_11/Tensordot/free:output:0Ftransformer_block/sequential/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2:
8transformer_block/sequential/dense_11/Tensordot/GatherV2?
?transformer_block/sequential/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block/sequential/dense_11/Tensordot/GatherV2_1/axis?
:transformer_block/sequential/dense_11/Tensordot/GatherV2_1GatherV2>transformer_block/sequential/dense_11/Tensordot/Shape:output:0=transformer_block/sequential/dense_11/Tensordot/axes:output:0Htransformer_block/sequential/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2<
:transformer_block/sequential/dense_11/Tensordot/GatherV2_1?
5transformer_block/sequential/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 27
5transformer_block/sequential/dense_11/Tensordot/Const?
4transformer_block/sequential/dense_11/Tensordot/ProdProdAtransformer_block/sequential/dense_11/Tensordot/GatherV2:output:0>transformer_block/sequential/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 26
4transformer_block/sequential/dense_11/Tensordot/Prod?
7transformer_block/sequential/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7transformer_block/sequential/dense_11/Tensordot/Const_1?
6transformer_block/sequential/dense_11/Tensordot/Prod_1ProdCtransformer_block/sequential/dense_11/Tensordot/GatherV2_1:output:0@transformer_block/sequential/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 28
6transformer_block/sequential/dense_11/Tensordot/Prod_1?
;transformer_block/sequential/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;transformer_block/sequential/dense_11/Tensordot/concat/axis?
6transformer_block/sequential/dense_11/Tensordot/concatConcatV2=transformer_block/sequential/dense_11/Tensordot/free:output:0=transformer_block/sequential/dense_11/Tensordot/axes:output:0Dtransformer_block/sequential/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:28
6transformer_block/sequential/dense_11/Tensordot/concat?
5transformer_block/sequential/dense_11/Tensordot/stackPack=transformer_block/sequential/dense_11/Tensordot/Prod:output:0?transformer_block/sequential/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_11/Tensordot/stack?
9transformer_block/sequential/dense_11/Tensordot/transpose	Transpose8transformer_block/sequential/dense_10/Relu:activations:0?transformer_block/sequential/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2;
9transformer_block/sequential/dense_11/Tensordot/transpose?
7transformer_block/sequential/dense_11/Tensordot/ReshapeReshape=transformer_block/sequential/dense_11/Tensordot/transpose:y:0>transformer_block/sequential/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????29
7transformer_block/sequential/dense_11/Tensordot/Reshape?
6transformer_block/sequential/dense_11/Tensordot/MatMulMatMul@transformer_block/sequential/dense_11/Tensordot/Reshape:output:0Ftransformer_block/sequential/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(28
6transformer_block/sequential/dense_11/Tensordot/MatMul?
7transformer_block/sequential/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(29
7transformer_block/sequential/dense_11/Tensordot/Const_2?
=transformer_block/sequential/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_11/Tensordot/concat_1/axis?
8transformer_block/sequential/dense_11/Tensordot/concat_1ConcatV2Atransformer_block/sequential/dense_11/Tensordot/GatherV2:output:0@transformer_block/sequential/dense_11/Tensordot/Const_2:output:0Ftransformer_block/sequential/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/sequential/dense_11/Tensordot/concat_1?
/transformer_block/sequential/dense_11/TensordotReshape@transformer_block/sequential/dense_11/Tensordot/MatMul:product:0Atransformer_block/sequential/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(21
/transformer_block/sequential/dense_11/Tensordot?
<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOpEtransformer_block_sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02>
<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?
-transformer_block/sequential/dense_11/BiasAddBiasAdd8transformer_block/sequential/dense_11/Tensordot:output:0Dtransformer_block/sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2/
-transformer_block/sequential/dense_11/BiasAdd?
)transformer_block/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2+
)transformer_block/dropout_1/dropout/Const?
'transformer_block/dropout_1/dropout/MulMul6transformer_block/sequential/dense_11/BiasAdd:output:02transformer_block/dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:??????????(2)
'transformer_block/dropout_1/dropout/Mul?
)transformer_block/dropout_1/dropout/ShapeShape6transformer_block/sequential/dense_11/BiasAdd:output:0*
T0*
_output_shapes
:2+
)transformer_block/dropout_1/dropout/Shape?
@transformer_block/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????(*
dtype02B
@transformer_block/dropout_1/dropout/random_uniform/RandomUniform?
2transformer_block/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>24
2transformer_block/dropout_1/dropout/GreaterEqual/y?
0transformer_block/dropout_1/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_1/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????(22
0transformer_block/dropout_1/dropout/GreaterEqual?
(transformer_block/dropout_1/dropout/CastCast4transformer_block/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????(2*
(transformer_block/dropout_1/dropout/Cast?
)transformer_block/dropout_1/dropout/Mul_1Mul+transformer_block/dropout_1/dropout/Mul:z:0,transformer_block/dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????(2+
)transformer_block/dropout_1/dropout/Mul_1?
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????(2
transformer_block/add_1?
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2H
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indices?
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(26
4transformer_block/layer_normalization_1/moments/mean?
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:??????????2>
<transformer_block/layer_normalization_1/moments/StopGradient?
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2C
Atransformer_block/layer_normalization_1/moments/SquaredDifference?
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indices?
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2:
8transformer_block/layer_normalization_1/moments/variance?
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?529
7transformer_block/layer_normalization_1/batchnorm/add/y?
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????27
5transformer_block/layer_normalization_1/batchnorm/add?
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????29
7transformer_block/layer_normalization_1/batchnorm/Rsqrt?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02F
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization_1/batchnorm/mul?
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(29
7transformer_block/layer_normalization_1/batchnorm/mul_1?
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(29
7transformer_block/layer_normalization_1/batchnorm/mul_2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02B
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization_1/batchnorm/sub?
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(29
7transformer_block/layer_normalization_1/batchnorm/add_1?
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices?
global_average_pooling1d/MeanMean;transformer_block/layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????(2
global_average_pooling1d/Meanw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????(2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(2
dropout_2/dropout/Mul_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_2/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_12/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMuldense_12/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_3/dropout/Mul}
dropout_3/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_3/dropout/Mul_1?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldropout_3/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd|
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_13/Softmaxu
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp@^transformer_block/multi_head_att/dense/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp=^transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?^transformer_block/sequential/dense_10/Tensordot/ReadVariableOp=^transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?^transformer_block/sequential/dense_11/Tensordot/ReadVariableOp^w2v_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp2|
<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp2?
>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp2|
<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp2?
>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp2@
w2v_embedding/embedding_lookupw2v_embedding/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_7100

inputs7
#w2v_embedding_embedding_lookup_6731:
??(Z
Htransformer_block_multi_head_att_dense_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_3_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_6_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_1_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_4_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_7_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_2_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_5_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_8_tensordot_readvariableop_resource:((\
Jtransformer_block_multi_head_att_dense_9_tensordot_readvariableop_resource:x(Y
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:(U
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:(Y
Gtransformer_block_sequential_dense_10_tensordot_readvariableop_resource:( S
Etransformer_block_sequential_dense_10_biasadd_readvariableop_resource: Y
Gtransformer_block_sequential_dense_11_tensordot_readvariableop_resource: (S
Etransformer_block_sequential_dense_11_biasadd_readvariableop_resource:([
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:(W
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:(9
'dense_12_matmul_readvariableop_resource:(6
(dense_12_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:6
(dense_13_biasadd_readvariableop_resource:
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?>transformer_block/layer_normalization/batchnorm/ReadVariableOp?Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp??transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp?Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp?<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp?<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp?w2v_embedding/embedding_lookupz
w2v_embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????2
w2v_embedding/Cast?
w2v_embedding/embedding_lookupResourceGather#w2v_embedding_embedding_lookup_6731w2v_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@w2v_embedding/embedding_lookup/6731*,
_output_shapes
:??????????(*
dtype02 
w2v_embedding/embedding_lookup?
'w2v_embedding/embedding_lookup/IdentityIdentity'w2v_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@w2v_embedding/embedding_lookup/6731*,
_output_shapes
:??????????(2)
'w2v_embedding/embedding_lookup/Identity?
)w2v_embedding/embedding_lookup/Identity_1Identity0w2v_embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????(2+
)w2v_embedding/embedding_lookup/Identity_1?
&transformer_block/multi_head_att/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2(
&transformer_block/multi_head_att/Shape?
4transformer_block/multi_head_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4transformer_block/multi_head_att/strided_slice/stack?
6transformer_block/multi_head_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6transformer_block/multi_head_att/strided_slice/stack_1?
6transformer_block/multi_head_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6transformer_block/multi_head_att/strided_slice/stack_2?
.transformer_block/multi_head_att/strided_sliceStridedSlice/transformer_block/multi_head_att/Shape:output:0=transformer_block/multi_head_att/strided_slice/stack:output:0?transformer_block/multi_head_att/strided_slice/stack_1:output:0?transformer_block/multi_head_att/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.transformer_block/multi_head_att/strided_slice?
?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOpReadVariableOpHtransformer_block_multi_head_att_dense_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02A
?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?
5transformer_block/multi_head_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5transformer_block/multi_head_att/dense/Tensordot/axes?
5transformer_block/multi_head_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5transformer_block/multi_head_att/dense/Tensordot/free?
6transformer_block/multi_head_att/dense/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:28
6transformer_block/multi_head_att/dense/Tensordot/Shape?
>transformer_block/multi_head_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense/Tensordot/GatherV2/axis?
9transformer_block/multi_head_att/dense/Tensordot/GatherV2GatherV2?transformer_block/multi_head_att/dense/Tensordot/Shape:output:0>transformer_block/multi_head_att/dense/Tensordot/free:output:0Gtransformer_block/multi_head_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense/Tensordot/GatherV2?
@transformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axis?
;transformer_block/multi_head_att/dense/Tensordot/GatherV2_1GatherV2?transformer_block/multi_head_att/dense/Tensordot/Shape:output:0>transformer_block/multi_head_att/dense/Tensordot/axes:output:0Itransformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense/Tensordot/GatherV2_1?
6transformer_block/multi_head_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6transformer_block/multi_head_att/dense/Tensordot/Const?
5transformer_block/multi_head_att/dense/Tensordot/ProdProdBtransformer_block/multi_head_att/dense/Tensordot/GatherV2:output:0?transformer_block/multi_head_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5transformer_block/multi_head_att/dense/Tensordot/Prod?
8transformer_block/multi_head_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense/Tensordot/Const_1?
7transformer_block/multi_head_att/dense/Tensordot/Prod_1ProdDtransformer_block/multi_head_att/dense/Tensordot/GatherV2_1:output:0Atransformer_block/multi_head_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense/Tensordot/Prod_1?
<transformer_block/multi_head_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/multi_head_att/dense/Tensordot/concat/axis?
7transformer_block/multi_head_att/dense/Tensordot/concatConcatV2>transformer_block/multi_head_att/dense/Tensordot/free:output:0>transformer_block/multi_head_att/dense/Tensordot/axes:output:0Etransformer_block/multi_head_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7transformer_block/multi_head_att/dense/Tensordot/concat?
6transformer_block/multi_head_att/dense/Tensordot/stackPack>transformer_block/multi_head_att/dense/Tensordot/Prod:output:0@transformer_block/multi_head_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6transformer_block/multi_head_att/dense/Tensordot/stack?
:transformer_block/multi_head_att/dense/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0@transformer_block/multi_head_att/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2<
:transformer_block/multi_head_att/dense/Tensordot/transpose?
8transformer_block/multi_head_att/dense/Tensordot/ReshapeReshape>transformer_block/multi_head_att/dense/Tensordot/transpose:y:0?transformer_block/multi_head_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8transformer_block/multi_head_att/dense/Tensordot/Reshape?
7transformer_block/multi_head_att/dense/Tensordot/MatMulMatMulAtransformer_block/multi_head_att/dense/Tensordot/Reshape:output:0Gtransformer_block/multi_head_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(29
7transformer_block/multi_head_att/dense/Tensordot/MatMul?
8transformer_block/multi_head_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2:
8transformer_block/multi_head_att/dense/Tensordot/Const_2?
>transformer_block/multi_head_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense/Tensordot/concat_1/axis?
9transformer_block/multi_head_att/dense/Tensordot/concat_1ConcatV2Btransformer_block/multi_head_att/dense/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense/Tensordot/Const_2:output:0Gtransformer_block/multi_head_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense/Tensordot/concat_1?
0transformer_block/multi_head_att/dense/TensordotReshapeAtransformer_block/multi_head_att/dense/Tensordot/MatMul:product:0Btransformer_block/multi_head_att/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(22
0transformer_block/multi_head_att/dense/Tensordot?
Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_3/Tensordot/axes?
7transformer_block/multi_head_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_3/Tensordot/free?
8transformer_block/multi_head_att/dense_3/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_3/Tensordot/Shape?
@transformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_3/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_3/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_3/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_3/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_3/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_3/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_3/Tensordot/Const?
7transformer_block/multi_head_att/dense_3/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_3/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_3/Tensordot/Prod?
:transformer_block/multi_head_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_3/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_3/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_3/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_3/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_3/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_3/Tensordot/free:output:0@transformer_block/multi_head_att/dense_3/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_3/Tensordot/concat?
8transformer_block/multi_head_att/dense_3/Tensordot/stackPack@transformer_block/multi_head_att/dense_3/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_3/Tensordot/stack?
<transformer_block/multi_head_att/dense_3/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_3/Tensordot/transpose?
:transformer_block/multi_head_att/dense_3/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_3/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_3/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_3/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_3/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_3/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_3/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_3/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_3/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_3/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_3/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_3/TensordotReshapeCtransformer_block/multi_head_att/dense_3/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_3/Tensordot?
Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_6/Tensordot/axes?
7transformer_block/multi_head_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_6/Tensordot/free?
8transformer_block/multi_head_att/dense_6/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_6/Tensordot/Shape?
@transformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_6/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_6/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_6/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_6/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_6/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_6/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_6/Tensordot/Const?
7transformer_block/multi_head_att/dense_6/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_6/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_6/Tensordot/Prod?
:transformer_block/multi_head_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_6/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_6/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_6/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_6/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_6/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_6/Tensordot/free:output:0@transformer_block/multi_head_att/dense_6/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_6/Tensordot/concat?
8transformer_block/multi_head_att/dense_6/Tensordot/stackPack@transformer_block/multi_head_att/dense_6/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_6/Tensordot/stack?
<transformer_block/multi_head_att/dense_6/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_6/Tensordot/transpose?
:transformer_block/multi_head_att/dense_6/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_6/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_6/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_6/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_6/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_6/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_6/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_6/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_6/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_6/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_6/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_6/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_6/TensordotReshapeCtransformer_block/multi_head_att/dense_6/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_6/Tensordot?
'transformer_block/multi_head_att/MatMulBatchMatMulV29transformer_block/multi_head_att/dense/Tensordot:output:0;transformer_block/multi_head_att/dense_3/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2)
'transformer_block/multi_head_att/MatMul?
*transformer_block/multi_head_att/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2,
*transformer_block/multi_head_att/truediv/y?
(transformer_block/multi_head_att/truedivRealDiv0transformer_block/multi_head_att/MatMul:output:03transformer_block/multi_head_att/truediv/y:output:0*
T0*-
_output_shapes
:???????????2*
(transformer_block/multi_head_att/truediv?
(transformer_block/multi_head_att/SoftmaxSoftmax,transformer_block/multi_head_att/truediv:z:0*
T0*-
_output_shapes
:???????????2*
(transformer_block/multi_head_att/Softmax?
)transformer_block/multi_head_att/MatMul_1BatchMatMulV22transformer_block/multi_head_att/Softmax:softmax:0;transformer_block/multi_head_att/dense_6/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2+
)transformer_block/multi_head_att/MatMul_1?
Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_1/Tensordot/axes?
7transformer_block/multi_head_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_1/Tensordot/free?
8transformer_block/multi_head_att/dense_1/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_1/Tensordot/Shape?
@transformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_1/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_1/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_1/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_1/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_1/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_1/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_1/Tensordot/Const?
7transformer_block/multi_head_att/dense_1/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_1/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_1/Tensordot/Prod?
:transformer_block/multi_head_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_1/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_1/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_1/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_1/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_1/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_1/Tensordot/free:output:0@transformer_block/multi_head_att/dense_1/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_1/Tensordot/concat?
8transformer_block/multi_head_att/dense_1/Tensordot/stackPack@transformer_block/multi_head_att/dense_1/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_1/Tensordot/stack?
<transformer_block/multi_head_att/dense_1/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_1/Tensordot/transpose?
:transformer_block/multi_head_att/dense_1/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_1/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_1/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_1/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_1/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_1/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_1/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_1/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_1/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_1/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_1/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_1/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_1/TensordotReshapeCtransformer_block/multi_head_att/dense_1/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_1/Tensordot?
Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_4/Tensordot/axes?
7transformer_block/multi_head_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_4/Tensordot/free?
8transformer_block/multi_head_att/dense_4/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_4/Tensordot/Shape?
@transformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_4/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_4/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_4/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_4/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_4/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_4/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_4/Tensordot/Const?
7transformer_block/multi_head_att/dense_4/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_4/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_4/Tensordot/Prod?
:transformer_block/multi_head_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_4/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_4/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_4/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_4/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_4/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_4/Tensordot/free:output:0@transformer_block/multi_head_att/dense_4/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_4/Tensordot/concat?
8transformer_block/multi_head_att/dense_4/Tensordot/stackPack@transformer_block/multi_head_att/dense_4/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_4/Tensordot/stack?
<transformer_block/multi_head_att/dense_4/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_4/Tensordot/transpose?
:transformer_block/multi_head_att/dense_4/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_4/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_4/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_4/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_4/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_4/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_4/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_4/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_4/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_4/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_4/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_4/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_4/TensordotReshapeCtransformer_block/multi_head_att/dense_4/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_4/Tensordot?
Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_7_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_7/Tensordot/axes?
7transformer_block/multi_head_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_7/Tensordot/free?
8transformer_block/multi_head_att/dense_7/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_7/Tensordot/Shape?
@transformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_7/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_7/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_7/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_7/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_7/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_7/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_7/Tensordot/Const?
7transformer_block/multi_head_att/dense_7/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_7/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_7/Tensordot/Prod?
:transformer_block/multi_head_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_7/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_7/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_7/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_7/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_7/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_7/Tensordot/free:output:0@transformer_block/multi_head_att/dense_7/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_7/Tensordot/concat?
8transformer_block/multi_head_att/dense_7/Tensordot/stackPack@transformer_block/multi_head_att/dense_7/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_7/Tensordot/stack?
<transformer_block/multi_head_att/dense_7/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_7/Tensordot/transpose?
:transformer_block/multi_head_att/dense_7/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_7/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_7/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_7/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_7/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_7/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_7/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_7/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_7/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_7/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_7/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_7/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_7/TensordotReshapeCtransformer_block/multi_head_att/dense_7/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_7/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_7/Tensordot?
)transformer_block/multi_head_att/MatMul_2BatchMatMulV2;transformer_block/multi_head_att/dense_1/Tensordot:output:0;transformer_block/multi_head_att/dense_4/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2+
)transformer_block/multi_head_att/MatMul_2?
,transformer_block/multi_head_att/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2.
,transformer_block/multi_head_att/truediv_1/y?
*transformer_block/multi_head_att/truediv_1RealDiv2transformer_block/multi_head_att/MatMul_2:output:05transformer_block/multi_head_att/truediv_1/y:output:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/truediv_1?
*transformer_block/multi_head_att/Softmax_1Softmax.transformer_block/multi_head_att/truediv_1:z:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/Softmax_1?
)transformer_block/multi_head_att/MatMul_3BatchMatMulV24transformer_block/multi_head_att/Softmax_1:softmax:0;transformer_block/multi_head_att/dense_7/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2+
)transformer_block/multi_head_att/MatMul_3?
Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_2/Tensordot/axes?
7transformer_block/multi_head_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_2/Tensordot/free?
8transformer_block/multi_head_att/dense_2/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_2/Tensordot/Shape?
@transformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_2/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_2/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_2/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_2/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_2/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_2/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_2/Tensordot/Const?
7transformer_block/multi_head_att/dense_2/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_2/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_2/Tensordot/Prod?
:transformer_block/multi_head_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_2/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_2/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_2/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_2/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_2/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_2/Tensordot/free:output:0@transformer_block/multi_head_att/dense_2/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_2/Tensordot/concat?
8transformer_block/multi_head_att/dense_2/Tensordot/stackPack@transformer_block/multi_head_att/dense_2/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_2/Tensordot/stack?
<transformer_block/multi_head_att/dense_2/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_2/Tensordot/transpose?
:transformer_block/multi_head_att/dense_2/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_2/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_2/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_2/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_2/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_2/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_2/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_2/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_2/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_2/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_2/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_2/TensordotReshapeCtransformer_block/multi_head_att/dense_2/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_2/Tensordot?
Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_5_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_5/Tensordot/axes?
7transformer_block/multi_head_att/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_5/Tensordot/free?
8transformer_block/multi_head_att/dense_5/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_5/Tensordot/Shape?
@transformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_5/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_5/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_5/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_5/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_5/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_5/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_5/Tensordot/Const?
7transformer_block/multi_head_att/dense_5/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_5/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_5/Tensordot/Prod?
:transformer_block/multi_head_att/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_5/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_5/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_5/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_5/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_5/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_5/Tensordot/free:output:0@transformer_block/multi_head_att/dense_5/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_5/Tensordot/concat?
8transformer_block/multi_head_att/dense_5/Tensordot/stackPack@transformer_block/multi_head_att/dense_5/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_5/Tensordot/stack?
<transformer_block/multi_head_att/dense_5/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_5/Tensordot/transpose?
:transformer_block/multi_head_att/dense_5/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_5/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_5/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_5/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_5/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_5/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_5/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_5/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_5/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_5/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_5/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_5/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_5/TensordotReshapeCtransformer_block/multi_head_att/dense_5/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_5/Tensordot?
Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02C
Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_8/Tensordot/axes?
7transformer_block/multi_head_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_8/Tensordot/free?
8transformer_block/multi_head_att/dense_8/Tensordot/ShapeShape2w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_8/Tensordot/Shape?
@transformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_8/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_8/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_8/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_8/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_8/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_8/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_8/Tensordot/Const?
7transformer_block/multi_head_att/dense_8/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_8/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_8/Tensordot/Prod?
:transformer_block/multi_head_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_8/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_8/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_8/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_8/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_8/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_8/Tensordot/free:output:0@transformer_block/multi_head_att/dense_8/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_8/Tensordot/concat?
8transformer_block/multi_head_att/dense_8/Tensordot/stackPack@transformer_block/multi_head_att/dense_8/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_8/Tensordot/stack?
<transformer_block/multi_head_att/dense_8/Tensordot/transpose	Transpose2w2v_embedding/embedding_lookup/Identity_1:output:0Btransformer_block/multi_head_att/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2>
<transformer_block/multi_head_att/dense_8/Tensordot/transpose?
:transformer_block/multi_head_att/dense_8/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_8/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_8/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_8/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_8/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_8/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_8/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_8/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_8/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_8/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_8/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_8/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_8/TensordotReshapeCtransformer_block/multi_head_att/dense_8/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_8/Tensordot?
)transformer_block/multi_head_att/MatMul_4BatchMatMulV2;transformer_block/multi_head_att/dense_2/Tensordot:output:0;transformer_block/multi_head_att/dense_5/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2+
)transformer_block/multi_head_att/MatMul_4?
,transformer_block/multi_head_att/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2.
,transformer_block/multi_head_att/truediv_2/y?
*transformer_block/multi_head_att/truediv_2RealDiv2transformer_block/multi_head_att/MatMul_4:output:05transformer_block/multi_head_att/truediv_2/y:output:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/truediv_2?
*transformer_block/multi_head_att/Softmax_2Softmax.transformer_block/multi_head_att/truediv_2:z:0*
T0*-
_output_shapes
:???????????2,
*transformer_block/multi_head_att/Softmax_2?
)transformer_block/multi_head_att/MatMul_5BatchMatMulV24transformer_block/multi_head_att/Softmax_2:softmax:0;transformer_block/multi_head_att/dense_8/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2+
)transformer_block/multi_head_att/MatMul_5?
,transformer_block/multi_head_att/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,transformer_block/multi_head_att/concat/axis?
'transformer_block/multi_head_att/concatConcatV22transformer_block/multi_head_att/MatMul_1:output:02transformer_block/multi_head_att/MatMul_3:output:02transformer_block/multi_head_att/MatMul_5:output:05transformer_block/multi_head_att/concat/axis:output:0*
N*
T0*,
_output_shapes
:??????????x2)
'transformer_block/multi_head_att/concat?
Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_multi_head_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

:x(*
dtype02C
Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp?
7transformer_block/multi_head_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block/multi_head_att/dense_9/Tensordot/axes?
7transformer_block/multi_head_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block/multi_head_att/dense_9/Tensordot/free?
8transformer_block/multi_head_att/dense_9/Tensordot/ShapeShape0transformer_block/multi_head_att/concat:output:0*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_9/Tensordot/Shape?
@transformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axis?
;transformer_block/multi_head_att/dense_9/Tensordot/GatherV2GatherV2Atransformer_block/multi_head_att/dense_9/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_9/Tensordot/free:output:0Itransformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_9/Tensordot/GatherV2?
Btransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axis?
=transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1GatherV2Atransformer_block/multi_head_att/dense_9/Tensordot/Shape:output:0@transformer_block/multi_head_att/dense_9/Tensordot/axes:output:0Ktransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1?
8transformer_block/multi_head_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block/multi_head_att/dense_9/Tensordot/Const?
7transformer_block/multi_head_att/dense_9/Tensordot/ProdProdDtransformer_block/multi_head_att/dense_9/Tensordot/GatherV2:output:0Atransformer_block/multi_head_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block/multi_head_att/dense_9/Tensordot/Prod?
:transformer_block/multi_head_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block/multi_head_att/dense_9/Tensordot/Const_1?
9transformer_block/multi_head_att/dense_9/Tensordot/Prod_1ProdFtransformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1:output:0Ctransformer_block/multi_head_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block/multi_head_att/dense_9/Tensordot/Prod_1?
>transformer_block/multi_head_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/multi_head_att/dense_9/Tensordot/concat/axis?
9transformer_block/multi_head_att/dense_9/Tensordot/concatConcatV2@transformer_block/multi_head_att/dense_9/Tensordot/free:output:0@transformer_block/multi_head_att/dense_9/Tensordot/axes:output:0Gtransformer_block/multi_head_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_att/dense_9/Tensordot/concat?
8transformer_block/multi_head_att/dense_9/Tensordot/stackPack@transformer_block/multi_head_att/dense_9/Tensordot/Prod:output:0Btransformer_block/multi_head_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/multi_head_att/dense_9/Tensordot/stack?
<transformer_block/multi_head_att/dense_9/Tensordot/transpose	Transpose0transformer_block/multi_head_att/concat:output:0Btransformer_block/multi_head_att/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????x2>
<transformer_block/multi_head_att/dense_9/Tensordot/transpose?
:transformer_block/multi_head_att/dense_9/Tensordot/ReshapeReshape@transformer_block/multi_head_att/dense_9/Tensordot/transpose:y:0Atransformer_block/multi_head_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block/multi_head_att/dense_9/Tensordot/Reshape?
9transformer_block/multi_head_att/dense_9/Tensordot/MatMulMatMulCtransformer_block/multi_head_att/dense_9/Tensordot/Reshape:output:0Itransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2;
9transformer_block/multi_head_att/dense_9/Tensordot/MatMul?
:transformer_block/multi_head_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2<
:transformer_block/multi_head_att/dense_9/Tensordot/Const_2?
@transformer_block/multi_head_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block/multi_head_att/dense_9/Tensordot/concat_1/axis?
;transformer_block/multi_head_att/dense_9/Tensordot/concat_1ConcatV2Dtransformer_block/multi_head_att/dense_9/Tensordot/GatherV2:output:0Ctransformer_block/multi_head_att/dense_9/Tensordot/Const_2:output:0Itransformer_block/multi_head_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_att/dense_9/Tensordot/concat_1?
2transformer_block/multi_head_att/dense_9/TensordotReshapeCtransformer_block/multi_head_att/dense_9/Tensordot/MatMul:product:0Dtransformer_block/multi_head_att/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(24
2transformer_block/multi_head_att/dense_9/Tensordot?
"transformer_block/dropout/IdentityIdentity;transformer_block/multi_head_att/dense_9/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2$
"transformer_block/dropout/Identity?
transformer_block/addAddV22w2v_embedding/embedding_lookup/Identity_1:output:0+transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:??????????(2
transformer_block/add?
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2F
Dtransformer_block/layer_normalization/moments/mean/reduction_indices?
2transformer_block/layer_normalization/moments/meanMeantransformer_block/add:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(24
2transformer_block/layer_normalization/moments/mean?
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:??????????2<
:transformer_block/layer_normalization/moments/StopGradient?
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2A
?transformer_block/layer_normalization/moments/SquaredDifference?
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block/layer_normalization/moments/variance/reduction_indices?
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(28
6transformer_block/layer_normalization/moments/variance?
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?527
5transformer_block/layer_normalization/batchnorm/add/y?
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????25
3transformer_block/layer_normalization/batchnorm/add?
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????27
5transformer_block/layer_normalization/batchnorm/Rsqrt?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02D
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(25
3transformer_block/layer_normalization/batchnorm/mul?
5transformer_block/layer_normalization/batchnorm/mul_1Multransformer_block/add:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization/batchnorm/mul_1?
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization/batchnorm/mul_2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02@
>transformer_block/layer_normalization/batchnorm/ReadVariableOp?
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(25
3transformer_block/layer_normalization/batchnorm/sub?
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization/batchnorm/add_1?
>transformer_block/sequential/dense_10/Tensordot/ReadVariableOpReadVariableOpGtransformer_block_sequential_dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02@
>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp?
4transformer_block/sequential/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:26
4transformer_block/sequential/dense_10/Tensordot/axes?
4transformer_block/sequential/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       26
4transformer_block/sequential/dense_10/Tensordot/free?
5transformer_block/sequential/dense_10/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_10/Tensordot/Shape?
=transformer_block/sequential/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_10/Tensordot/GatherV2/axis?
8transformer_block/sequential/dense_10/Tensordot/GatherV2GatherV2>transformer_block/sequential/dense_10/Tensordot/Shape:output:0=transformer_block/sequential/dense_10/Tensordot/free:output:0Ftransformer_block/sequential/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2:
8transformer_block/sequential/dense_10/Tensordot/GatherV2?
?transformer_block/sequential/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block/sequential/dense_10/Tensordot/GatherV2_1/axis?
:transformer_block/sequential/dense_10/Tensordot/GatherV2_1GatherV2>transformer_block/sequential/dense_10/Tensordot/Shape:output:0=transformer_block/sequential/dense_10/Tensordot/axes:output:0Htransformer_block/sequential/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2<
:transformer_block/sequential/dense_10/Tensordot/GatherV2_1?
5transformer_block/sequential/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 27
5transformer_block/sequential/dense_10/Tensordot/Const?
4transformer_block/sequential/dense_10/Tensordot/ProdProdAtransformer_block/sequential/dense_10/Tensordot/GatherV2:output:0>transformer_block/sequential/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 26
4transformer_block/sequential/dense_10/Tensordot/Prod?
7transformer_block/sequential/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7transformer_block/sequential/dense_10/Tensordot/Const_1?
6transformer_block/sequential/dense_10/Tensordot/Prod_1ProdCtransformer_block/sequential/dense_10/Tensordot/GatherV2_1:output:0@transformer_block/sequential/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 28
6transformer_block/sequential/dense_10/Tensordot/Prod_1?
;transformer_block/sequential/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;transformer_block/sequential/dense_10/Tensordot/concat/axis?
6transformer_block/sequential/dense_10/Tensordot/concatConcatV2=transformer_block/sequential/dense_10/Tensordot/free:output:0=transformer_block/sequential/dense_10/Tensordot/axes:output:0Dtransformer_block/sequential/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:28
6transformer_block/sequential/dense_10/Tensordot/concat?
5transformer_block/sequential/dense_10/Tensordot/stackPack=transformer_block/sequential/dense_10/Tensordot/Prod:output:0?transformer_block/sequential/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_10/Tensordot/stack?
9transformer_block/sequential/dense_10/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0?transformer_block/sequential/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2;
9transformer_block/sequential/dense_10/Tensordot/transpose?
7transformer_block/sequential/dense_10/Tensordot/ReshapeReshape=transformer_block/sequential/dense_10/Tensordot/transpose:y:0>transformer_block/sequential/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????29
7transformer_block/sequential/dense_10/Tensordot/Reshape?
6transformer_block/sequential/dense_10/Tensordot/MatMulMatMul@transformer_block/sequential/dense_10/Tensordot/Reshape:output:0Ftransformer_block/sequential/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 28
6transformer_block/sequential/dense_10/Tensordot/MatMul?
7transformer_block/sequential/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 29
7transformer_block/sequential/dense_10/Tensordot/Const_2?
=transformer_block/sequential/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_10/Tensordot/concat_1/axis?
8transformer_block/sequential/dense_10/Tensordot/concat_1ConcatV2Atransformer_block/sequential/dense_10/Tensordot/GatherV2:output:0@transformer_block/sequential/dense_10/Tensordot/Const_2:output:0Ftransformer_block/sequential/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/sequential/dense_10/Tensordot/concat_1?
/transformer_block/sequential/dense_10/TensordotReshape@transformer_block/sequential/dense_10/Tensordot/MatMul:product:0Atransformer_block/sequential/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 21
/transformer_block/sequential/dense_10/Tensordot?
<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOpEtransformer_block_sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?
-transformer_block/sequential/dense_10/BiasAddBiasAdd8transformer_block/sequential/dense_10/Tensordot:output:0Dtransformer_block/sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2/
-transformer_block/sequential/dense_10/BiasAdd?
*transformer_block/sequential/dense_10/ReluRelu6transformer_block/sequential/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2,
*transformer_block/sequential/dense_10/Relu?
>transformer_block/sequential/dense_11/Tensordot/ReadVariableOpReadVariableOpGtransformer_block_sequential_dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02@
>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp?
4transformer_block/sequential/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:26
4transformer_block/sequential/dense_11/Tensordot/axes?
4transformer_block/sequential/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       26
4transformer_block/sequential/dense_11/Tensordot/free?
5transformer_block/sequential/dense_11/Tensordot/ShapeShape8transformer_block/sequential/dense_10/Relu:activations:0*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_11/Tensordot/Shape?
=transformer_block/sequential/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_11/Tensordot/GatherV2/axis?
8transformer_block/sequential/dense_11/Tensordot/GatherV2GatherV2>transformer_block/sequential/dense_11/Tensordot/Shape:output:0=transformer_block/sequential/dense_11/Tensordot/free:output:0Ftransformer_block/sequential/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2:
8transformer_block/sequential/dense_11/Tensordot/GatherV2?
?transformer_block/sequential/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block/sequential/dense_11/Tensordot/GatherV2_1/axis?
:transformer_block/sequential/dense_11/Tensordot/GatherV2_1GatherV2>transformer_block/sequential/dense_11/Tensordot/Shape:output:0=transformer_block/sequential/dense_11/Tensordot/axes:output:0Htransformer_block/sequential/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2<
:transformer_block/sequential/dense_11/Tensordot/GatherV2_1?
5transformer_block/sequential/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 27
5transformer_block/sequential/dense_11/Tensordot/Const?
4transformer_block/sequential/dense_11/Tensordot/ProdProdAtransformer_block/sequential/dense_11/Tensordot/GatherV2:output:0>transformer_block/sequential/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 26
4transformer_block/sequential/dense_11/Tensordot/Prod?
7transformer_block/sequential/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 29
7transformer_block/sequential/dense_11/Tensordot/Const_1?
6transformer_block/sequential/dense_11/Tensordot/Prod_1ProdCtransformer_block/sequential/dense_11/Tensordot/GatherV2_1:output:0@transformer_block/sequential/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 28
6transformer_block/sequential/dense_11/Tensordot/Prod_1?
;transformer_block/sequential/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;transformer_block/sequential/dense_11/Tensordot/concat/axis?
6transformer_block/sequential/dense_11/Tensordot/concatConcatV2=transformer_block/sequential/dense_11/Tensordot/free:output:0=transformer_block/sequential/dense_11/Tensordot/axes:output:0Dtransformer_block/sequential/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:28
6transformer_block/sequential/dense_11/Tensordot/concat?
5transformer_block/sequential/dense_11/Tensordot/stackPack=transformer_block/sequential/dense_11/Tensordot/Prod:output:0?transformer_block/sequential/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_11/Tensordot/stack?
9transformer_block/sequential/dense_11/Tensordot/transpose	Transpose8transformer_block/sequential/dense_10/Relu:activations:0?transformer_block/sequential/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2;
9transformer_block/sequential/dense_11/Tensordot/transpose?
7transformer_block/sequential/dense_11/Tensordot/ReshapeReshape=transformer_block/sequential/dense_11/Tensordot/transpose:y:0>transformer_block/sequential/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????29
7transformer_block/sequential/dense_11/Tensordot/Reshape?
6transformer_block/sequential/dense_11/Tensordot/MatMulMatMul@transformer_block/sequential/dense_11/Tensordot/Reshape:output:0Ftransformer_block/sequential/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(28
6transformer_block/sequential/dense_11/Tensordot/MatMul?
7transformer_block/sequential/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(29
7transformer_block/sequential/dense_11/Tensordot/Const_2?
=transformer_block/sequential/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
=transformer_block/sequential/dense_11/Tensordot/concat_1/axis?
8transformer_block/sequential/dense_11/Tensordot/concat_1ConcatV2Atransformer_block/sequential/dense_11/Tensordot/GatherV2:output:0@transformer_block/sequential/dense_11/Tensordot/Const_2:output:0Ftransformer_block/sequential/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block/sequential/dense_11/Tensordot/concat_1?
/transformer_block/sequential/dense_11/TensordotReshape@transformer_block/sequential/dense_11/Tensordot/MatMul:product:0Atransformer_block/sequential/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(21
/transformer_block/sequential/dense_11/Tensordot?
<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOpEtransformer_block_sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02>
<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?
-transformer_block/sequential/dense_11/BiasAddBiasAdd8transformer_block/sequential/dense_11/Tensordot:output:0Dtransformer_block/sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2/
-transformer_block/sequential/dense_11/BiasAdd?
$transformer_block/dropout_1/IdentityIdentity6transformer_block/sequential/dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:??????????(2&
$transformer_block/dropout_1/Identity?
transformer_block/add_1AddV29transformer_block/layer_normalization/batchnorm/add_1:z:0-transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:??????????(2
transformer_block/add_1?
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2H
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indices?
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add_1:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(26
4transformer_block/layer_normalization_1/moments/mean?
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:??????????2>
<transformer_block/layer_normalization_1/moments/StopGradient?
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add_1:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2C
Atransformer_block/layer_normalization_1/moments/SquaredDifference?
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indices?
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2:
8transformer_block/layer_normalization_1/moments/variance?
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?529
7transformer_block/layer_normalization_1/batchnorm/add/y?
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????27
5transformer_block/layer_normalization_1/batchnorm/add?
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????29
7transformer_block/layer_normalization_1/batchnorm/Rsqrt?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02F
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization_1/batchnorm/mul?
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add_1:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(29
7transformer_block/layer_normalization_1/batchnorm/mul_1?
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(29
7transformer_block/layer_normalization_1/batchnorm/mul_2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02B
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(27
5transformer_block/layer_normalization_1/batchnorm/sub?
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(29
7transformer_block/layer_normalization_1/batchnorm/add_1?
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices?
global_average_pooling1d/MeanMean;transformer_block/layer_normalization_1/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????(2
global_average_pooling1d/Mean?
dropout_2/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:?????????(2
dropout_2/Identity?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMuldropout_2/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_12/Relu?
dropout_3/IdentityIdentitydense_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_3/Identity?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMuldropout_3/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdd|
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_13/Softmaxu
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp@^transformer_block/multi_head_att/dense/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpB^transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp=^transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?^transformer_block/sequential/dense_10/Tensordot/ReadVariableOp=^transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?^transformer_block/sequential/dense_11/Tensordot/ReadVariableOp^w2v_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2?
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2?
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2?
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2?
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp2?
Atransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOpAtransformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp2|
<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp<transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp2?
>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp>transformer_block/sequential/dense_10/Tensordot/ReadVariableOp2|
<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp<transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp2?
>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp>transformer_block/sequential/dense_11/Tensordot/ReadVariableOp2@
w2v_embedding/embedding_lookupw2v_embedding/embedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_8500

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8426

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????(2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????(:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5230

inputs
dense_10_5219:( 
dense_10_5221: 
dense_11_5224: (
dense_11_5226:(
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_5219dense_10_5221*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_51272"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5224dense_11_5226*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_51632"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
'__inference_dense_10_layer_call_fn_8710

inputs
unknown:( 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_51272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
? 
?
B__inference_dense_11_layer_call_and_return_conditional_losses_8740

inputs3
!tensordot_readvariableop_resource: (-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?-
?	
?__inference_model_layer_call_and_return_conditional_losses_6668
input_1&
w2v_embedding_6614:
??((
transformer_block_6617:(((
transformer_block_6619:(((
transformer_block_6621:(((
transformer_block_6623:(((
transformer_block_6625:(((
transformer_block_6627:(((
transformer_block_6629:(((
transformer_block_6631:(((
transformer_block_6633:(((
transformer_block_6635:x($
transformer_block_6637:($
transformer_block_6639:((
transformer_block_6641:( $
transformer_block_6643: (
transformer_block_6645: ($
transformer_block_6647:($
transformer_block_6649:($
transformer_block_6651:(
dense_12_6656:(
dense_12_6658:
dense_13_6662:
dense_13_6664:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?%w2v_embedding/StatefulPartitionedCall?
%w2v_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1w2v_embedding_6614*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_53232'
%w2v_embedding/StatefulPartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall.w2v_embedding/StatefulPartitionedCall:output:0transformer_block_6617transformer_block_6619transformer_block_6621transformer_block_6623transformer_block_6625transformer_block_6627transformer_block_6629transformer_block_6631transformer_block_6633transformer_block_6635transformer_block_6637transformer_block_6639transformer_block_6641transformer_block_6643transformer_block_6645transformer_block_6647transformer_block_6649transformer_block_6651*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_62972+
)transformer_block/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_57192*
(global_average_pooling1d/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_58822#
!dropout_2/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_12_6656dense_12_6658*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_57392"
 dense_12/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_58492#
!dropout_3/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_13_6662dense_13_6664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_57632"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall&^w2v_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2N
%w2v_embedding/StatefulPartitionedCall%w2v_embedding/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5719

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????(2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????(:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_5726

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????(2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
D
(__inference_dropout_2_layer_call_fn_8458

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
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_57262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_8670

inputs
unknown:( 
	unknown_0: 
	unknown_1: (
	unknown_2:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_52302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
D
(__inference_dropout_3_layer_call_fn_8505

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_57502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_6727
input_1
unknown:
??(
	unknown_0:((
	unknown_1:((
	unknown_2:((
	unknown_3:((
	unknown_4:((
	unknown_5:((
	unknown_6:((
	unknown_7:((
	unknown_8:((
	unknown_9:x(

unknown_10:(

unknown_11:(

unknown_12:( 

unknown_13: 

unknown_14: (

unknown_15:(

unknown_16:(

unknown_17:(

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_50892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
'__inference_dense_12_layer_call_fn_8483

inputs
unknown:(
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_57392
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
S
7__inference_global_average_pooling1d_layer_call_fn_8436

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
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_57192
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????(:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_5254
dense_10_input
unknown:( 
	unknown_0: 
	unknown_1: (
	unknown_2:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_10_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_52302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:??????????(
(
_user_specified_namedense_10_input
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5282
dense_10_input
dense_10_5271:( 
dense_10_5273: 
dense_11_5276: (
dense_11_5278:(
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_5271dense_10_5273*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_51272"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5276dense_11_5278*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_51632"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:\ X
,
_output_shapes
:??????????(
(
_user_specified_namedense_10_input
?

?
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_7613

inputs)
embedding_lookup_7607:
??(
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_7607Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/7607*,
_output_shapes
:??????????(*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/7607*,
_output_shapes
:??????????(2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????(2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_transformer_block_layer_call_fn_8373

inputs
unknown:((
	unknown_0:((
	unknown_1:((
	unknown_2:((
	unknown_3:((
	unknown_4:((
	unknown_5:((
	unknown_6:((
	unknown_7:((
	unknown_8:x(
	unknown_9:(

unknown_10:(

unknown_11:( 

unknown_12: 

unknown_13: (

unknown_14:(

unknown_15:(

unknown_16:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_56762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????(: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?K
?
D__inference_sequential_layer_call_and_return_conditional_losses_8644

inputs<
*dense_10_tensordot_readvariableop_resource:( 6
(dense_10_biasadd_readvariableop_resource: <
*dense_11_tensordot_readvariableop_resource: (6
(dense_11_biasadd_readvariableop_resource:(
identity??dense_10/BiasAdd/ReadVariableOp?!dense_10/Tensordot/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/Tensordot/ReadVariableOp?
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes?
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/freej
dense_10/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape?
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axis?
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2?
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis?
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const?
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod?
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1?
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1?
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axis?
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat?
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack?
dense_10/Tensordot/transpose	Transposeinputs"dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2
dense_10/Tensordot/transpose?
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_10/Tensordot/Reshape?
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_10/Tensordot/MatMul?
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_2?
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axis?
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1?
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
dense_10/Tensordot?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2
dense_10/BiasAddx
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
dense_10/Relu?
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axes?
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/free
dense_11/Tensordot/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dense_11/Tensordot/Shape?
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axis?
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2?
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axis?
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2_1~
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const?
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod?
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1?
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1?
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axis?
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat?
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stack?
dense_11/Tensordot/transpose	Transposedense_10/Relu:activations:0"dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2
dense_11/Tensordot/transpose?
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_11/Tensordot/Reshape?
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_11/Tensordot/MatMul?
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_11/Tensordot/Const_2?
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axis?
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1?
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
dense_11/Tensordot?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2
dense_11/BiasAddy
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?*
?
?__inference_model_layer_call_and_return_conditional_losses_5770

inputs&
w2v_embedding_5324:
??((
transformer_block_5677:(((
transformer_block_5679:(((
transformer_block_5681:(((
transformer_block_5683:(((
transformer_block_5685:(((
transformer_block_5687:(((
transformer_block_5689:(((
transformer_block_5691:(((
transformer_block_5693:(((
transformer_block_5695:x($
transformer_block_5697:($
transformer_block_5699:((
transformer_block_5701:( $
transformer_block_5703: (
transformer_block_5705: ($
transformer_block_5707:($
transformer_block_5709:($
transformer_block_5711:(
dense_12_5740:(
dense_12_5742:
dense_13_5764:
dense_13_5766:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?%w2v_embedding/StatefulPartitionedCall?
%w2v_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_5324*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_53232'
%w2v_embedding/StatefulPartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall.w2v_embedding/StatefulPartitionedCall:output:0transformer_block_5677transformer_block_5679transformer_block_5681transformer_block_5683transformer_block_5685transformer_block_5687transformer_block_5689transformer_block_5691transformer_block_5693transformer_block_5695transformer_block_5697transformer_block_5699transformer_block_5701transformer_block_5703transformer_block_5705transformer_block_5707transformer_block_5709transformer_block_5711*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_56762+
)transformer_block/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_57192*
(global_average_pooling1d/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_57262
dropout_2/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_12_5740dense_12_5742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_57392"
 dense_12/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_57502
dropout_3/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_13_5764dense_13_5766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_57632"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall&^w2v_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2N
%w2v_embedding/StatefulPartitionedCall%w2v_embedding/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_3_layer_call_fn_8510

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_58492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_5292

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8420

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_8488

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_11_layer_call_fn_8749

inputs
unknown: (
	unknown_0:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_51632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
B__inference_dense_12_layer_call_and_return_conditional_losses_5739

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_5089
input_1=
)model_w2v_embedding_embedding_lookup_4720:
??(`
Nmodel_transformer_block_multi_head_att_dense_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_3_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_6_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_1_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_4_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_7_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_2_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_5_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_8_tensordot_readvariableop_resource:((b
Pmodel_transformer_block_multi_head_att_dense_9_tensordot_readvariableop_resource:x(_
Qmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:([
Mmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource:(_
Mmodel_transformer_block_sequential_dense_10_tensordot_readvariableop_resource:( Y
Kmodel_transformer_block_sequential_dense_10_biasadd_readvariableop_resource: _
Mmodel_transformer_block_sequential_dense_11_tensordot_readvariableop_resource: (Y
Kmodel_transformer_block_sequential_dense_11_biasadd_readvariableop_resource:(a
Smodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:(]
Omodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource:(?
-model_dense_12_matmul_readvariableop_resource:(<
.model_dense_12_biasadd_readvariableop_resource:?
-model_dense_13_matmul_readvariableop_resource:<
.model_dense_13_biasadd_readvariableop_resource:
identity??%model/dense_12/BiasAdd/ReadVariableOp?$model/dense_12/MatMul/ReadVariableOp?%model/dense_13/BiasAdd/ReadVariableOp?$model/dense_13/MatMul/ReadVariableOp?Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp?Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?Emodel/transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp?Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp?Bmodel/transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?Dmodel/transformer_block/sequential/dense_10/Tensordot/ReadVariableOp?Bmodel/transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?Dmodel/transformer_block/sequential/dense_11/Tensordot/ReadVariableOp?$model/w2v_embedding/embedding_lookup?
model/w2v_embedding/CastCastinput_1*

DstT0*

SrcT0*(
_output_shapes
:??????????2
model/w2v_embedding/Cast?
$model/w2v_embedding/embedding_lookupResourceGather)model_w2v_embedding_embedding_lookup_4720model/w2v_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model/w2v_embedding/embedding_lookup/4720*,
_output_shapes
:??????????(*
dtype02&
$model/w2v_embedding/embedding_lookup?
-model/w2v_embedding/embedding_lookup/IdentityIdentity-model/w2v_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model/w2v_embedding/embedding_lookup/4720*,
_output_shapes
:??????????(2/
-model/w2v_embedding/embedding_lookup/Identity?
/model/w2v_embedding/embedding_lookup/Identity_1Identity6model/w2v_embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????(21
/model/w2v_embedding/embedding_lookup/Identity_1?
,model/transformer_block/multi_head_att/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2.
,model/transformer_block/multi_head_att/Shape?
:model/transformer_block/multi_head_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:model/transformer_block/multi_head_att/strided_slice/stack?
<model/transformer_block/multi_head_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/transformer_block/multi_head_att/strided_slice/stack_1?
<model/transformer_block/multi_head_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model/transformer_block/multi_head_att/strided_slice/stack_2?
4model/transformer_block/multi_head_att/strided_sliceStridedSlice5model/transformer_block/multi_head_att/Shape:output:0Cmodel/transformer_block/multi_head_att/strided_slice/stack:output:0Emodel/transformer_block/multi_head_att/strided_slice/stack_1:output:0Emodel/transformer_block/multi_head_att/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model/transformer_block/multi_head_att/strided_slice?
Emodel/transformer_block/multi_head_att/dense/Tensordot/ReadVariableOpReadVariableOpNmodel_transformer_block_multi_head_att_dense_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02G
Emodel/transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp?
;model/transformer_block/multi_head_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2=
;model/transformer_block/multi_head_att/dense/Tensordot/axes?
;model/transformer_block/multi_head_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2=
;model/transformer_block/multi_head_att/dense/Tensordot/free?
<model/transformer_block/multi_head_att/dense/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2>
<model/transformer_block/multi_head_att/dense/Tensordot/Shape?
Dmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2/axis?
?model/transformer_block/multi_head_att/dense/Tensordot/GatherV2GatherV2Emodel/transformer_block/multi_head_att/dense/Tensordot/Shape:output:0Dmodel/transformer_block/multi_head_att/dense/Tensordot/free:output:0Mmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense/Tensordot/GatherV2?
Fmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axis?
Amodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2_1GatherV2Emodel/transformer_block/multi_head_att/dense/Tensordot/Shape:output:0Dmodel/transformer_block/multi_head_att/dense/Tensordot/axes:output:0Omodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2_1?
<model/transformer_block/multi_head_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<model/transformer_block/multi_head_att/dense/Tensordot/Const?
;model/transformer_block/multi_head_att/dense/Tensordot/ProdProdHmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2:output:0Emodel/transformer_block/multi_head_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2=
;model/transformer_block/multi_head_att/dense/Tensordot/Prod?
>model/transformer_block/multi_head_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense/Tensordot/Const_1?
=model/transformer_block/multi_head_att/dense/Tensordot/Prod_1ProdJmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2_1:output:0Gmodel/transformer_block/multi_head_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense/Tensordot/Prod_1?
Bmodel/transformer_block/multi_head_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmodel/transformer_block/multi_head_att/dense/Tensordot/concat/axis?
=model/transformer_block/multi_head_att/dense/Tensordot/concatConcatV2Dmodel/transformer_block/multi_head_att/dense/Tensordot/free:output:0Dmodel/transformer_block/multi_head_att/dense/Tensordot/axes:output:0Kmodel/transformer_block/multi_head_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
=model/transformer_block/multi_head_att/dense/Tensordot/concat?
<model/transformer_block/multi_head_att/dense/Tensordot/stackPackDmodel/transformer_block/multi_head_att/dense/Tensordot/Prod:output:0Fmodel/transformer_block/multi_head_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2>
<model/transformer_block/multi_head_att/dense/Tensordot/stack?
@model/transformer_block/multi_head_att/dense/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Fmodel/transformer_block/multi_head_att/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2B
@model/transformer_block/multi_head_att/dense/Tensordot/transpose?
>model/transformer_block/multi_head_att/dense/Tensordot/ReshapeReshapeDmodel/transformer_block/multi_head_att/dense/Tensordot/transpose:y:0Emodel/transformer_block/multi_head_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2@
>model/transformer_block/multi_head_att/dense/Tensordot/Reshape?
=model/transformer_block/multi_head_att/dense/Tensordot/MatMulMatMulGmodel/transformer_block/multi_head_att/dense/Tensordot/Reshape:output:0Mmodel/transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2?
=model/transformer_block/multi_head_att/dense/Tensordot/MatMul?
>model/transformer_block/multi_head_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2@
>model/transformer_block/multi_head_att/dense/Tensordot/Const_2?
Dmodel/transformer_block/multi_head_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense/Tensordot/concat_1/axis?
?model/transformer_block/multi_head_att/dense/Tensordot/concat_1ConcatV2Hmodel/transformer_block/multi_head_att/dense/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense/Tensordot/Const_2:output:0Mmodel/transformer_block/multi_head_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense/Tensordot/concat_1?
6model/transformer_block/multi_head_att/dense/TensordotReshapeGmodel/transformer_block/multi_head_att/dense/Tensordot/MatMul:product:0Hmodel/transformer_block/multi_head_att/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(28
6model/transformer_block/multi_head_att/dense/Tensordot?
Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_3/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_3/Tensordot/free?
>model/transformer_block/multi_head_att/dense_3/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_3/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_3/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_3/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_3/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_3/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_3/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_3/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_3/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_3/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_3/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_3/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_3/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_3/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_3/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_3/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_3/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_3/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_3/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_3/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_3/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_3/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_3/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_3/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_3/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_3/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_3/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_3/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_3/TensordotReshapeImodel/transformer_block/multi_head_att/dense_3/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_3/Tensordot?
Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_6/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_6/Tensordot/free?
>model/transformer_block/multi_head_att/dense_6/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_6/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_6/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_6/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_6/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_6/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_6/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_6/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_6/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_6/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_6/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_6/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_6/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_6/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_6/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_6/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_6/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_6/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_6/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_6/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_6/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_6/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_6/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_6/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_6/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_6/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_6/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_6/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_6/TensordotReshapeImodel/transformer_block/multi_head_att/dense_6/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_6/Tensordot?
-model/transformer_block/multi_head_att/MatMulBatchMatMulV2?model/transformer_block/multi_head_att/dense/Tensordot:output:0Amodel/transformer_block/multi_head_att/dense_3/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2/
-model/transformer_block/multi_head_att/MatMul?
0model/transformer_block/multi_head_att/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A22
0model/transformer_block/multi_head_att/truediv/y?
.model/transformer_block/multi_head_att/truedivRealDiv6model/transformer_block/multi_head_att/MatMul:output:09model/transformer_block/multi_head_att/truediv/y:output:0*
T0*-
_output_shapes
:???????????20
.model/transformer_block/multi_head_att/truediv?
.model/transformer_block/multi_head_att/SoftmaxSoftmax2model/transformer_block/multi_head_att/truediv:z:0*
T0*-
_output_shapes
:???????????20
.model/transformer_block/multi_head_att/Softmax?
/model/transformer_block/multi_head_att/MatMul_1BatchMatMulV28model/transformer_block/multi_head_att/Softmax:softmax:0Amodel/transformer_block/multi_head_att/dense_6/Tensordot:output:0*
T0*,
_output_shapes
:??????????(21
/model/transformer_block/multi_head_att/MatMul_1?
Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_1/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_1/Tensordot/free?
>model/transformer_block/multi_head_att/dense_1/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_1/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_1/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_1/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_1/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_1/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_1/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_1/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_1/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_1/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_1/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_1/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_1/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_1/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_1/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_1/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_1/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_1/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_1/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_1/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_1/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_1/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_1/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_1/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_1/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_1/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_1/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_1/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_1/TensordotReshapeImodel/transformer_block/multi_head_att/dense_1/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_1/Tensordot?
Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_4/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_4/Tensordot/free?
>model/transformer_block/multi_head_att/dense_4/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_4/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_4/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_4/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_4/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_4/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_4/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_4/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_4/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_4/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_4/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_4/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_4/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_4/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_4/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_4/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_4/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_4/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_4/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_4/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_4/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_4/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_4/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_4/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_4/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_4/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_4/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_4/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_4/TensordotReshapeImodel/transformer_block/multi_head_att/dense_4/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_4/Tensordot?
Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_7_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_7/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_7/Tensordot/free?
>model/transformer_block/multi_head_att/dense_7/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_7/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_7/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_7/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_7/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_7/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_7/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_7/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_7/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_7/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_7/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_7/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_7/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_7/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_7/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_7/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_7/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_7/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_7/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_7/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_7/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_7/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_7/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_7/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_7/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_7/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_7/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_7/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_7/TensordotReshapeImodel/transformer_block/multi_head_att/dense_7/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_7/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_7/Tensordot?
/model/transformer_block/multi_head_att/MatMul_2BatchMatMulV2Amodel/transformer_block/multi_head_att/dense_1/Tensordot:output:0Amodel/transformer_block/multi_head_att/dense_4/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(21
/model/transformer_block/multi_head_att/MatMul_2?
2model/transformer_block/multi_head_att/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A24
2model/transformer_block/multi_head_att/truediv_1/y?
0model/transformer_block/multi_head_att/truediv_1RealDiv8model/transformer_block/multi_head_att/MatMul_2:output:0;model/transformer_block/multi_head_att/truediv_1/y:output:0*
T0*-
_output_shapes
:???????????22
0model/transformer_block/multi_head_att/truediv_1?
0model/transformer_block/multi_head_att/Softmax_1Softmax4model/transformer_block/multi_head_att/truediv_1:z:0*
T0*-
_output_shapes
:???????????22
0model/transformer_block/multi_head_att/Softmax_1?
/model/transformer_block/multi_head_att/MatMul_3BatchMatMulV2:model/transformer_block/multi_head_att/Softmax_1:softmax:0Amodel/transformer_block/multi_head_att/dense_7/Tensordot:output:0*
T0*,
_output_shapes
:??????????(21
/model/transformer_block/multi_head_att/MatMul_3?
Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_2/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_2/Tensordot/free?
>model/transformer_block/multi_head_att/dense_2/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_2/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_2/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_2/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_2/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_2/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_2/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_2/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_2/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_2/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_2/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_2/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_2/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_2/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_2/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_2/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_2/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_2/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_2/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_2/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_2/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_2/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_2/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_2/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_2/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_2/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_2/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_2/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_2/TensordotReshapeImodel/transformer_block/multi_head_att/dense_2/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_2/Tensordot?
Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_5_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_5/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_5/Tensordot/free?
>model/transformer_block/multi_head_att/dense_5/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_5/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_5/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_5/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_5/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_5/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_5/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_5/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_5/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_5/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_5/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_5/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_5/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_5/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_5/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_5/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_5/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_5/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_5/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_5/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_5/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_5/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_5/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_5/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_5/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_5/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_5/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_5/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_5/TensordotReshapeImodel/transformer_block/multi_head_att/dense_5/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_5/Tensordot?
Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_8/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_8/Tensordot/free?
>model/transformer_block/multi_head_att/dense_8/Tensordot/ShapeShape8model/w2v_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_8/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_8/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_8/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_8/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_8/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_8/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_8/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_8/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_8/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_8/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_8/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_8/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_8/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_8/Tensordot/transpose	Transpose8model/w2v_embedding/embedding_lookup/Identity_1:output:0Hmodel/transformer_block/multi_head_att/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2D
Bmodel/transformer_block/multi_head_att/dense_8/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_8/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_8/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_8/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_8/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_8/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_8/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_8/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_8/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_8/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_8/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_8/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_8/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_8/TensordotReshapeImodel/transformer_block/multi_head_att/dense_8/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_8/Tensordot?
/model/transformer_block/multi_head_att/MatMul_4BatchMatMulV2Amodel/transformer_block/multi_head_att/dense_2/Tensordot:output:0Amodel/transformer_block/multi_head_att/dense_5/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(21
/model/transformer_block/multi_head_att/MatMul_4?
2model/transformer_block/multi_head_att/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A24
2model/transformer_block/multi_head_att/truediv_2/y?
0model/transformer_block/multi_head_att/truediv_2RealDiv8model/transformer_block/multi_head_att/MatMul_4:output:0;model/transformer_block/multi_head_att/truediv_2/y:output:0*
T0*-
_output_shapes
:???????????22
0model/transformer_block/multi_head_att/truediv_2?
0model/transformer_block/multi_head_att/Softmax_2Softmax4model/transformer_block/multi_head_att/truediv_2:z:0*
T0*-
_output_shapes
:???????????22
0model/transformer_block/multi_head_att/Softmax_2?
/model/transformer_block/multi_head_att/MatMul_5BatchMatMulV2:model/transformer_block/multi_head_att/Softmax_2:softmax:0Amodel/transformer_block/multi_head_att/dense_8/Tensordot:output:0*
T0*,
_output_shapes
:??????????(21
/model/transformer_block/multi_head_att/MatMul_5?
2model/transformer_block/multi_head_att/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :24
2model/transformer_block/multi_head_att/concat/axis?
-model/transformer_block/multi_head_att/concatConcatV28model/transformer_block/multi_head_att/MatMul_1:output:08model/transformer_block/multi_head_att/MatMul_3:output:08model/transformer_block/multi_head_att/MatMul_5:output:0;model/transformer_block/multi_head_att/concat/axis:output:0*
N*
T0*,
_output_shapes
:??????????x2/
-model/transformer_block/multi_head_att/concat?
Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_multi_head_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

:x(*
dtype02I
Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp?
=model/transformer_block/multi_head_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block/multi_head_att/dense_9/Tensordot/axes?
=model/transformer_block/multi_head_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block/multi_head_att/dense_9/Tensordot/free?
>model/transformer_block/multi_head_att/dense_9/Tensordot/ShapeShape6model/transformer_block/multi_head_att/concat:output:0*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_9/Tensordot/Shape?
Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axis?
Amodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2GatherV2Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/free:output:0Omodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2?
Hmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/Shape:output:0Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/axes:output:0Qmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1?
>model/transformer_block/multi_head_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block/multi_head_att/dense_9/Tensordot/Const?
=model/transformer_block/multi_head_att/dense_9/Tensordot/ProdProdJmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2:output:0Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block/multi_head_att/dense_9/Tensordot/Prod?
@model/transformer_block/multi_head_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block/multi_head_att/dense_9/Tensordot/Const_1?
?model/transformer_block/multi_head_att/dense_9/Tensordot/Prod_1ProdLmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2_1:output:0Imodel/transformer_block/multi_head_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block/multi_head_att/dense_9/Tensordot/Prod_1?
Dmodel/transformer_block/multi_head_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block/multi_head_att/dense_9/Tensordot/concat/axis?
?model/transformer_block/multi_head_att/dense_9/Tensordot/concatConcatV2Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/free:output:0Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/axes:output:0Mmodel/transformer_block/multi_head_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block/multi_head_att/dense_9/Tensordot/concat?
>model/transformer_block/multi_head_att/dense_9/Tensordot/stackPackFmodel/transformer_block/multi_head_att/dense_9/Tensordot/Prod:output:0Hmodel/transformer_block/multi_head_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/multi_head_att/dense_9/Tensordot/stack?
Bmodel/transformer_block/multi_head_att/dense_9/Tensordot/transpose	Transpose6model/transformer_block/multi_head_att/concat:output:0Hmodel/transformer_block/multi_head_att/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????x2D
Bmodel/transformer_block/multi_head_att/dense_9/Tensordot/transpose?
@model/transformer_block/multi_head_att/dense_9/Tensordot/ReshapeReshapeFmodel/transformer_block/multi_head_att/dense_9/Tensordot/transpose:y:0Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block/multi_head_att/dense_9/Tensordot/Reshape?
?model/transformer_block/multi_head_att/dense_9/Tensordot/MatMulMatMulImodel/transformer_block/multi_head_att/dense_9/Tensordot/Reshape:output:0Omodel/transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2A
?model/transformer_block/multi_head_att/dense_9/Tensordot/MatMul?
@model/transformer_block/multi_head_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2B
@model/transformer_block/multi_head_att/dense_9/Tensordot/Const_2?
Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block/multi_head_att/dense_9/Tensordot/concat_1/axis?
Amodel/transformer_block/multi_head_att/dense_9/Tensordot/concat_1ConcatV2Jmodel/transformer_block/multi_head_att/dense_9/Tensordot/GatherV2:output:0Imodel/transformer_block/multi_head_att/dense_9/Tensordot/Const_2:output:0Omodel/transformer_block/multi_head_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block/multi_head_att/dense_9/Tensordot/concat_1?
8model/transformer_block/multi_head_att/dense_9/TensordotReshapeImodel/transformer_block/multi_head_att/dense_9/Tensordot/MatMul:product:0Jmodel/transformer_block/multi_head_att/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2:
8model/transformer_block/multi_head_att/dense_9/Tensordot?
(model/transformer_block/dropout/IdentityIdentityAmodel/transformer_block/multi_head_att/dense_9/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2*
(model/transformer_block/dropout/Identity?
model/transformer_block/addAddV28model/w2v_embedding/embedding_lookup/Identity_1:output:01model/transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:??????????(2
model/transformer_block/add?
Jmodel/transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jmodel/transformer_block/layer_normalization/moments/mean/reduction_indices?
8model/transformer_block/layer_normalization/moments/meanMeanmodel/transformer_block/add:z:0Smodel/transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2:
8model/transformer_block/layer_normalization/moments/mean?
@model/transformer_block/layer_normalization/moments/StopGradientStopGradientAmodel/transformer_block/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:??????????2B
@model/transformer_block/layer_normalization/moments/StopGradient?
Emodel/transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel/transformer_block/add:z:0Imodel/transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2G
Emodel/transformer_block/layer_normalization/moments/SquaredDifference?
Nmodel/transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block/layer_normalization/moments/variance/reduction_indices?
<model/transformer_block/layer_normalization/moments/varianceMeanImodel/transformer_block/layer_normalization/moments/SquaredDifference:z:0Wmodel/transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2>
<model/transformer_block/layer_normalization/moments/variance?
;model/transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52=
;model/transformer_block/layer_normalization/batchnorm/add/y?
9model/transformer_block/layer_normalization/batchnorm/addAddV2Emodel/transformer_block/layer_normalization/moments/variance:output:0Dmodel/transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2;
9model/transformer_block/layer_normalization/batchnorm/add?
;model/transformer_block/layer_normalization/batchnorm/RsqrtRsqrt=model/transformer_block/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2=
;model/transformer_block/layer_normalization/batchnorm/Rsqrt?
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpQmodel_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02J
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp?
9model/transformer_block/layer_normalization/batchnorm/mulMul?model/transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Pmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2;
9model/transformer_block/layer_normalization/batchnorm/mul?
;model/transformer_block/layer_normalization/batchnorm/mul_1Mulmodel/transformer_block/add:z:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2=
;model/transformer_block/layer_normalization/batchnorm/mul_1?
;model/transformer_block/layer_normalization/batchnorm/mul_2MulAmodel/transformer_block/layer_normalization/moments/mean:output:0=model/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2=
;model/transformer_block/layer_normalization/batchnorm/mul_2?
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpMmodel_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02F
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp?
9model/transformer_block/layer_normalization/batchnorm/subSubLmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0?model/transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2;
9model/transformer_block/layer_normalization/batchnorm/sub?
;model/transformer_block/layer_normalization/batchnorm/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/mul_1:z:0=model/transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2=
;model/transformer_block/layer_normalization/batchnorm/add_1?
Dmodel/transformer_block/sequential/dense_10/Tensordot/ReadVariableOpReadVariableOpMmodel_transformer_block_sequential_dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02F
Dmodel/transformer_block/sequential/dense_10/Tensordot/ReadVariableOp?
:model/transformer_block/sequential/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2<
:model/transformer_block/sequential/dense_10/Tensordot/axes?
:model/transformer_block/sequential/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2<
:model/transformer_block/sequential/dense_10/Tensordot/free?
;model/transformer_block/sequential/dense_10/Tensordot/ShapeShape?model/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2=
;model/transformer_block/sequential/dense_10/Tensordot/Shape?
Cmodel/transformer_block/sequential/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Cmodel/transformer_block/sequential/dense_10/Tensordot/GatherV2/axis?
>model/transformer_block/sequential/dense_10/Tensordot/GatherV2GatherV2Dmodel/transformer_block/sequential/dense_10/Tensordot/Shape:output:0Cmodel/transformer_block/sequential/dense_10/Tensordot/free:output:0Lmodel/transformer_block/sequential/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>model/transformer_block/sequential/dense_10/Tensordot/GatherV2?
Emodel/transformer_block/sequential/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Emodel/transformer_block/sequential/dense_10/Tensordot/GatherV2_1/axis?
@model/transformer_block/sequential/dense_10/Tensordot/GatherV2_1GatherV2Dmodel/transformer_block/sequential/dense_10/Tensordot/Shape:output:0Cmodel/transformer_block/sequential/dense_10/Tensordot/axes:output:0Nmodel/transformer_block/sequential/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2B
@model/transformer_block/sequential/dense_10/Tensordot/GatherV2_1?
;model/transformer_block/sequential/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model/transformer_block/sequential/dense_10/Tensordot/Const?
:model/transformer_block/sequential/dense_10/Tensordot/ProdProdGmodel/transformer_block/sequential/dense_10/Tensordot/GatherV2:output:0Dmodel/transformer_block/sequential/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2<
:model/transformer_block/sequential/dense_10/Tensordot/Prod?
=model/transformer_block/sequential/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=model/transformer_block/sequential/dense_10/Tensordot/Const_1?
<model/transformer_block/sequential/dense_10/Tensordot/Prod_1ProdImodel/transformer_block/sequential/dense_10/Tensordot/GatherV2_1:output:0Fmodel/transformer_block/sequential/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2>
<model/transformer_block/sequential/dense_10/Tensordot/Prod_1?
Amodel/transformer_block/sequential/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Amodel/transformer_block/sequential/dense_10/Tensordot/concat/axis?
<model/transformer_block/sequential/dense_10/Tensordot/concatConcatV2Cmodel/transformer_block/sequential/dense_10/Tensordot/free:output:0Cmodel/transformer_block/sequential/dense_10/Tensordot/axes:output:0Jmodel/transformer_block/sequential/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2>
<model/transformer_block/sequential/dense_10/Tensordot/concat?
;model/transformer_block/sequential/dense_10/Tensordot/stackPackCmodel/transformer_block/sequential/dense_10/Tensordot/Prod:output:0Emodel/transformer_block/sequential/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2=
;model/transformer_block/sequential/dense_10/Tensordot/stack?
?model/transformer_block/sequential/dense_10/Tensordot/transpose	Transpose?model/transformer_block/layer_normalization/batchnorm/add_1:z:0Emodel/transformer_block/sequential/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2A
?model/transformer_block/sequential/dense_10/Tensordot/transpose?
=model/transformer_block/sequential/dense_10/Tensordot/ReshapeReshapeCmodel/transformer_block/sequential/dense_10/Tensordot/transpose:y:0Dmodel/transformer_block/sequential/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2?
=model/transformer_block/sequential/dense_10/Tensordot/Reshape?
<model/transformer_block/sequential/dense_10/Tensordot/MatMulMatMulFmodel/transformer_block/sequential/dense_10/Tensordot/Reshape:output:0Lmodel/transformer_block/sequential/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2>
<model/transformer_block/sequential/dense_10/Tensordot/MatMul?
=model/transformer_block/sequential/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2?
=model/transformer_block/sequential/dense_10/Tensordot/Const_2?
Cmodel/transformer_block/sequential/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Cmodel/transformer_block/sequential/dense_10/Tensordot/concat_1/axis?
>model/transformer_block/sequential/dense_10/Tensordot/concat_1ConcatV2Gmodel/transformer_block/sequential/dense_10/Tensordot/GatherV2:output:0Fmodel/transformer_block/sequential/dense_10/Tensordot/Const_2:output:0Lmodel/transformer_block/sequential/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/sequential/dense_10/Tensordot/concat_1?
5model/transformer_block/sequential/dense_10/TensordotReshapeFmodel/transformer_block/sequential/dense_10/Tensordot/MatMul:product:0Gmodel/transformer_block/sequential/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 27
5model/transformer_block/sequential/dense_10/Tensordot?
Bmodel/transformer_block/sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOpKmodel_transformer_block_sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Bmodel/transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp?
3model/transformer_block/sequential/dense_10/BiasAddBiasAdd>model/transformer_block/sequential/dense_10/Tensordot:output:0Jmodel/transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 25
3model/transformer_block/sequential/dense_10/BiasAdd?
0model/transformer_block/sequential/dense_10/ReluRelu<model/transformer_block/sequential/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 22
0model/transformer_block/sequential/dense_10/Relu?
Dmodel/transformer_block/sequential/dense_11/Tensordot/ReadVariableOpReadVariableOpMmodel_transformer_block_sequential_dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02F
Dmodel/transformer_block/sequential/dense_11/Tensordot/ReadVariableOp?
:model/transformer_block/sequential/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2<
:model/transformer_block/sequential/dense_11/Tensordot/axes?
:model/transformer_block/sequential/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2<
:model/transformer_block/sequential/dense_11/Tensordot/free?
;model/transformer_block/sequential/dense_11/Tensordot/ShapeShape>model/transformer_block/sequential/dense_10/Relu:activations:0*
T0*
_output_shapes
:2=
;model/transformer_block/sequential/dense_11/Tensordot/Shape?
Cmodel/transformer_block/sequential/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Cmodel/transformer_block/sequential/dense_11/Tensordot/GatherV2/axis?
>model/transformer_block/sequential/dense_11/Tensordot/GatherV2GatherV2Dmodel/transformer_block/sequential/dense_11/Tensordot/Shape:output:0Cmodel/transformer_block/sequential/dense_11/Tensordot/free:output:0Lmodel/transformer_block/sequential/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>model/transformer_block/sequential/dense_11/Tensordot/GatherV2?
Emodel/transformer_block/sequential/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
Emodel/transformer_block/sequential/dense_11/Tensordot/GatherV2_1/axis?
@model/transformer_block/sequential/dense_11/Tensordot/GatherV2_1GatherV2Dmodel/transformer_block/sequential/dense_11/Tensordot/Shape:output:0Cmodel/transformer_block/sequential/dense_11/Tensordot/axes:output:0Nmodel/transformer_block/sequential/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2B
@model/transformer_block/sequential/dense_11/Tensordot/GatherV2_1?
;model/transformer_block/sequential/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model/transformer_block/sequential/dense_11/Tensordot/Const?
:model/transformer_block/sequential/dense_11/Tensordot/ProdProdGmodel/transformer_block/sequential/dense_11/Tensordot/GatherV2:output:0Dmodel/transformer_block/sequential/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2<
:model/transformer_block/sequential/dense_11/Tensordot/Prod?
=model/transformer_block/sequential/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
=model/transformer_block/sequential/dense_11/Tensordot/Const_1?
<model/transformer_block/sequential/dense_11/Tensordot/Prod_1ProdImodel/transformer_block/sequential/dense_11/Tensordot/GatherV2_1:output:0Fmodel/transformer_block/sequential/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2>
<model/transformer_block/sequential/dense_11/Tensordot/Prod_1?
Amodel/transformer_block/sequential/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Amodel/transformer_block/sequential/dense_11/Tensordot/concat/axis?
<model/transformer_block/sequential/dense_11/Tensordot/concatConcatV2Cmodel/transformer_block/sequential/dense_11/Tensordot/free:output:0Cmodel/transformer_block/sequential/dense_11/Tensordot/axes:output:0Jmodel/transformer_block/sequential/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2>
<model/transformer_block/sequential/dense_11/Tensordot/concat?
;model/transformer_block/sequential/dense_11/Tensordot/stackPackCmodel/transformer_block/sequential/dense_11/Tensordot/Prod:output:0Emodel/transformer_block/sequential/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2=
;model/transformer_block/sequential/dense_11/Tensordot/stack?
?model/transformer_block/sequential/dense_11/Tensordot/transpose	Transpose>model/transformer_block/sequential/dense_10/Relu:activations:0Emodel/transformer_block/sequential/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2A
?model/transformer_block/sequential/dense_11/Tensordot/transpose?
=model/transformer_block/sequential/dense_11/Tensordot/ReshapeReshapeCmodel/transformer_block/sequential/dense_11/Tensordot/transpose:y:0Dmodel/transformer_block/sequential/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2?
=model/transformer_block/sequential/dense_11/Tensordot/Reshape?
<model/transformer_block/sequential/dense_11/Tensordot/MatMulMatMulFmodel/transformer_block/sequential/dense_11/Tensordot/Reshape:output:0Lmodel/transformer_block/sequential/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2>
<model/transformer_block/sequential/dense_11/Tensordot/MatMul?
=model/transformer_block/sequential/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2?
=model/transformer_block/sequential/dense_11/Tensordot/Const_2?
Cmodel/transformer_block/sequential/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Cmodel/transformer_block/sequential/dense_11/Tensordot/concat_1/axis?
>model/transformer_block/sequential/dense_11/Tensordot/concat_1ConcatV2Gmodel/transformer_block/sequential/dense_11/Tensordot/GatherV2:output:0Fmodel/transformer_block/sequential/dense_11/Tensordot/Const_2:output:0Lmodel/transformer_block/sequential/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block/sequential/dense_11/Tensordot/concat_1?
5model/transformer_block/sequential/dense_11/TensordotReshapeFmodel/transformer_block/sequential/dense_11/Tensordot/MatMul:product:0Gmodel/transformer_block/sequential/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(27
5model/transformer_block/sequential/dense_11/Tensordot?
Bmodel/transformer_block/sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOpKmodel_transformer_block_sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02D
Bmodel/transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp?
3model/transformer_block/sequential/dense_11/BiasAddBiasAdd>model/transformer_block/sequential/dense_11/Tensordot:output:0Jmodel/transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(25
3model/transformer_block/sequential/dense_11/BiasAdd?
*model/transformer_block/dropout_1/IdentityIdentity<model/transformer_block/sequential/dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:??????????(2,
*model/transformer_block/dropout_1/Identity?
model/transformer_block/add_1AddV2?model/transformer_block/layer_normalization/batchnorm/add_1:z:03model/transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:??????????(2
model/transformer_block/add_1?
Lmodel/transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Lmodel/transformer_block/layer_normalization_1/moments/mean/reduction_indices?
:model/transformer_block/layer_normalization_1/moments/meanMean!model/transformer_block/add_1:z:0Umodel/transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2<
:model/transformer_block/layer_normalization_1/moments/mean?
Bmodel/transformer_block/layer_normalization_1/moments/StopGradientStopGradientCmodel/transformer_block/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:??????????2D
Bmodel/transformer_block/layer_normalization_1/moments/StopGradient?
Gmodel/transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!model/transformer_block/add_1:z:0Kmodel/transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2I
Gmodel/transformer_block/layer_normalization_1/moments/SquaredDifference?
Pmodel/transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel/transformer_block/layer_normalization_1/moments/variance/reduction_indices?
>model/transformer_block/layer_normalization_1/moments/varianceMeanKmodel/transformer_block/layer_normalization_1/moments/SquaredDifference:z:0Ymodel/transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2@
>model/transformer_block/layer_normalization_1/moments/variance?
=model/transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52?
=model/transformer_block/layer_normalization_1/batchnorm/add/y?
;model/transformer_block/layer_normalization_1/batchnorm/addAddV2Gmodel/transformer_block/layer_normalization_1/moments/variance:output:0Fmodel/transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2=
;model/transformer_block/layer_normalization_1/batchnorm/add?
=model/transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt?model/transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2?
=model/transformer_block/layer_normalization_1/batchnorm/Rsqrt?
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpSmodel_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02L
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp?
;model/transformer_block/layer_normalization_1/batchnorm/mulMulAmodel/transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Rmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2=
;model/transformer_block/layer_normalization_1/batchnorm/mul?
=model/transformer_block/layer_normalization_1/batchnorm/mul_1Mul!model/transformer_block/add_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2?
=model/transformer_block/layer_normalization_1/batchnorm/mul_1?
=model/transformer_block/layer_normalization_1/batchnorm/mul_2MulCmodel/transformer_block/layer_normalization_1/moments/mean:output:0?model/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2?
=model/transformer_block/layer_normalization_1/batchnorm/mul_2?
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpOmodel_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02H
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp?
;model/transformer_block/layer_normalization_1/batchnorm/subSubNmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Amodel/transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2=
;model/transformer_block/layer_normalization_1/batchnorm/sub?
=model/transformer_block/layer_normalization_1/batchnorm/add_1AddV2Amodel/transformer_block/layer_normalization_1/batchnorm/mul_1:z:0?model/transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2?
=model/transformer_block/layer_normalization_1/batchnorm/add_1?
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :27
5model/global_average_pooling1d/Mean/reduction_indices?
#model/global_average_pooling1d/MeanMeanAmodel/transformer_block/layer_normalization_1/batchnorm/add_1:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????(2%
#model/global_average_pooling1d/Mean?
model/dropout_2/IdentityIdentity,model/global_average_pooling1d/Mean:output:0*
T0*'
_output_shapes
:?????????(2
model/dropout_2/Identity?
$model/dense_12/MatMul/ReadVariableOpReadVariableOp-model_dense_12_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02&
$model/dense_12/MatMul/ReadVariableOp?
model/dense_12/MatMulMatMul!model/dropout_2/Identity:output:0,model/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_12/MatMul?
%model/dense_12/BiasAdd/ReadVariableOpReadVariableOp.model_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/dense_12/BiasAdd/ReadVariableOp?
model/dense_12/BiasAddBiasAddmodel/dense_12/MatMul:product:0-model/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_12/BiasAdd?
model/dense_12/ReluRelumodel/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense_12/Relu?
model/dropout_3/IdentityIdentity!model/dense_12/Relu:activations:0*
T0*'
_output_shapes
:?????????2
model/dropout_3/Identity?
$model/dense_13/MatMul/ReadVariableOpReadVariableOp-model_dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$model/dense_13/MatMul/ReadVariableOp?
model/dense_13/MatMulMatMul!model/dropout_3/Identity:output:0,model/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_13/MatMul?
%model/dense_13/BiasAdd/ReadVariableOpReadVariableOp.model_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/dense_13/BiasAdd/ReadVariableOp?
model/dense_13/BiasAddBiasAddmodel/dense_13/MatMul:product:0-model/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_13/BiasAdd?
model/dense_13/SoftmaxSoftmaxmodel/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense_13/Softmax{
IdentityIdentity model/dense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^model/dense_12/BiasAdd/ReadVariableOp%^model/dense_12/MatMul/ReadVariableOp&^model/dense_13/BiasAdd/ReadVariableOp%^model/dense_13/MatMul/ReadVariableOpE^model/transformer_block/layer_normalization/batchnorm/ReadVariableOpI^model/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpG^model/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpK^model/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpF^model/transformer_block/multi_head_att/dense/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpH^model/transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOpC^model/transformer_block/sequential/dense_10/BiasAdd/ReadVariableOpE^model/transformer_block/sequential/dense_10/Tensordot/ReadVariableOpC^model/transformer_block/sequential/dense_11/BiasAdd/ReadVariableOpE^model/transformer_block/sequential/dense_11/Tensordot/ReadVariableOp%^model/w2v_embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 2N
%model/dense_12/BiasAdd/ReadVariableOp%model/dense_12/BiasAdd/ReadVariableOp2L
$model/dense_12/MatMul/ReadVariableOp$model/dense_12/MatMul/ReadVariableOp2N
%model/dense_13/BiasAdd/ReadVariableOp%model/dense_13/BiasAdd/ReadVariableOp2L
$model/dense_13/MatMul/ReadVariableOp$model/dense_13/MatMul/ReadVariableOp2?
Dmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOpDmodel/transformer_block/layer_normalization/batchnorm/ReadVariableOp2?
Hmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpHmodel/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2?
Fmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpFmodel/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2?
Jmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpJmodel/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Emodel/transformer_block/multi_head_att/dense/Tensordot/ReadVariableOpEmodel/transformer_block/multi_head_att/dense/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_1/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_2/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_3/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_4/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_5/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_6/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_7/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_8/Tensordot/ReadVariableOp2?
Gmodel/transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOpGmodel/transformer_block/multi_head_att/dense_9/Tensordot/ReadVariableOp2?
Bmodel/transformer_block/sequential/dense_10/BiasAdd/ReadVariableOpBmodel/transformer_block/sequential/dense_10/BiasAdd/ReadVariableOp2?
Dmodel/transformer_block/sequential/dense_10/Tensordot/ReadVariableOpDmodel/transformer_block/sequential/dense_10/Tensordot/ReadVariableOp2?
Bmodel/transformer_block/sequential/dense_11/BiasAdd/ReadVariableOpBmodel/transformer_block/sequential/dense_11/BiasAdd/ReadVariableOp2?
Dmodel/transformer_block/sequential/dense_11/Tensordot/ReadVariableOpDmodel/transformer_block/sequential/dense_11/Tensordot/ReadVariableOp2L
$model/w2v_embedding/embedding_lookup$model/w2v_embedding/embedding_lookup:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5268
dense_10_input
dense_10_5257:( 
dense_10_5259: 
dense_11_5262: (
dense_11_5264:(
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCalldense_10_inputdense_10_5257dense_10_5259*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_51272"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5262dense_11_5264*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_51632"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:\ X
,
_output_shapes
:??????????(
(
_user_specified_namedense_10_input
?!
?
B__inference_dense_10_layer_call_and_return_conditional_losses_5127

inputs3
!tensordot_readvariableop_resource:( -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?

?
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_5323

inputs)
embedding_lookup_5317:
??(
identity??embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:??????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_5317Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/5317*,
_output_shapes
:??????????(*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/5317*,
_output_shapes
:??????????(2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????(2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?K
?
D__inference_sequential_layer_call_and_return_conditional_losses_8587

inputs<
*dense_10_tensordot_readvariableop_resource:( 6
(dense_10_biasadd_readvariableop_resource: <
*dense_11_tensordot_readvariableop_resource: (6
(dense_11_biasadd_readvariableop_resource:(
identity??dense_10/BiasAdd/ReadVariableOp?!dense_10/Tensordot/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/Tensordot/ReadVariableOp?
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes?
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/freej
dense_10/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape?
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axis?
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2?
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis?
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const?
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod?
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1?
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1?
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axis?
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat?
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack?
dense_10/Tensordot/transpose	Transposeinputs"dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2
dense_10/Tensordot/transpose?
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_10/Tensordot/Reshape?
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_10/Tensordot/MatMul?
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_2?
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axis?
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1?
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
dense_10/Tensordot?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2
dense_10/BiasAddx
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
dense_10/Relu?
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axes?
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/free
dense_11/Tensordot/ShapeShapedense_10/Relu:activations:0*
T0*
_output_shapes
:2
dense_11/Tensordot/Shape?
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axis?
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2?
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axis?
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2_1~
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const?
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod?
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1?
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1?
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axis?
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat?
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stack?
dense_11/Tensordot/transpose	Transposedense_10/Relu:activations:0"dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2
dense_11/Tensordot/transpose?
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_11/Tensordot/Reshape?
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_11/Tensordot/MatMul?
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_11/Tensordot/Const_2?
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axis?
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1?
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
dense_11/Tensordot?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2
dense_11/BiasAddy
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
??
?:
 __inference__traced_restore_9250
file_prefixG
3assignvariableop_word2_vec_w2v_embedding_embeddings:
??(4
"assignvariableop_1_dense_12_kernel:(.
 assignvariableop_2_dense_12_bias:4
"assignvariableop_3_dense_13_kernel:.
 assignvariableop_4_dense_13_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: S
Aassignvariableop_10_transformer_block_multi_head_att_dense_kernel:((U
Cassignvariableop_11_transformer_block_multi_head_att_dense_1_kernel:((U
Cassignvariableop_12_transformer_block_multi_head_att_dense_2_kernel:((U
Cassignvariableop_13_transformer_block_multi_head_att_dense_3_kernel:((U
Cassignvariableop_14_transformer_block_multi_head_att_dense_4_kernel:((U
Cassignvariableop_15_transformer_block_multi_head_att_dense_5_kernel:((U
Cassignvariableop_16_transformer_block_multi_head_att_dense_6_kernel:((U
Cassignvariableop_17_transformer_block_multi_head_att_dense_7_kernel:((U
Cassignvariableop_18_transformer_block_multi_head_att_dense_8_kernel:((U
Cassignvariableop_19_transformer_block_multi_head_att_dense_9_kernel:x(5
#assignvariableop_20_dense_10_kernel:( /
!assignvariableop_21_dense_10_bias: 5
#assignvariableop_22_dense_11_kernel: (/
!assignvariableop_23_dense_11_bias:(M
?assignvariableop_24_transformer_block_layer_normalization_gamma:(L
>assignvariableop_25_transformer_block_layer_normalization_beta:(O
Aassignvariableop_26_transformer_block_layer_normalization_1_gamma:(N
@assignvariableop_27_transformer_block_layer_normalization_1_beta:(#
assignvariableop_28_total: #
assignvariableop_29_count: %
assignvariableop_30_total_1: %
assignvariableop_31_count_1: Q
=assignvariableop_32_adam_word2_vec_w2v_embedding_embeddings_m:
??(<
*assignvariableop_33_adam_dense_12_kernel_m:(6
(assignvariableop_34_adam_dense_12_bias_m:<
*assignvariableop_35_adam_dense_13_kernel_m:6
(assignvariableop_36_adam_dense_13_bias_m:Z
Hassignvariableop_37_adam_transformer_block_multi_head_att_dense_kernel_m:((\
Jassignvariableop_38_adam_transformer_block_multi_head_att_dense_1_kernel_m:((\
Jassignvariableop_39_adam_transformer_block_multi_head_att_dense_2_kernel_m:((\
Jassignvariableop_40_adam_transformer_block_multi_head_att_dense_3_kernel_m:((\
Jassignvariableop_41_adam_transformer_block_multi_head_att_dense_4_kernel_m:((\
Jassignvariableop_42_adam_transformer_block_multi_head_att_dense_5_kernel_m:((\
Jassignvariableop_43_adam_transformer_block_multi_head_att_dense_6_kernel_m:((\
Jassignvariableop_44_adam_transformer_block_multi_head_att_dense_7_kernel_m:((\
Jassignvariableop_45_adam_transformer_block_multi_head_att_dense_8_kernel_m:((\
Jassignvariableop_46_adam_transformer_block_multi_head_att_dense_9_kernel_m:x(<
*assignvariableop_47_adam_dense_10_kernel_m:( 6
(assignvariableop_48_adam_dense_10_bias_m: <
*assignvariableop_49_adam_dense_11_kernel_m: (6
(assignvariableop_50_adam_dense_11_bias_m:(T
Fassignvariableop_51_adam_transformer_block_layer_normalization_gamma_m:(S
Eassignvariableop_52_adam_transformer_block_layer_normalization_beta_m:(V
Hassignvariableop_53_adam_transformer_block_layer_normalization_1_gamma_m:(U
Gassignvariableop_54_adam_transformer_block_layer_normalization_1_beta_m:(Q
=assignvariableop_55_adam_word2_vec_w2v_embedding_embeddings_v:
??(<
*assignvariableop_56_adam_dense_12_kernel_v:(6
(assignvariableop_57_adam_dense_12_bias_v:<
*assignvariableop_58_adam_dense_13_kernel_v:6
(assignvariableop_59_adam_dense_13_bias_v:Z
Hassignvariableop_60_adam_transformer_block_multi_head_att_dense_kernel_v:((\
Jassignvariableop_61_adam_transformer_block_multi_head_att_dense_1_kernel_v:((\
Jassignvariableop_62_adam_transformer_block_multi_head_att_dense_2_kernel_v:((\
Jassignvariableop_63_adam_transformer_block_multi_head_att_dense_3_kernel_v:((\
Jassignvariableop_64_adam_transformer_block_multi_head_att_dense_4_kernel_v:((\
Jassignvariableop_65_adam_transformer_block_multi_head_att_dense_5_kernel_v:((\
Jassignvariableop_66_adam_transformer_block_multi_head_att_dense_6_kernel_v:((\
Jassignvariableop_67_adam_transformer_block_multi_head_att_dense_7_kernel_v:((\
Jassignvariableop_68_adam_transformer_block_multi_head_att_dense_8_kernel_v:((\
Jassignvariableop_69_adam_transformer_block_multi_head_att_dense_9_kernel_v:x(<
*assignvariableop_70_adam_dense_10_kernel_v:( 6
(assignvariableop_71_adam_dense_10_bias_v: <
*assignvariableop_72_adam_dense_11_kernel_v: (6
(assignvariableop_73_adam_dense_11_bias_v:(T
Fassignvariableop_74_adam_transformer_block_layer_normalization_gamma_v:(S
Eassignvariableop_75_adam_transformer_block_layer_normalization_beta_v:(V
Hassignvariableop_76_adam_transformer_block_layer_normalization_1_gamma_v:(U
Gassignvariableop_77_adam_transformer_block_layer_normalization_1_beta_v:(
identity_79??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_8?AssignVariableOp_9?*
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*?)
value?)B?)OB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*?
value?B?OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*]
dtypesS
Q2O	2
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
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_12_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_12_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_13_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_13_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpAassignvariableop_10_transformer_block_multi_head_att_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_transformer_block_multi_head_att_dense_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpCassignvariableop_12_transformer_block_multi_head_att_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpCassignvariableop_13_transformer_block_multi_head_att_dense_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpCassignvariableop_14_transformer_block_multi_head_att_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpCassignvariableop_15_transformer_block_multi_head_att_dense_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpCassignvariableop_16_transformer_block_multi_head_att_dense_6_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpCassignvariableop_17_transformer_block_multi_head_att_dense_7_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpCassignvariableop_18_transformer_block_multi_head_att_dense_8_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpCassignvariableop_19_transformer_block_multi_head_att_dense_9_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp?assignvariableop_24_transformer_block_layer_normalization_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp>assignvariableop_25_transformer_block_layer_normalization_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpAassignvariableop_26_transformer_block_layer_normalization_1_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp@assignvariableop_27_transformer_block_layer_normalization_1_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp=assignvariableop_32_adam_word2_vec_w2v_embedding_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_12_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_12_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_13_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_13_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpHassignvariableop_37_adam_transformer_block_multi_head_att_dense_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpJassignvariableop_38_adam_transformer_block_multi_head_att_dense_1_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpJassignvariableop_39_adam_transformer_block_multi_head_att_dense_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpJassignvariableop_40_adam_transformer_block_multi_head_att_dense_3_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpJassignvariableop_41_adam_transformer_block_multi_head_att_dense_4_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpJassignvariableop_42_adam_transformer_block_multi_head_att_dense_5_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpJassignvariableop_43_adam_transformer_block_multi_head_att_dense_6_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpJassignvariableop_44_adam_transformer_block_multi_head_att_dense_7_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpJassignvariableop_45_adam_transformer_block_multi_head_att_dense_8_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpJassignvariableop_46_adam_transformer_block_multi_head_att_dense_9_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_10_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_10_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_11_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_11_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpFassignvariableop_51_adam_transformer_block_layer_normalization_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpEassignvariableop_52_adam_transformer_block_layer_normalization_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpHassignvariableop_53_adam_transformer_block_layer_normalization_1_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpGassignvariableop_54_adam_transformer_block_layer_normalization_1_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp=assignvariableop_55_adam_word2_vec_w2v_embedding_embeddings_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_dense_12_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_dense_12_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_13_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_13_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpHassignvariableop_60_adam_transformer_block_multi_head_att_dense_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpJassignvariableop_61_adam_transformer_block_multi_head_att_dense_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpJassignvariableop_62_adam_transformer_block_multi_head_att_dense_2_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpJassignvariableop_63_adam_transformer_block_multi_head_att_dense_3_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpJassignvariableop_64_adam_transformer_block_multi_head_att_dense_4_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpJassignvariableop_65_adam_transformer_block_multi_head_att_dense_5_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpJassignvariableop_66_adam_transformer_block_multi_head_att_dense_6_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpJassignvariableop_67_adam_transformer_block_multi_head_att_dense_7_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpJassignvariableop_68_adam_transformer_block_multi_head_att_dense_8_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpJassignvariableop_69_adam_transformer_block_multi_head_att_dense_9_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_dense_10_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dense_10_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_11_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense_11_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpFassignvariableop_74_adam_transformer_block_layer_normalization_gamma_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpEassignvariableop_75_adam_transformer_block_layer_normalization_beta_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOpHassignvariableop_76_adam_transformer_block_layer_normalization_1_gamma_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOpGassignvariableop_77_adam_transformer_block_layer_normalization_1_beta_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_779
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_78Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_78f
Identity_79IdentityIdentity_78:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_79?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_79Identity_79:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
B__inference_dense_13_layer_call_and_return_conditional_losses_8521

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5819
input_1
unknown:
??(
	unknown_0:((
	unknown_1:((
	unknown_2:((
	unknown_3:((
	unknown_4:((
	unknown_5:((
	unknown_6:((
	unknown_7:((
	unknown_8:((
	unknown_9:x(

unknown_10:(

unknown_11:(

unknown_12:( 

unknown_13: 

unknown_14: (

unknown_15:(

unknown_16:(

unknown_17:(

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_57702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
S
7__inference_global_average_pooling1d_layer_call_fn_8431

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_52922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_8453

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?*
?
?__inference_model_layer_call_and_return_conditional_losses_6611
input_1&
w2v_embedding_6557:
??((
transformer_block_6560:(((
transformer_block_6562:(((
transformer_block_6564:(((
transformer_block_6566:(((
transformer_block_6568:(((
transformer_block_6570:(((
transformer_block_6572:(((
transformer_block_6574:(((
transformer_block_6576:(((
transformer_block_6578:x($
transformer_block_6580:($
transformer_block_6582:((
transformer_block_6584:( $
transformer_block_6586: (
transformer_block_6588: ($
transformer_block_6590:($
transformer_block_6592:($
transformer_block_6594:(
dense_12_6599:(
dense_12_6601:
dense_13_6605:
dense_13_6607:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?%w2v_embedding/StatefulPartitionedCall?
%w2v_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1w2v_embedding_6557*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_53232'
%w2v_embedding/StatefulPartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall.w2v_embedding/StatefulPartitionedCall:output:0transformer_block_6560transformer_block_6562transformer_block_6564transformer_block_6566transformer_block_6568transformer_block_6570transformer_block_6572transformer_block_6574transformer_block_6576transformer_block_6578transformer_block_6580transformer_block_6582transformer_block_6584transformer_block_6586transformer_block_6588transformer_block_6590transformer_block_6592transformer_block_6594*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_56762+
)transformer_block/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_57192*
(global_average_pooling1d/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_57262
dropout_2/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_12_6599dense_12_6601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_57392"
 dense_12/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_57502
dropout_3/PartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_13_6605dense_13_6607*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_57632"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall&^w2v_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2N
%w2v_embedding/StatefulPartitionedCall%w2v_embedding/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_5882

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????(2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????(*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????(2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????(2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????(2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_8441

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????(2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????(2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?!
?
B__inference_dense_10_layer_call_and_return_conditional_losses_8701

inputs3
!tensordot_readvariableop_resource:( -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_5750

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_dense_12_layer_call_and_return_conditional_losses_8474

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
? 
?
B__inference_dense_11_layer_call_and_return_conditional_losses_5163

inputs3
!tensordot_readvariableop_resource: (-
biasadd_readvariableop_resource:(
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_8657

inputs
unknown:( 
	unknown_0: 
	unknown_1: (
	unknown_2:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_51702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
0__inference_transformer_block_layer_call_fn_8414

inputs
unknown:((
	unknown_0:((
	unknown_1:((
	unknown_2:((
	unknown_3:((
	unknown_4:((
	unknown_5:((
	unknown_6:((
	unknown_7:((
	unknown_8:x(
	unknown_9:(

unknown_10:(

unknown_11:( 

unknown_12: 

unknown_13: (

unknown_14:(

unknown_15:(

unknown_16:(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_62972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????(: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
a
(__inference_dropout_2_layer_call_fn_8463

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_58822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
??
?
K__inference_transformer_block_layer_call_and_return_conditional_losses_5676

inputsH
6multi_head_att_dense_tensordot_readvariableop_resource:((J
8multi_head_att_dense_3_tensordot_readvariableop_resource:((J
8multi_head_att_dense_6_tensordot_readvariableop_resource:((J
8multi_head_att_dense_1_tensordot_readvariableop_resource:((J
8multi_head_att_dense_4_tensordot_readvariableop_resource:((J
8multi_head_att_dense_7_tensordot_readvariableop_resource:((J
8multi_head_att_dense_2_tensordot_readvariableop_resource:((J
8multi_head_att_dense_5_tensordot_readvariableop_resource:((J
8multi_head_att_dense_8_tensordot_readvariableop_resource:((J
8multi_head_att_dense_9_tensordot_readvariableop_resource:x(G
9layer_normalization_batchnorm_mul_readvariableop_resource:(C
5layer_normalization_batchnorm_readvariableop_resource:(G
5sequential_dense_10_tensordot_readvariableop_resource:( A
3sequential_dense_10_biasadd_readvariableop_resource: G
5sequential_dense_11_tensordot_readvariableop_resource: (A
3sequential_dense_11_biasadd_readvariableop_resource:(I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:(E
7layer_normalization_1_batchnorm_readvariableop_resource:(
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?-multi_head_att/dense/Tensordot/ReadVariableOp?/multi_head_att/dense_1/Tensordot/ReadVariableOp?/multi_head_att/dense_2/Tensordot/ReadVariableOp?/multi_head_att/dense_3/Tensordot/ReadVariableOp?/multi_head_att/dense_4/Tensordot/ReadVariableOp?/multi_head_att/dense_5/Tensordot/ReadVariableOp?/multi_head_att/dense_6/Tensordot/ReadVariableOp?/multi_head_att/dense_7/Tensordot/ReadVariableOp?/multi_head_att/dense_8/Tensordot/ReadVariableOp?/multi_head_att/dense_9/Tensordot/ReadVariableOp?*sequential/dense_10/BiasAdd/ReadVariableOp?,sequential/dense_10/Tensordot/ReadVariableOp?*sequential/dense_11/BiasAdd/ReadVariableOp?,sequential/dense_11/Tensordot/ReadVariableOpb
multi_head_att/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_att/Shape?
"multi_head_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"multi_head_att/strided_slice/stack?
$multi_head_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_1?
$multi_head_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_2?
multi_head_att/strided_sliceStridedSlicemulti_head_att/Shape:output:0+multi_head_att/strided_slice/stack:output:0-multi_head_att/strided_slice/stack_1:output:0-multi_head_att/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
multi_head_att/strided_slice?
-multi_head_att/dense/Tensordot/ReadVariableOpReadVariableOp6multi_head_att_dense_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02/
-multi_head_att/dense/Tensordot/ReadVariableOp?
#multi_head_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#multi_head_att/dense/Tensordot/axes?
#multi_head_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#multi_head_att/dense/Tensordot/free?
$multi_head_att/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/Shape?
,multi_head_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/GatherV2/axis?
'multi_head_att/dense/Tensordot/GatherV2GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/free:output:05multi_head_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/GatherV2?
.multi_head_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense/Tensordot/GatherV2_1/axis?
)multi_head_att/dense/Tensordot/GatherV2_1GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/axes:output:07multi_head_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense/Tensordot/GatherV2_1?
$multi_head_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$multi_head_att/dense/Tensordot/Const?
#multi_head_att/dense/Tensordot/ProdProd0multi_head_att/dense/Tensordot/GatherV2:output:0-multi_head_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#multi_head_att/dense/Tensordot/Prod?
&multi_head_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense/Tensordot/Const_1?
%multi_head_att/dense/Tensordot/Prod_1Prod2multi_head_att/dense/Tensordot/GatherV2_1:output:0/multi_head_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense/Tensordot/Prod_1?
*multi_head_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*multi_head_att/dense/Tensordot/concat/axis?
%multi_head_att/dense/Tensordot/concatConcatV2,multi_head_att/dense/Tensordot/free:output:0,multi_head_att/dense/Tensordot/axes:output:03multi_head_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%multi_head_att/dense/Tensordot/concat?
$multi_head_att/dense/Tensordot/stackPack,multi_head_att/dense/Tensordot/Prod:output:0.multi_head_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/stack?
(multi_head_att/dense/Tensordot/transpose	Transposeinputs.multi_head_att/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2*
(multi_head_att/dense/Tensordot/transpose?
&multi_head_att/dense/Tensordot/ReshapeReshape,multi_head_att/dense/Tensordot/transpose:y:0-multi_head_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&multi_head_att/dense/Tensordot/Reshape?
%multi_head_att/dense/Tensordot/MatMulMatMul/multi_head_att/dense/Tensordot/Reshape:output:05multi_head_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2'
%multi_head_att/dense/Tensordot/MatMul?
&multi_head_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2(
&multi_head_att/dense/Tensordot/Const_2?
,multi_head_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/concat_1/axis?
'multi_head_att/dense/Tensordot/concat_1ConcatV20multi_head_att/dense/Tensordot/GatherV2:output:0/multi_head_att/dense/Tensordot/Const_2:output:05multi_head_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/concat_1?
multi_head_att/dense/TensordotReshape/multi_head_att/dense/Tensordot/MatMul:product:00multi_head_att/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2 
multi_head_att/dense/Tensordot?
/multi_head_att/dense_3/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_3/Tensordot/ReadVariableOp?
%multi_head_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_3/Tensordot/axes?
%multi_head_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_3/Tensordot/free?
&multi_head_att/dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/Shape?
.multi_head_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/GatherV2/axis?
)multi_head_att/dense_3/Tensordot/GatherV2GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/free:output:07multi_head_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/GatherV2?
0multi_head_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_3/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_3/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/axes:output:09multi_head_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_3/Tensordot/GatherV2_1?
&multi_head_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_3/Tensordot/Const?
%multi_head_att/dense_3/Tensordot/ProdProd2multi_head_att/dense_3/Tensordot/GatherV2:output:0/multi_head_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_3/Tensordot/Prod?
(multi_head_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_3/Tensordot/Const_1?
'multi_head_att/dense_3/Tensordot/Prod_1Prod4multi_head_att/dense_3/Tensordot/GatherV2_1:output:01multi_head_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_3/Tensordot/Prod_1?
,multi_head_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_3/Tensordot/concat/axis?
'multi_head_att/dense_3/Tensordot/concatConcatV2.multi_head_att/dense_3/Tensordot/free:output:0.multi_head_att/dense_3/Tensordot/axes:output:05multi_head_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_3/Tensordot/concat?
&multi_head_att/dense_3/Tensordot/stackPack.multi_head_att/dense_3/Tensordot/Prod:output:00multi_head_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/stack?
*multi_head_att/dense_3/Tensordot/transpose	Transposeinputs0multi_head_att/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_3/Tensordot/transpose?
(multi_head_att/dense_3/Tensordot/ReshapeReshape.multi_head_att/dense_3/Tensordot/transpose:y:0/multi_head_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_3/Tensordot/Reshape?
'multi_head_att/dense_3/Tensordot/MatMulMatMul1multi_head_att/dense_3/Tensordot/Reshape:output:07multi_head_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_3/Tensordot/MatMul?
(multi_head_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_3/Tensordot/Const_2?
.multi_head_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/concat_1/axis?
)multi_head_att/dense_3/Tensordot/concat_1ConcatV22multi_head_att/dense_3/Tensordot/GatherV2:output:01multi_head_att/dense_3/Tensordot/Const_2:output:07multi_head_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/concat_1?
 multi_head_att/dense_3/TensordotReshape1multi_head_att/dense_3/Tensordot/MatMul:product:02multi_head_att/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_3/Tensordot?
/multi_head_att/dense_6/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_6/Tensordot/ReadVariableOp?
%multi_head_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_6/Tensordot/axes?
%multi_head_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_6/Tensordot/free?
&multi_head_att/dense_6/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/Shape?
.multi_head_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/GatherV2/axis?
)multi_head_att/dense_6/Tensordot/GatherV2GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/free:output:07multi_head_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/GatherV2?
0multi_head_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_6/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_6/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/axes:output:09multi_head_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_6/Tensordot/GatherV2_1?
&multi_head_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_6/Tensordot/Const?
%multi_head_att/dense_6/Tensordot/ProdProd2multi_head_att/dense_6/Tensordot/GatherV2:output:0/multi_head_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_6/Tensordot/Prod?
(multi_head_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_6/Tensordot/Const_1?
'multi_head_att/dense_6/Tensordot/Prod_1Prod4multi_head_att/dense_6/Tensordot/GatherV2_1:output:01multi_head_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_6/Tensordot/Prod_1?
,multi_head_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_6/Tensordot/concat/axis?
'multi_head_att/dense_6/Tensordot/concatConcatV2.multi_head_att/dense_6/Tensordot/free:output:0.multi_head_att/dense_6/Tensordot/axes:output:05multi_head_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_6/Tensordot/concat?
&multi_head_att/dense_6/Tensordot/stackPack.multi_head_att/dense_6/Tensordot/Prod:output:00multi_head_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/stack?
*multi_head_att/dense_6/Tensordot/transpose	Transposeinputs0multi_head_att/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_6/Tensordot/transpose?
(multi_head_att/dense_6/Tensordot/ReshapeReshape.multi_head_att/dense_6/Tensordot/transpose:y:0/multi_head_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_6/Tensordot/Reshape?
'multi_head_att/dense_6/Tensordot/MatMulMatMul1multi_head_att/dense_6/Tensordot/Reshape:output:07multi_head_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_6/Tensordot/MatMul?
(multi_head_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_6/Tensordot/Const_2?
.multi_head_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/concat_1/axis?
)multi_head_att/dense_6/Tensordot/concat_1ConcatV22multi_head_att/dense_6/Tensordot/GatherV2:output:01multi_head_att/dense_6/Tensordot/Const_2:output:07multi_head_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/concat_1?
 multi_head_att/dense_6/TensordotReshape1multi_head_att/dense_6/Tensordot/MatMul:product:02multi_head_att/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_6/Tensordot?
multi_head_att/MatMulBatchMatMulV2'multi_head_att/dense/Tensordot:output:0)multi_head_att/dense_3/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMuly
multi_head_att/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv/y?
multi_head_att/truedivRealDivmulti_head_att/MatMul:output:0!multi_head_att/truediv/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv?
multi_head_att/SoftmaxSoftmaxmulti_head_att/truediv:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax?
multi_head_att/MatMul_1BatchMatMulV2 multi_head_att/Softmax:softmax:0)multi_head_att/dense_6/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_1?
/multi_head_att/dense_1/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_1/Tensordot/ReadVariableOp?
%multi_head_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_1/Tensordot/axes?
%multi_head_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_1/Tensordot/free?
&multi_head_att/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/Shape?
.multi_head_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/GatherV2/axis?
)multi_head_att/dense_1/Tensordot/GatherV2GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/free:output:07multi_head_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/GatherV2?
0multi_head_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_1/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_1/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/axes:output:09multi_head_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_1/Tensordot/GatherV2_1?
&multi_head_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_1/Tensordot/Const?
%multi_head_att/dense_1/Tensordot/ProdProd2multi_head_att/dense_1/Tensordot/GatherV2:output:0/multi_head_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_1/Tensordot/Prod?
(multi_head_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_1/Tensordot/Const_1?
'multi_head_att/dense_1/Tensordot/Prod_1Prod4multi_head_att/dense_1/Tensordot/GatherV2_1:output:01multi_head_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_1/Tensordot/Prod_1?
,multi_head_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_1/Tensordot/concat/axis?
'multi_head_att/dense_1/Tensordot/concatConcatV2.multi_head_att/dense_1/Tensordot/free:output:0.multi_head_att/dense_1/Tensordot/axes:output:05multi_head_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_1/Tensordot/concat?
&multi_head_att/dense_1/Tensordot/stackPack.multi_head_att/dense_1/Tensordot/Prod:output:00multi_head_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/stack?
*multi_head_att/dense_1/Tensordot/transpose	Transposeinputs0multi_head_att/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_1/Tensordot/transpose?
(multi_head_att/dense_1/Tensordot/ReshapeReshape.multi_head_att/dense_1/Tensordot/transpose:y:0/multi_head_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_1/Tensordot/Reshape?
'multi_head_att/dense_1/Tensordot/MatMulMatMul1multi_head_att/dense_1/Tensordot/Reshape:output:07multi_head_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_1/Tensordot/MatMul?
(multi_head_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_1/Tensordot/Const_2?
.multi_head_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/concat_1/axis?
)multi_head_att/dense_1/Tensordot/concat_1ConcatV22multi_head_att/dense_1/Tensordot/GatherV2:output:01multi_head_att/dense_1/Tensordot/Const_2:output:07multi_head_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/concat_1?
 multi_head_att/dense_1/TensordotReshape1multi_head_att/dense_1/Tensordot/MatMul:product:02multi_head_att/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_1/Tensordot?
/multi_head_att/dense_4/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_4/Tensordot/ReadVariableOp?
%multi_head_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_4/Tensordot/axes?
%multi_head_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_4/Tensordot/free?
&multi_head_att/dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/Shape?
.multi_head_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/GatherV2/axis?
)multi_head_att/dense_4/Tensordot/GatherV2GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/free:output:07multi_head_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/GatherV2?
0multi_head_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_4/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_4/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/axes:output:09multi_head_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_4/Tensordot/GatherV2_1?
&multi_head_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_4/Tensordot/Const?
%multi_head_att/dense_4/Tensordot/ProdProd2multi_head_att/dense_4/Tensordot/GatherV2:output:0/multi_head_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_4/Tensordot/Prod?
(multi_head_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_4/Tensordot/Const_1?
'multi_head_att/dense_4/Tensordot/Prod_1Prod4multi_head_att/dense_4/Tensordot/GatherV2_1:output:01multi_head_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_4/Tensordot/Prod_1?
,multi_head_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_4/Tensordot/concat/axis?
'multi_head_att/dense_4/Tensordot/concatConcatV2.multi_head_att/dense_4/Tensordot/free:output:0.multi_head_att/dense_4/Tensordot/axes:output:05multi_head_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_4/Tensordot/concat?
&multi_head_att/dense_4/Tensordot/stackPack.multi_head_att/dense_4/Tensordot/Prod:output:00multi_head_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/stack?
*multi_head_att/dense_4/Tensordot/transpose	Transposeinputs0multi_head_att/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_4/Tensordot/transpose?
(multi_head_att/dense_4/Tensordot/ReshapeReshape.multi_head_att/dense_4/Tensordot/transpose:y:0/multi_head_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_4/Tensordot/Reshape?
'multi_head_att/dense_4/Tensordot/MatMulMatMul1multi_head_att/dense_4/Tensordot/Reshape:output:07multi_head_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_4/Tensordot/MatMul?
(multi_head_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_4/Tensordot/Const_2?
.multi_head_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/concat_1/axis?
)multi_head_att/dense_4/Tensordot/concat_1ConcatV22multi_head_att/dense_4/Tensordot/GatherV2:output:01multi_head_att/dense_4/Tensordot/Const_2:output:07multi_head_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/concat_1?
 multi_head_att/dense_4/TensordotReshape1multi_head_att/dense_4/Tensordot/MatMul:product:02multi_head_att/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_4/Tensordot?
/multi_head_att/dense_7/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_7_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_7/Tensordot/ReadVariableOp?
%multi_head_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_7/Tensordot/axes?
%multi_head_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_7/Tensordot/free?
&multi_head_att/dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/Shape?
.multi_head_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/GatherV2/axis?
)multi_head_att/dense_7/Tensordot/GatherV2GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/free:output:07multi_head_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/GatherV2?
0multi_head_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_7/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_7/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/axes:output:09multi_head_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_7/Tensordot/GatherV2_1?
&multi_head_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_7/Tensordot/Const?
%multi_head_att/dense_7/Tensordot/ProdProd2multi_head_att/dense_7/Tensordot/GatherV2:output:0/multi_head_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_7/Tensordot/Prod?
(multi_head_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_7/Tensordot/Const_1?
'multi_head_att/dense_7/Tensordot/Prod_1Prod4multi_head_att/dense_7/Tensordot/GatherV2_1:output:01multi_head_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_7/Tensordot/Prod_1?
,multi_head_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_7/Tensordot/concat/axis?
'multi_head_att/dense_7/Tensordot/concatConcatV2.multi_head_att/dense_7/Tensordot/free:output:0.multi_head_att/dense_7/Tensordot/axes:output:05multi_head_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_7/Tensordot/concat?
&multi_head_att/dense_7/Tensordot/stackPack.multi_head_att/dense_7/Tensordot/Prod:output:00multi_head_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/stack?
*multi_head_att/dense_7/Tensordot/transpose	Transposeinputs0multi_head_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_7/Tensordot/transpose?
(multi_head_att/dense_7/Tensordot/ReshapeReshape.multi_head_att/dense_7/Tensordot/transpose:y:0/multi_head_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_7/Tensordot/Reshape?
'multi_head_att/dense_7/Tensordot/MatMulMatMul1multi_head_att/dense_7/Tensordot/Reshape:output:07multi_head_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_7/Tensordot/MatMul?
(multi_head_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_7/Tensordot/Const_2?
.multi_head_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/concat_1/axis?
)multi_head_att/dense_7/Tensordot/concat_1ConcatV22multi_head_att/dense_7/Tensordot/GatherV2:output:01multi_head_att/dense_7/Tensordot/Const_2:output:07multi_head_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/concat_1?
 multi_head_att/dense_7/TensordotReshape1multi_head_att/dense_7/Tensordot/MatMul:product:02multi_head_att/dense_7/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_7/Tensordot?
multi_head_att/MatMul_2BatchMatMulV2)multi_head_att/dense_1/Tensordot:output:0)multi_head_att/dense_4/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_2}
multi_head_att/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_1/y?
multi_head_att/truediv_1RealDiv multi_head_att/MatMul_2:output:0#multi_head_att/truediv_1/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_1?
multi_head_att/Softmax_1Softmaxmulti_head_att/truediv_1:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_1?
multi_head_att/MatMul_3BatchMatMulV2"multi_head_att/Softmax_1:softmax:0)multi_head_att/dense_7/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_3?
/multi_head_att/dense_2/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_2/Tensordot/ReadVariableOp?
%multi_head_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_2/Tensordot/axes?
%multi_head_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_2/Tensordot/free?
&multi_head_att/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/Shape?
.multi_head_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/GatherV2/axis?
)multi_head_att/dense_2/Tensordot/GatherV2GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/free:output:07multi_head_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/GatherV2?
0multi_head_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_2/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_2/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/axes:output:09multi_head_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_2/Tensordot/GatherV2_1?
&multi_head_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_2/Tensordot/Const?
%multi_head_att/dense_2/Tensordot/ProdProd2multi_head_att/dense_2/Tensordot/GatherV2:output:0/multi_head_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_2/Tensordot/Prod?
(multi_head_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_2/Tensordot/Const_1?
'multi_head_att/dense_2/Tensordot/Prod_1Prod4multi_head_att/dense_2/Tensordot/GatherV2_1:output:01multi_head_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_2/Tensordot/Prod_1?
,multi_head_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_2/Tensordot/concat/axis?
'multi_head_att/dense_2/Tensordot/concatConcatV2.multi_head_att/dense_2/Tensordot/free:output:0.multi_head_att/dense_2/Tensordot/axes:output:05multi_head_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_2/Tensordot/concat?
&multi_head_att/dense_2/Tensordot/stackPack.multi_head_att/dense_2/Tensordot/Prod:output:00multi_head_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/stack?
*multi_head_att/dense_2/Tensordot/transpose	Transposeinputs0multi_head_att/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_2/Tensordot/transpose?
(multi_head_att/dense_2/Tensordot/ReshapeReshape.multi_head_att/dense_2/Tensordot/transpose:y:0/multi_head_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_2/Tensordot/Reshape?
'multi_head_att/dense_2/Tensordot/MatMulMatMul1multi_head_att/dense_2/Tensordot/Reshape:output:07multi_head_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_2/Tensordot/MatMul?
(multi_head_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_2/Tensordot/Const_2?
.multi_head_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/concat_1/axis?
)multi_head_att/dense_2/Tensordot/concat_1ConcatV22multi_head_att/dense_2/Tensordot/GatherV2:output:01multi_head_att/dense_2/Tensordot/Const_2:output:07multi_head_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/concat_1?
 multi_head_att/dense_2/TensordotReshape1multi_head_att/dense_2/Tensordot/MatMul:product:02multi_head_att/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_2/Tensordot?
/multi_head_att/dense_5/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_5_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_5/Tensordot/ReadVariableOp?
%multi_head_att/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_5/Tensordot/axes?
%multi_head_att/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_5/Tensordot/free?
&multi_head_att/dense_5/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/Shape?
.multi_head_att/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/GatherV2/axis?
)multi_head_att/dense_5/Tensordot/GatherV2GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/free:output:07multi_head_att/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/GatherV2?
0multi_head_att/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_5/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_5/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/axes:output:09multi_head_att/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_5/Tensordot/GatherV2_1?
&multi_head_att/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_5/Tensordot/Const?
%multi_head_att/dense_5/Tensordot/ProdProd2multi_head_att/dense_5/Tensordot/GatherV2:output:0/multi_head_att/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_5/Tensordot/Prod?
(multi_head_att/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_5/Tensordot/Const_1?
'multi_head_att/dense_5/Tensordot/Prod_1Prod4multi_head_att/dense_5/Tensordot/GatherV2_1:output:01multi_head_att/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_5/Tensordot/Prod_1?
,multi_head_att/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_5/Tensordot/concat/axis?
'multi_head_att/dense_5/Tensordot/concatConcatV2.multi_head_att/dense_5/Tensordot/free:output:0.multi_head_att/dense_5/Tensordot/axes:output:05multi_head_att/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_5/Tensordot/concat?
&multi_head_att/dense_5/Tensordot/stackPack.multi_head_att/dense_5/Tensordot/Prod:output:00multi_head_att/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/stack?
*multi_head_att/dense_5/Tensordot/transpose	Transposeinputs0multi_head_att/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_5/Tensordot/transpose?
(multi_head_att/dense_5/Tensordot/ReshapeReshape.multi_head_att/dense_5/Tensordot/transpose:y:0/multi_head_att/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_5/Tensordot/Reshape?
'multi_head_att/dense_5/Tensordot/MatMulMatMul1multi_head_att/dense_5/Tensordot/Reshape:output:07multi_head_att/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_5/Tensordot/MatMul?
(multi_head_att/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_5/Tensordot/Const_2?
.multi_head_att/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/concat_1/axis?
)multi_head_att/dense_5/Tensordot/concat_1ConcatV22multi_head_att/dense_5/Tensordot/GatherV2:output:01multi_head_att/dense_5/Tensordot/Const_2:output:07multi_head_att/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/concat_1?
 multi_head_att/dense_5/TensordotReshape1multi_head_att/dense_5/Tensordot/MatMul:product:02multi_head_att/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_5/Tensordot?
/multi_head_att/dense_8/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_8/Tensordot/ReadVariableOp?
%multi_head_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_8/Tensordot/axes?
%multi_head_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_8/Tensordot/free?
&multi_head_att/dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/Shape?
.multi_head_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/GatherV2/axis?
)multi_head_att/dense_8/Tensordot/GatherV2GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/free:output:07multi_head_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/GatherV2?
0multi_head_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_8/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_8/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/axes:output:09multi_head_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_8/Tensordot/GatherV2_1?
&multi_head_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_8/Tensordot/Const?
%multi_head_att/dense_8/Tensordot/ProdProd2multi_head_att/dense_8/Tensordot/GatherV2:output:0/multi_head_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_8/Tensordot/Prod?
(multi_head_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_8/Tensordot/Const_1?
'multi_head_att/dense_8/Tensordot/Prod_1Prod4multi_head_att/dense_8/Tensordot/GatherV2_1:output:01multi_head_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_8/Tensordot/Prod_1?
,multi_head_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_8/Tensordot/concat/axis?
'multi_head_att/dense_8/Tensordot/concatConcatV2.multi_head_att/dense_8/Tensordot/free:output:0.multi_head_att/dense_8/Tensordot/axes:output:05multi_head_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_8/Tensordot/concat?
&multi_head_att/dense_8/Tensordot/stackPack.multi_head_att/dense_8/Tensordot/Prod:output:00multi_head_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/stack?
*multi_head_att/dense_8/Tensordot/transpose	Transposeinputs0multi_head_att/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_8/Tensordot/transpose?
(multi_head_att/dense_8/Tensordot/ReshapeReshape.multi_head_att/dense_8/Tensordot/transpose:y:0/multi_head_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_8/Tensordot/Reshape?
'multi_head_att/dense_8/Tensordot/MatMulMatMul1multi_head_att/dense_8/Tensordot/Reshape:output:07multi_head_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_8/Tensordot/MatMul?
(multi_head_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_8/Tensordot/Const_2?
.multi_head_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/concat_1/axis?
)multi_head_att/dense_8/Tensordot/concat_1ConcatV22multi_head_att/dense_8/Tensordot/GatherV2:output:01multi_head_att/dense_8/Tensordot/Const_2:output:07multi_head_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/concat_1?
 multi_head_att/dense_8/TensordotReshape1multi_head_att/dense_8/Tensordot/MatMul:product:02multi_head_att/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_8/Tensordot?
multi_head_att/MatMul_4BatchMatMulV2)multi_head_att/dense_2/Tensordot:output:0)multi_head_att/dense_5/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_4}
multi_head_att/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_2/y?
multi_head_att/truediv_2RealDiv multi_head_att/MatMul_4:output:0#multi_head_att/truediv_2/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_2?
multi_head_att/Softmax_2Softmaxmulti_head_att/truediv_2:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_2?
multi_head_att/MatMul_5BatchMatMulV2"multi_head_att/Softmax_2:softmax:0)multi_head_att/dense_8/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_5z
multi_head_att/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
multi_head_att/concat/axis?
multi_head_att/concatConcatV2 multi_head_att/MatMul_1:output:0 multi_head_att/MatMul_3:output:0 multi_head_att/MatMul_5:output:0#multi_head_att/concat/axis:output:0*
N*
T0*,
_output_shapes
:??????????x2
multi_head_att/concat?
/multi_head_att/dense_9/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

:x(*
dtype021
/multi_head_att/dense_9/Tensordot/ReadVariableOp?
%multi_head_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_9/Tensordot/axes?
%multi_head_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_9/Tensordot/free?
&multi_head_att/dense_9/Tensordot/ShapeShapemulti_head_att/concat:output:0*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/Shape?
.multi_head_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/GatherV2/axis?
)multi_head_att/dense_9/Tensordot/GatherV2GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/free:output:07multi_head_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/GatherV2?
0multi_head_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_9/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_9/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/axes:output:09multi_head_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_9/Tensordot/GatherV2_1?
&multi_head_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_9/Tensordot/Const?
%multi_head_att/dense_9/Tensordot/ProdProd2multi_head_att/dense_9/Tensordot/GatherV2:output:0/multi_head_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_9/Tensordot/Prod?
(multi_head_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_9/Tensordot/Const_1?
'multi_head_att/dense_9/Tensordot/Prod_1Prod4multi_head_att/dense_9/Tensordot/GatherV2_1:output:01multi_head_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_9/Tensordot/Prod_1?
,multi_head_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_9/Tensordot/concat/axis?
'multi_head_att/dense_9/Tensordot/concatConcatV2.multi_head_att/dense_9/Tensordot/free:output:0.multi_head_att/dense_9/Tensordot/axes:output:05multi_head_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_9/Tensordot/concat?
&multi_head_att/dense_9/Tensordot/stackPack.multi_head_att/dense_9/Tensordot/Prod:output:00multi_head_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/stack?
*multi_head_att/dense_9/Tensordot/transpose	Transposemulti_head_att/concat:output:00multi_head_att/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????x2,
*multi_head_att/dense_9/Tensordot/transpose?
(multi_head_att/dense_9/Tensordot/ReshapeReshape.multi_head_att/dense_9/Tensordot/transpose:y:0/multi_head_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_9/Tensordot/Reshape?
'multi_head_att/dense_9/Tensordot/MatMulMatMul1multi_head_att/dense_9/Tensordot/Reshape:output:07multi_head_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_9/Tensordot/MatMul?
(multi_head_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_9/Tensordot/Const_2?
.multi_head_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/concat_1/axis?
)multi_head_att/dense_9/Tensordot/concat_1ConcatV22multi_head_att/dense_9/Tensordot/GatherV2:output:01multi_head_att/dense_9/Tensordot/Const_2:output:07multi_head_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/concat_1?
 multi_head_att/dense_9/TensordotReshape1multi_head_att/dense_9/Tensordot/MatMul:product:02multi_head_att/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_9/Tensordot?
dropout/IdentityIdentity)multi_head_att/dense_9/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
dropout/Identitym
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:??????????(2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:??????????2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/add_1?
,sequential/dense_10/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02.
,sequential/dense_10/Tensordot/ReadVariableOp?
"sequential/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_10/Tensordot/axes?
"sequential/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_10/Tensordot/free?
#sequential/dense_10/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/Shape?
+sequential/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/GatherV2/axis?
&sequential/dense_10/Tensordot/GatherV2GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/free:output:04sequential/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/GatherV2?
-sequential/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_10/Tensordot/GatherV2_1/axis?
(sequential/dense_10/Tensordot/GatherV2_1GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/axes:output:06sequential/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_10/Tensordot/GatherV2_1?
#sequential/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_10/Tensordot/Const?
"sequential/dense_10/Tensordot/ProdProd/sequential/dense_10/Tensordot/GatherV2:output:0,sequential/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_10/Tensordot/Prod?
%sequential/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_1?
$sequential/dense_10/Tensordot/Prod_1Prod1sequential/dense_10/Tensordot/GatherV2_1:output:0.sequential/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_10/Tensordot/Prod_1?
)sequential/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_10/Tensordot/concat/axis?
$sequential/dense_10/Tensordot/concatConcatV2+sequential/dense_10/Tensordot/free:output:0+sequential/dense_10/Tensordot/axes:output:02sequential/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_10/Tensordot/concat?
#sequential/dense_10/Tensordot/stackPack+sequential/dense_10/Tensordot/Prod:output:0-sequential/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/stack?
'sequential/dense_10/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0-sequential/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2)
'sequential/dense_10/Tensordot/transpose?
%sequential/dense_10/Tensordot/ReshapeReshape+sequential/dense_10/Tensordot/transpose:y:0,sequential/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_10/Tensordot/Reshape?
$sequential/dense_10/Tensordot/MatMulMatMul.sequential/dense_10/Tensordot/Reshape:output:04sequential/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$sequential/dense_10/Tensordot/MatMul?
%sequential/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_2?
+sequential/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/concat_1/axis?
&sequential/dense_10/Tensordot/concat_1ConcatV2/sequential/dense_10/Tensordot/GatherV2:output:0.sequential/dense_10/Tensordot/Const_2:output:04sequential/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/concat_1?
sequential/dense_10/TensordotReshape.sequential/dense_10/Tensordot/MatMul:product:0/sequential/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Tensordot?
*sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/dense_10/BiasAdd/ReadVariableOp?
sequential/dense_10/BiasAddBiasAdd&sequential/dense_10/Tensordot:output:02sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/BiasAdd?
sequential/dense_10/ReluRelu$sequential/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Relu?
,sequential/dense_11/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02.
,sequential/dense_11/Tensordot/ReadVariableOp?
"sequential/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_11/Tensordot/axes?
"sequential/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_11/Tensordot/free?
#sequential/dense_11/Tensordot/ShapeShape&sequential/dense_10/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/Shape?
+sequential/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/GatherV2/axis?
&sequential/dense_11/Tensordot/GatherV2GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/free:output:04sequential/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/GatherV2?
-sequential/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_11/Tensordot/GatherV2_1/axis?
(sequential/dense_11/Tensordot/GatherV2_1GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/axes:output:06sequential/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_11/Tensordot/GatherV2_1?
#sequential/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_11/Tensordot/Const?
"sequential/dense_11/Tensordot/ProdProd/sequential/dense_11/Tensordot/GatherV2:output:0,sequential/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_11/Tensordot/Prod?
%sequential/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_11/Tensordot/Const_1?
$sequential/dense_11/Tensordot/Prod_1Prod1sequential/dense_11/Tensordot/GatherV2_1:output:0.sequential/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_11/Tensordot/Prod_1?
)sequential/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_11/Tensordot/concat/axis?
$sequential/dense_11/Tensordot/concatConcatV2+sequential/dense_11/Tensordot/free:output:0+sequential/dense_11/Tensordot/axes:output:02sequential/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_11/Tensordot/concat?
#sequential/dense_11/Tensordot/stackPack+sequential/dense_11/Tensordot/Prod:output:0-sequential/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/stack?
'sequential/dense_11/Tensordot/transpose	Transpose&sequential/dense_10/Relu:activations:0-sequential/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2)
'sequential/dense_11/Tensordot/transpose?
%sequential/dense_11/Tensordot/ReshapeReshape+sequential/dense_11/Tensordot/transpose:y:0,sequential/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_11/Tensordot/Reshape?
$sequential/dense_11/Tensordot/MatMulMatMul.sequential/dense_11/Tensordot/Reshape:output:04sequential/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2&
$sequential/dense_11/Tensordot/MatMul?
%sequential/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2'
%sequential/dense_11/Tensordot/Const_2?
+sequential/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/concat_1/axis?
&sequential/dense_11/Tensordot/concat_1ConcatV2/sequential/dense_11/Tensordot/GatherV2:output:0.sequential/dense_11/Tensordot/Const_2:output:04sequential/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/concat_1?
sequential/dense_11/TensordotReshape.sequential/dense_11/Tensordot/MatMul:product:0/sequential/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/Tensordot?
*sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*sequential/dense_11/BiasAdd/ReadVariableOp?
sequential/dense_11/BiasAddBiasAdd&sequential/dense_11/Tensordot:output:02sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/BiasAdd?
dropout_1/IdentityIdentity$sequential/dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:??????????(2
dropout_1/Identity?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:??????????(2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:??????????2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp.^multi_head_att/dense/Tensordot/ReadVariableOp0^multi_head_att/dense_1/Tensordot/ReadVariableOp0^multi_head_att/dense_2/Tensordot/ReadVariableOp0^multi_head_att/dense_3/Tensordot/ReadVariableOp0^multi_head_att/dense_4/Tensordot/ReadVariableOp0^multi_head_att/dense_5/Tensordot/ReadVariableOp0^multi_head_att/dense_6/Tensordot/ReadVariableOp0^multi_head_att/dense_7/Tensordot/ReadVariableOp0^multi_head_att/dense_8/Tensordot/ReadVariableOp0^multi_head_att/dense_9/Tensordot/ReadVariableOp+^sequential/dense_10/BiasAdd/ReadVariableOp-^sequential/dense_10/Tensordot/ReadVariableOp+^sequential/dense_11/BiasAdd/ReadVariableOp-^sequential/dense_11/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????(: : : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2^
-multi_head_att/dense/Tensordot/ReadVariableOp-multi_head_att/dense/Tensordot/ReadVariableOp2b
/multi_head_att/dense_1/Tensordot/ReadVariableOp/multi_head_att/dense_1/Tensordot/ReadVariableOp2b
/multi_head_att/dense_2/Tensordot/ReadVariableOp/multi_head_att/dense_2/Tensordot/ReadVariableOp2b
/multi_head_att/dense_3/Tensordot/ReadVariableOp/multi_head_att/dense_3/Tensordot/ReadVariableOp2b
/multi_head_att/dense_4/Tensordot/ReadVariableOp/multi_head_att/dense_4/Tensordot/ReadVariableOp2b
/multi_head_att/dense_5/Tensordot/ReadVariableOp/multi_head_att/dense_5/Tensordot/ReadVariableOp2b
/multi_head_att/dense_6/Tensordot/ReadVariableOp/multi_head_att/dense_6/Tensordot/ReadVariableOp2b
/multi_head_att/dense_7/Tensordot/ReadVariableOp/multi_head_att/dense_7/Tensordot/ReadVariableOp2b
/multi_head_att/dense_8/Tensordot/ReadVariableOp/multi_head_att/dense_8/Tensordot/ReadVariableOp2b
/multi_head_att/dense_9/Tensordot/ReadVariableOp/multi_head_att/dense_9/Tensordot/ReadVariableOp2X
*sequential/dense_10/BiasAdd/ReadVariableOp*sequential/dense_10/BiasAdd/ReadVariableOp2\
,sequential/dense_10/Tensordot/ReadVariableOp,sequential/dense_10/Tensordot/ReadVariableOp2X
*sequential/dense_11/BiasAdd/ReadVariableOp*sequential/dense_11/BiasAdd/ReadVariableOp2\
,sequential/dense_11/Tensordot/ReadVariableOp,sequential/dense_11/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
'__inference_dense_13_layer_call_fn_8530

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_57632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_5170

inputs
dense_10_5128:( 
dense_10_5130: 
dense_11_5164: (
dense_11_5166:(
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_5128dense_10_5130*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_51272"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_5164dense_11_5166*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_51632"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????(: : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
,__inference_w2v_embedding_layer_call_fn_7620

inputs
unknown:
??(
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_53232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?	
?__inference_model_layer_call_and_return_conditional_losses_6454

inputs&
w2v_embedding_6400:
??((
transformer_block_6403:(((
transformer_block_6405:(((
transformer_block_6407:(((
transformer_block_6409:(((
transformer_block_6411:(((
transformer_block_6413:(((
transformer_block_6415:(((
transformer_block_6417:(((
transformer_block_6419:(((
transformer_block_6421:x($
transformer_block_6423:($
transformer_block_6425:((
transformer_block_6427:( $
transformer_block_6429: (
transformer_block_6431: ($
transformer_block_6433:($
transformer_block_6435:($
transformer_block_6437:(
dense_12_6442:(
dense_12_6444:
dense_13_6448:
dense_13_6450:
identity?? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?)transformer_block/StatefulPartitionedCall?%w2v_embedding/StatefulPartitionedCall?
%w2v_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_6400*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_53232'
%w2v_embedding/StatefulPartitionedCall?
)transformer_block/StatefulPartitionedCallStatefulPartitionedCall.w2v_embedding/StatefulPartitionedCall:output:0transformer_block_6403transformer_block_6405transformer_block_6407transformer_block_6409transformer_block_6411transformer_block_6413transformer_block_6415transformer_block_6417transformer_block_6419transformer_block_6421transformer_block_6423transformer_block_6425transformer_block_6427transformer_block_6429transformer_block_6431transformer_block_6433transformer_block_6435transformer_block_6437*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????(*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_transformer_block_layer_call_and_return_conditional_losses_62972+
)transformer_block/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_57192*
(global_average_pooling1d/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_58822#
!dropout_2/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_12_6442dense_12_6444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_57392"
 dense_12/StatefulPartitionedCall?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_3_layer_call_and_return_conditional_losses_58492#
!dropout_3/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_13_6448dense_13_6450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_57632"
 dense_13/StatefulPartitionedCall?
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall&^w2v_embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall2N
%w2v_embedding/StatefulPartitionedCall%w2v_embedding/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?*
__inference__traced_save_9006
file_prefixA
=savev2_word2_vec_w2v_embedding_embeddings_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopL
Hsavev2_transformer_block_multi_head_att_dense_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_1_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_2_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_3_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_4_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_5_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_6_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_7_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_8_kernel_read_readvariableopN
Jsavev2_transformer_block_multi_head_att_dense_9_kernel_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableopJ
Fsavev2_transformer_block_layer_normalization_gamma_read_readvariableopI
Esavev2_transformer_block_layer_normalization_beta_read_readvariableopL
Hsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopH
Dsavev2_adam_word2_vec_w2v_embedding_embeddings_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableopS
Osavev2_adam_transformer_block_multi_head_att_dense_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_1_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_2_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_3_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_4_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_5_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_6_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_7_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_8_kernel_m_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_9_kernel_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableopH
Dsavev2_adam_word2_vec_w2v_embedding_embeddings_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableopS
Osavev2_adam_transformer_block_multi_head_att_dense_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_1_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_2_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_3_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_4_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_5_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_6_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_7_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_8_kernel_v_read_readvariableopU
Qsavev2_adam_transformer_block_multi_head_att_dense_9_kernel_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableop
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
ShardedFilename?*
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*?)
value?)B?)OB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:O*
dtype0*?
value?B?OB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_word2_vec_w2v_embedding_embeddings_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopHsavev2_transformer_block_multi_head_att_dense_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_1_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_2_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_3_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_4_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_5_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_6_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_7_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_8_kernel_read_readvariableopJsavev2_transformer_block_multi_head_att_dense_9_kernel_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableopFsavev2_transformer_block_layer_normalization_gamma_read_readvariableopEsavev2_transformer_block_layer_normalization_beta_read_readvariableopHsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopGsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopDsavev2_adam_word2_vec_w2v_embedding_embeddings_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableopOsavev2_adam_transformer_block_multi_head_att_dense_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_1_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_2_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_3_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_4_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_5_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_6_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_7_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_8_kernel_m_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_9_kernel_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableopDsavev2_adam_word2_vec_w2v_embedding_embeddings_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopOsavev2_adam_transformer_block_multi_head_att_dense_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_1_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_2_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_3_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_4_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_5_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_6_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_7_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_8_kernel_v_read_readvariableopQsavev2_adam_transformer_block_multi_head_att_dense_9_kernel_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *]
dtypesS
Q2O	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??(:(:::: : : : : :((:((:((:((:((:((:((:((:((:x(:( : : (:(:(:(:(:(: : : : :
??(:(::::((:((:((:((:((:((:((:((:((:x(:( : : (:(:(:(:(:(:
??(:(::::((:((:((:((:((:((:((:((:((:x(:( : : (:(:(:(:(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??(:$ 

_output_shapes

:(: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::
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
: :$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:((:$ 

_output_shapes

:x(:$ 

_output_shapes

:( : 

_output_shapes
: :$ 

_output_shapes

: (: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :&!"
 
_output_shapes
:
??(:$" 

_output_shapes

:(: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:((:$' 

_output_shapes

:((:$( 

_output_shapes

:((:$) 

_output_shapes

:((:$* 

_output_shapes

:((:$+ 

_output_shapes

:((:$, 

_output_shapes

:((:$- 

_output_shapes

:((:$. 

_output_shapes

:((:$/ 

_output_shapes

:x(:$0 

_output_shapes

:( : 1

_output_shapes
: :$2 

_output_shapes

: (: 3

_output_shapes
:(: 4

_output_shapes
:(: 5

_output_shapes
:(: 6

_output_shapes
:(: 7

_output_shapes
:(:&8"
 
_output_shapes
:
??(:$9 

_output_shapes

:(: :

_output_shapes
::$; 

_output_shapes

:: <

_output_shapes
::$= 

_output_shapes

:((:$> 

_output_shapes

:((:$? 

_output_shapes

:((:$@ 

_output_shapes

:((:$A 

_output_shapes

:((:$B 

_output_shapes

:((:$C 

_output_shapes

:((:$D 

_output_shapes

:((:$E 

_output_shapes

:((:$F 

_output_shapes

:x(:$G 

_output_shapes

:( : H

_output_shapes
: :$I 

_output_shapes

: (: J

_output_shapes
:(: K

_output_shapes
:(: L

_output_shapes
:(: M

_output_shapes
:(: N

_output_shapes
:(:O

_output_shapes
: 
??
?
K__inference_transformer_block_layer_call_and_return_conditional_losses_6297

inputsH
6multi_head_att_dense_tensordot_readvariableop_resource:((J
8multi_head_att_dense_3_tensordot_readvariableop_resource:((J
8multi_head_att_dense_6_tensordot_readvariableop_resource:((J
8multi_head_att_dense_1_tensordot_readvariableop_resource:((J
8multi_head_att_dense_4_tensordot_readvariableop_resource:((J
8multi_head_att_dense_7_tensordot_readvariableop_resource:((J
8multi_head_att_dense_2_tensordot_readvariableop_resource:((J
8multi_head_att_dense_5_tensordot_readvariableop_resource:((J
8multi_head_att_dense_8_tensordot_readvariableop_resource:((J
8multi_head_att_dense_9_tensordot_readvariableop_resource:x(G
9layer_normalization_batchnorm_mul_readvariableop_resource:(C
5layer_normalization_batchnorm_readvariableop_resource:(G
5sequential_dense_10_tensordot_readvariableop_resource:( A
3sequential_dense_10_biasadd_readvariableop_resource: G
5sequential_dense_11_tensordot_readvariableop_resource: (A
3sequential_dense_11_biasadd_readvariableop_resource:(I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:(E
7layer_normalization_1_batchnorm_readvariableop_resource:(
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?-multi_head_att/dense/Tensordot/ReadVariableOp?/multi_head_att/dense_1/Tensordot/ReadVariableOp?/multi_head_att/dense_2/Tensordot/ReadVariableOp?/multi_head_att/dense_3/Tensordot/ReadVariableOp?/multi_head_att/dense_4/Tensordot/ReadVariableOp?/multi_head_att/dense_5/Tensordot/ReadVariableOp?/multi_head_att/dense_6/Tensordot/ReadVariableOp?/multi_head_att/dense_7/Tensordot/ReadVariableOp?/multi_head_att/dense_8/Tensordot/ReadVariableOp?/multi_head_att/dense_9/Tensordot/ReadVariableOp?*sequential/dense_10/BiasAdd/ReadVariableOp?,sequential/dense_10/Tensordot/ReadVariableOp?*sequential/dense_11/BiasAdd/ReadVariableOp?,sequential/dense_11/Tensordot/ReadVariableOpb
multi_head_att/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_att/Shape?
"multi_head_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"multi_head_att/strided_slice/stack?
$multi_head_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_1?
$multi_head_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_2?
multi_head_att/strided_sliceStridedSlicemulti_head_att/Shape:output:0+multi_head_att/strided_slice/stack:output:0-multi_head_att/strided_slice/stack_1:output:0-multi_head_att/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
multi_head_att/strided_slice?
-multi_head_att/dense/Tensordot/ReadVariableOpReadVariableOp6multi_head_att_dense_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02/
-multi_head_att/dense/Tensordot/ReadVariableOp?
#multi_head_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#multi_head_att/dense/Tensordot/axes?
#multi_head_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#multi_head_att/dense/Tensordot/free?
$multi_head_att/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/Shape?
,multi_head_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/GatherV2/axis?
'multi_head_att/dense/Tensordot/GatherV2GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/free:output:05multi_head_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/GatherV2?
.multi_head_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense/Tensordot/GatherV2_1/axis?
)multi_head_att/dense/Tensordot/GatherV2_1GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/axes:output:07multi_head_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense/Tensordot/GatherV2_1?
$multi_head_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$multi_head_att/dense/Tensordot/Const?
#multi_head_att/dense/Tensordot/ProdProd0multi_head_att/dense/Tensordot/GatherV2:output:0-multi_head_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#multi_head_att/dense/Tensordot/Prod?
&multi_head_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense/Tensordot/Const_1?
%multi_head_att/dense/Tensordot/Prod_1Prod2multi_head_att/dense/Tensordot/GatherV2_1:output:0/multi_head_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense/Tensordot/Prod_1?
*multi_head_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*multi_head_att/dense/Tensordot/concat/axis?
%multi_head_att/dense/Tensordot/concatConcatV2,multi_head_att/dense/Tensordot/free:output:0,multi_head_att/dense/Tensordot/axes:output:03multi_head_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%multi_head_att/dense/Tensordot/concat?
$multi_head_att/dense/Tensordot/stackPack,multi_head_att/dense/Tensordot/Prod:output:0.multi_head_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/stack?
(multi_head_att/dense/Tensordot/transpose	Transposeinputs.multi_head_att/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2*
(multi_head_att/dense/Tensordot/transpose?
&multi_head_att/dense/Tensordot/ReshapeReshape,multi_head_att/dense/Tensordot/transpose:y:0-multi_head_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&multi_head_att/dense/Tensordot/Reshape?
%multi_head_att/dense/Tensordot/MatMulMatMul/multi_head_att/dense/Tensordot/Reshape:output:05multi_head_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2'
%multi_head_att/dense/Tensordot/MatMul?
&multi_head_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2(
&multi_head_att/dense/Tensordot/Const_2?
,multi_head_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/concat_1/axis?
'multi_head_att/dense/Tensordot/concat_1ConcatV20multi_head_att/dense/Tensordot/GatherV2:output:0/multi_head_att/dense/Tensordot/Const_2:output:05multi_head_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/concat_1?
multi_head_att/dense/TensordotReshape/multi_head_att/dense/Tensordot/MatMul:product:00multi_head_att/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2 
multi_head_att/dense/Tensordot?
/multi_head_att/dense_3/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_3/Tensordot/ReadVariableOp?
%multi_head_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_3/Tensordot/axes?
%multi_head_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_3/Tensordot/free?
&multi_head_att/dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/Shape?
.multi_head_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/GatherV2/axis?
)multi_head_att/dense_3/Tensordot/GatherV2GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/free:output:07multi_head_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/GatherV2?
0multi_head_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_3/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_3/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/axes:output:09multi_head_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_3/Tensordot/GatherV2_1?
&multi_head_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_3/Tensordot/Const?
%multi_head_att/dense_3/Tensordot/ProdProd2multi_head_att/dense_3/Tensordot/GatherV2:output:0/multi_head_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_3/Tensordot/Prod?
(multi_head_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_3/Tensordot/Const_1?
'multi_head_att/dense_3/Tensordot/Prod_1Prod4multi_head_att/dense_3/Tensordot/GatherV2_1:output:01multi_head_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_3/Tensordot/Prod_1?
,multi_head_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_3/Tensordot/concat/axis?
'multi_head_att/dense_3/Tensordot/concatConcatV2.multi_head_att/dense_3/Tensordot/free:output:0.multi_head_att/dense_3/Tensordot/axes:output:05multi_head_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_3/Tensordot/concat?
&multi_head_att/dense_3/Tensordot/stackPack.multi_head_att/dense_3/Tensordot/Prod:output:00multi_head_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/stack?
*multi_head_att/dense_3/Tensordot/transpose	Transposeinputs0multi_head_att/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_3/Tensordot/transpose?
(multi_head_att/dense_3/Tensordot/ReshapeReshape.multi_head_att/dense_3/Tensordot/transpose:y:0/multi_head_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_3/Tensordot/Reshape?
'multi_head_att/dense_3/Tensordot/MatMulMatMul1multi_head_att/dense_3/Tensordot/Reshape:output:07multi_head_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_3/Tensordot/MatMul?
(multi_head_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_3/Tensordot/Const_2?
.multi_head_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/concat_1/axis?
)multi_head_att/dense_3/Tensordot/concat_1ConcatV22multi_head_att/dense_3/Tensordot/GatherV2:output:01multi_head_att/dense_3/Tensordot/Const_2:output:07multi_head_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/concat_1?
 multi_head_att/dense_3/TensordotReshape1multi_head_att/dense_3/Tensordot/MatMul:product:02multi_head_att/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_3/Tensordot?
/multi_head_att/dense_6/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_6/Tensordot/ReadVariableOp?
%multi_head_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_6/Tensordot/axes?
%multi_head_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_6/Tensordot/free?
&multi_head_att/dense_6/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/Shape?
.multi_head_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/GatherV2/axis?
)multi_head_att/dense_6/Tensordot/GatherV2GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/free:output:07multi_head_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/GatherV2?
0multi_head_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_6/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_6/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/axes:output:09multi_head_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_6/Tensordot/GatherV2_1?
&multi_head_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_6/Tensordot/Const?
%multi_head_att/dense_6/Tensordot/ProdProd2multi_head_att/dense_6/Tensordot/GatherV2:output:0/multi_head_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_6/Tensordot/Prod?
(multi_head_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_6/Tensordot/Const_1?
'multi_head_att/dense_6/Tensordot/Prod_1Prod4multi_head_att/dense_6/Tensordot/GatherV2_1:output:01multi_head_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_6/Tensordot/Prod_1?
,multi_head_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_6/Tensordot/concat/axis?
'multi_head_att/dense_6/Tensordot/concatConcatV2.multi_head_att/dense_6/Tensordot/free:output:0.multi_head_att/dense_6/Tensordot/axes:output:05multi_head_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_6/Tensordot/concat?
&multi_head_att/dense_6/Tensordot/stackPack.multi_head_att/dense_6/Tensordot/Prod:output:00multi_head_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/stack?
*multi_head_att/dense_6/Tensordot/transpose	Transposeinputs0multi_head_att/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_6/Tensordot/transpose?
(multi_head_att/dense_6/Tensordot/ReshapeReshape.multi_head_att/dense_6/Tensordot/transpose:y:0/multi_head_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_6/Tensordot/Reshape?
'multi_head_att/dense_6/Tensordot/MatMulMatMul1multi_head_att/dense_6/Tensordot/Reshape:output:07multi_head_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_6/Tensordot/MatMul?
(multi_head_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_6/Tensordot/Const_2?
.multi_head_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/concat_1/axis?
)multi_head_att/dense_6/Tensordot/concat_1ConcatV22multi_head_att/dense_6/Tensordot/GatherV2:output:01multi_head_att/dense_6/Tensordot/Const_2:output:07multi_head_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/concat_1?
 multi_head_att/dense_6/TensordotReshape1multi_head_att/dense_6/Tensordot/MatMul:product:02multi_head_att/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_6/Tensordot?
multi_head_att/MatMulBatchMatMulV2'multi_head_att/dense/Tensordot:output:0)multi_head_att/dense_3/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMuly
multi_head_att/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv/y?
multi_head_att/truedivRealDivmulti_head_att/MatMul:output:0!multi_head_att/truediv/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv?
multi_head_att/SoftmaxSoftmaxmulti_head_att/truediv:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax?
multi_head_att/MatMul_1BatchMatMulV2 multi_head_att/Softmax:softmax:0)multi_head_att/dense_6/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_1?
/multi_head_att/dense_1/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_1/Tensordot/ReadVariableOp?
%multi_head_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_1/Tensordot/axes?
%multi_head_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_1/Tensordot/free?
&multi_head_att/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/Shape?
.multi_head_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/GatherV2/axis?
)multi_head_att/dense_1/Tensordot/GatherV2GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/free:output:07multi_head_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/GatherV2?
0multi_head_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_1/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_1/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/axes:output:09multi_head_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_1/Tensordot/GatherV2_1?
&multi_head_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_1/Tensordot/Const?
%multi_head_att/dense_1/Tensordot/ProdProd2multi_head_att/dense_1/Tensordot/GatherV2:output:0/multi_head_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_1/Tensordot/Prod?
(multi_head_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_1/Tensordot/Const_1?
'multi_head_att/dense_1/Tensordot/Prod_1Prod4multi_head_att/dense_1/Tensordot/GatherV2_1:output:01multi_head_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_1/Tensordot/Prod_1?
,multi_head_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_1/Tensordot/concat/axis?
'multi_head_att/dense_1/Tensordot/concatConcatV2.multi_head_att/dense_1/Tensordot/free:output:0.multi_head_att/dense_1/Tensordot/axes:output:05multi_head_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_1/Tensordot/concat?
&multi_head_att/dense_1/Tensordot/stackPack.multi_head_att/dense_1/Tensordot/Prod:output:00multi_head_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/stack?
*multi_head_att/dense_1/Tensordot/transpose	Transposeinputs0multi_head_att/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_1/Tensordot/transpose?
(multi_head_att/dense_1/Tensordot/ReshapeReshape.multi_head_att/dense_1/Tensordot/transpose:y:0/multi_head_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_1/Tensordot/Reshape?
'multi_head_att/dense_1/Tensordot/MatMulMatMul1multi_head_att/dense_1/Tensordot/Reshape:output:07multi_head_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_1/Tensordot/MatMul?
(multi_head_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_1/Tensordot/Const_2?
.multi_head_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/concat_1/axis?
)multi_head_att/dense_1/Tensordot/concat_1ConcatV22multi_head_att/dense_1/Tensordot/GatherV2:output:01multi_head_att/dense_1/Tensordot/Const_2:output:07multi_head_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/concat_1?
 multi_head_att/dense_1/TensordotReshape1multi_head_att/dense_1/Tensordot/MatMul:product:02multi_head_att/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_1/Tensordot?
/multi_head_att/dense_4/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_4/Tensordot/ReadVariableOp?
%multi_head_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_4/Tensordot/axes?
%multi_head_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_4/Tensordot/free?
&multi_head_att/dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/Shape?
.multi_head_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/GatherV2/axis?
)multi_head_att/dense_4/Tensordot/GatherV2GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/free:output:07multi_head_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/GatherV2?
0multi_head_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_4/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_4/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/axes:output:09multi_head_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_4/Tensordot/GatherV2_1?
&multi_head_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_4/Tensordot/Const?
%multi_head_att/dense_4/Tensordot/ProdProd2multi_head_att/dense_4/Tensordot/GatherV2:output:0/multi_head_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_4/Tensordot/Prod?
(multi_head_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_4/Tensordot/Const_1?
'multi_head_att/dense_4/Tensordot/Prod_1Prod4multi_head_att/dense_4/Tensordot/GatherV2_1:output:01multi_head_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_4/Tensordot/Prod_1?
,multi_head_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_4/Tensordot/concat/axis?
'multi_head_att/dense_4/Tensordot/concatConcatV2.multi_head_att/dense_4/Tensordot/free:output:0.multi_head_att/dense_4/Tensordot/axes:output:05multi_head_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_4/Tensordot/concat?
&multi_head_att/dense_4/Tensordot/stackPack.multi_head_att/dense_4/Tensordot/Prod:output:00multi_head_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/stack?
*multi_head_att/dense_4/Tensordot/transpose	Transposeinputs0multi_head_att/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_4/Tensordot/transpose?
(multi_head_att/dense_4/Tensordot/ReshapeReshape.multi_head_att/dense_4/Tensordot/transpose:y:0/multi_head_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_4/Tensordot/Reshape?
'multi_head_att/dense_4/Tensordot/MatMulMatMul1multi_head_att/dense_4/Tensordot/Reshape:output:07multi_head_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_4/Tensordot/MatMul?
(multi_head_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_4/Tensordot/Const_2?
.multi_head_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/concat_1/axis?
)multi_head_att/dense_4/Tensordot/concat_1ConcatV22multi_head_att/dense_4/Tensordot/GatherV2:output:01multi_head_att/dense_4/Tensordot/Const_2:output:07multi_head_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/concat_1?
 multi_head_att/dense_4/TensordotReshape1multi_head_att/dense_4/Tensordot/MatMul:product:02multi_head_att/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_4/Tensordot?
/multi_head_att/dense_7/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_7_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_7/Tensordot/ReadVariableOp?
%multi_head_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_7/Tensordot/axes?
%multi_head_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_7/Tensordot/free?
&multi_head_att/dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/Shape?
.multi_head_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/GatherV2/axis?
)multi_head_att/dense_7/Tensordot/GatherV2GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/free:output:07multi_head_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/GatherV2?
0multi_head_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_7/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_7/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/axes:output:09multi_head_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_7/Tensordot/GatherV2_1?
&multi_head_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_7/Tensordot/Const?
%multi_head_att/dense_7/Tensordot/ProdProd2multi_head_att/dense_7/Tensordot/GatherV2:output:0/multi_head_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_7/Tensordot/Prod?
(multi_head_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_7/Tensordot/Const_1?
'multi_head_att/dense_7/Tensordot/Prod_1Prod4multi_head_att/dense_7/Tensordot/GatherV2_1:output:01multi_head_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_7/Tensordot/Prod_1?
,multi_head_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_7/Tensordot/concat/axis?
'multi_head_att/dense_7/Tensordot/concatConcatV2.multi_head_att/dense_7/Tensordot/free:output:0.multi_head_att/dense_7/Tensordot/axes:output:05multi_head_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_7/Tensordot/concat?
&multi_head_att/dense_7/Tensordot/stackPack.multi_head_att/dense_7/Tensordot/Prod:output:00multi_head_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/stack?
*multi_head_att/dense_7/Tensordot/transpose	Transposeinputs0multi_head_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_7/Tensordot/transpose?
(multi_head_att/dense_7/Tensordot/ReshapeReshape.multi_head_att/dense_7/Tensordot/transpose:y:0/multi_head_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_7/Tensordot/Reshape?
'multi_head_att/dense_7/Tensordot/MatMulMatMul1multi_head_att/dense_7/Tensordot/Reshape:output:07multi_head_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_7/Tensordot/MatMul?
(multi_head_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_7/Tensordot/Const_2?
.multi_head_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/concat_1/axis?
)multi_head_att/dense_7/Tensordot/concat_1ConcatV22multi_head_att/dense_7/Tensordot/GatherV2:output:01multi_head_att/dense_7/Tensordot/Const_2:output:07multi_head_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/concat_1?
 multi_head_att/dense_7/TensordotReshape1multi_head_att/dense_7/Tensordot/MatMul:product:02multi_head_att/dense_7/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_7/Tensordot?
multi_head_att/MatMul_2BatchMatMulV2)multi_head_att/dense_1/Tensordot:output:0)multi_head_att/dense_4/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_2}
multi_head_att/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_1/y?
multi_head_att/truediv_1RealDiv multi_head_att/MatMul_2:output:0#multi_head_att/truediv_1/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_1?
multi_head_att/Softmax_1Softmaxmulti_head_att/truediv_1:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_1?
multi_head_att/MatMul_3BatchMatMulV2"multi_head_att/Softmax_1:softmax:0)multi_head_att/dense_7/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_3?
/multi_head_att/dense_2/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_2/Tensordot/ReadVariableOp?
%multi_head_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_2/Tensordot/axes?
%multi_head_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_2/Tensordot/free?
&multi_head_att/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/Shape?
.multi_head_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/GatherV2/axis?
)multi_head_att/dense_2/Tensordot/GatherV2GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/free:output:07multi_head_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/GatherV2?
0multi_head_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_2/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_2/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/axes:output:09multi_head_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_2/Tensordot/GatherV2_1?
&multi_head_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_2/Tensordot/Const?
%multi_head_att/dense_2/Tensordot/ProdProd2multi_head_att/dense_2/Tensordot/GatherV2:output:0/multi_head_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_2/Tensordot/Prod?
(multi_head_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_2/Tensordot/Const_1?
'multi_head_att/dense_2/Tensordot/Prod_1Prod4multi_head_att/dense_2/Tensordot/GatherV2_1:output:01multi_head_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_2/Tensordot/Prod_1?
,multi_head_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_2/Tensordot/concat/axis?
'multi_head_att/dense_2/Tensordot/concatConcatV2.multi_head_att/dense_2/Tensordot/free:output:0.multi_head_att/dense_2/Tensordot/axes:output:05multi_head_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_2/Tensordot/concat?
&multi_head_att/dense_2/Tensordot/stackPack.multi_head_att/dense_2/Tensordot/Prod:output:00multi_head_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/stack?
*multi_head_att/dense_2/Tensordot/transpose	Transposeinputs0multi_head_att/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_2/Tensordot/transpose?
(multi_head_att/dense_2/Tensordot/ReshapeReshape.multi_head_att/dense_2/Tensordot/transpose:y:0/multi_head_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_2/Tensordot/Reshape?
'multi_head_att/dense_2/Tensordot/MatMulMatMul1multi_head_att/dense_2/Tensordot/Reshape:output:07multi_head_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_2/Tensordot/MatMul?
(multi_head_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_2/Tensordot/Const_2?
.multi_head_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/concat_1/axis?
)multi_head_att/dense_2/Tensordot/concat_1ConcatV22multi_head_att/dense_2/Tensordot/GatherV2:output:01multi_head_att/dense_2/Tensordot/Const_2:output:07multi_head_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/concat_1?
 multi_head_att/dense_2/TensordotReshape1multi_head_att/dense_2/Tensordot/MatMul:product:02multi_head_att/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_2/Tensordot?
/multi_head_att/dense_5/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_5_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_5/Tensordot/ReadVariableOp?
%multi_head_att/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_5/Tensordot/axes?
%multi_head_att/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_5/Tensordot/free?
&multi_head_att/dense_5/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/Shape?
.multi_head_att/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/GatherV2/axis?
)multi_head_att/dense_5/Tensordot/GatherV2GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/free:output:07multi_head_att/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/GatherV2?
0multi_head_att/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_5/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_5/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/axes:output:09multi_head_att/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_5/Tensordot/GatherV2_1?
&multi_head_att/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_5/Tensordot/Const?
%multi_head_att/dense_5/Tensordot/ProdProd2multi_head_att/dense_5/Tensordot/GatherV2:output:0/multi_head_att/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_5/Tensordot/Prod?
(multi_head_att/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_5/Tensordot/Const_1?
'multi_head_att/dense_5/Tensordot/Prod_1Prod4multi_head_att/dense_5/Tensordot/GatherV2_1:output:01multi_head_att/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_5/Tensordot/Prod_1?
,multi_head_att/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_5/Tensordot/concat/axis?
'multi_head_att/dense_5/Tensordot/concatConcatV2.multi_head_att/dense_5/Tensordot/free:output:0.multi_head_att/dense_5/Tensordot/axes:output:05multi_head_att/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_5/Tensordot/concat?
&multi_head_att/dense_5/Tensordot/stackPack.multi_head_att/dense_5/Tensordot/Prod:output:00multi_head_att/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/stack?
*multi_head_att/dense_5/Tensordot/transpose	Transposeinputs0multi_head_att/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_5/Tensordot/transpose?
(multi_head_att/dense_5/Tensordot/ReshapeReshape.multi_head_att/dense_5/Tensordot/transpose:y:0/multi_head_att/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_5/Tensordot/Reshape?
'multi_head_att/dense_5/Tensordot/MatMulMatMul1multi_head_att/dense_5/Tensordot/Reshape:output:07multi_head_att/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_5/Tensordot/MatMul?
(multi_head_att/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_5/Tensordot/Const_2?
.multi_head_att/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/concat_1/axis?
)multi_head_att/dense_5/Tensordot/concat_1ConcatV22multi_head_att/dense_5/Tensordot/GatherV2:output:01multi_head_att/dense_5/Tensordot/Const_2:output:07multi_head_att/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/concat_1?
 multi_head_att/dense_5/TensordotReshape1multi_head_att/dense_5/Tensordot/MatMul:product:02multi_head_att/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_5/Tensordot?
/multi_head_att/dense_8/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_8/Tensordot/ReadVariableOp?
%multi_head_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_8/Tensordot/axes?
%multi_head_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_8/Tensordot/free?
&multi_head_att/dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/Shape?
.multi_head_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/GatherV2/axis?
)multi_head_att/dense_8/Tensordot/GatherV2GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/free:output:07multi_head_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/GatherV2?
0multi_head_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_8/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_8/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/axes:output:09multi_head_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_8/Tensordot/GatherV2_1?
&multi_head_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_8/Tensordot/Const?
%multi_head_att/dense_8/Tensordot/ProdProd2multi_head_att/dense_8/Tensordot/GatherV2:output:0/multi_head_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_8/Tensordot/Prod?
(multi_head_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_8/Tensordot/Const_1?
'multi_head_att/dense_8/Tensordot/Prod_1Prod4multi_head_att/dense_8/Tensordot/GatherV2_1:output:01multi_head_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_8/Tensordot/Prod_1?
,multi_head_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_8/Tensordot/concat/axis?
'multi_head_att/dense_8/Tensordot/concatConcatV2.multi_head_att/dense_8/Tensordot/free:output:0.multi_head_att/dense_8/Tensordot/axes:output:05multi_head_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_8/Tensordot/concat?
&multi_head_att/dense_8/Tensordot/stackPack.multi_head_att/dense_8/Tensordot/Prod:output:00multi_head_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/stack?
*multi_head_att/dense_8/Tensordot/transpose	Transposeinputs0multi_head_att/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_8/Tensordot/transpose?
(multi_head_att/dense_8/Tensordot/ReshapeReshape.multi_head_att/dense_8/Tensordot/transpose:y:0/multi_head_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_8/Tensordot/Reshape?
'multi_head_att/dense_8/Tensordot/MatMulMatMul1multi_head_att/dense_8/Tensordot/Reshape:output:07multi_head_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_8/Tensordot/MatMul?
(multi_head_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_8/Tensordot/Const_2?
.multi_head_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/concat_1/axis?
)multi_head_att/dense_8/Tensordot/concat_1ConcatV22multi_head_att/dense_8/Tensordot/GatherV2:output:01multi_head_att/dense_8/Tensordot/Const_2:output:07multi_head_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/concat_1?
 multi_head_att/dense_8/TensordotReshape1multi_head_att/dense_8/Tensordot/MatMul:product:02multi_head_att/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_8/Tensordot?
multi_head_att/MatMul_4BatchMatMulV2)multi_head_att/dense_2/Tensordot:output:0)multi_head_att/dense_5/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_4}
multi_head_att/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_2/y?
multi_head_att/truediv_2RealDiv multi_head_att/MatMul_4:output:0#multi_head_att/truediv_2/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_2?
multi_head_att/Softmax_2Softmaxmulti_head_att/truediv_2:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_2?
multi_head_att/MatMul_5BatchMatMulV2"multi_head_att/Softmax_2:softmax:0)multi_head_att/dense_8/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_5z
multi_head_att/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
multi_head_att/concat/axis?
multi_head_att/concatConcatV2 multi_head_att/MatMul_1:output:0 multi_head_att/MatMul_3:output:0 multi_head_att/MatMul_5:output:0#multi_head_att/concat/axis:output:0*
N*
T0*,
_output_shapes
:??????????x2
multi_head_att/concat?
/multi_head_att/dense_9/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

:x(*
dtype021
/multi_head_att/dense_9/Tensordot/ReadVariableOp?
%multi_head_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_9/Tensordot/axes?
%multi_head_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_9/Tensordot/free?
&multi_head_att/dense_9/Tensordot/ShapeShapemulti_head_att/concat:output:0*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/Shape?
.multi_head_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/GatherV2/axis?
)multi_head_att/dense_9/Tensordot/GatherV2GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/free:output:07multi_head_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/GatherV2?
0multi_head_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_9/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_9/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/axes:output:09multi_head_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_9/Tensordot/GatherV2_1?
&multi_head_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_9/Tensordot/Const?
%multi_head_att/dense_9/Tensordot/ProdProd2multi_head_att/dense_9/Tensordot/GatherV2:output:0/multi_head_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_9/Tensordot/Prod?
(multi_head_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_9/Tensordot/Const_1?
'multi_head_att/dense_9/Tensordot/Prod_1Prod4multi_head_att/dense_9/Tensordot/GatherV2_1:output:01multi_head_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_9/Tensordot/Prod_1?
,multi_head_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_9/Tensordot/concat/axis?
'multi_head_att/dense_9/Tensordot/concatConcatV2.multi_head_att/dense_9/Tensordot/free:output:0.multi_head_att/dense_9/Tensordot/axes:output:05multi_head_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_9/Tensordot/concat?
&multi_head_att/dense_9/Tensordot/stackPack.multi_head_att/dense_9/Tensordot/Prod:output:00multi_head_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/stack?
*multi_head_att/dense_9/Tensordot/transpose	Transposemulti_head_att/concat:output:00multi_head_att/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????x2,
*multi_head_att/dense_9/Tensordot/transpose?
(multi_head_att/dense_9/Tensordot/ReshapeReshape.multi_head_att/dense_9/Tensordot/transpose:y:0/multi_head_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_9/Tensordot/Reshape?
'multi_head_att/dense_9/Tensordot/MatMulMatMul1multi_head_att/dense_9/Tensordot/Reshape:output:07multi_head_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_9/Tensordot/MatMul?
(multi_head_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_9/Tensordot/Const_2?
.multi_head_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/concat_1/axis?
)multi_head_att/dense_9/Tensordot/concat_1ConcatV22multi_head_att/dense_9/Tensordot/GatherV2:output:01multi_head_att/dense_9/Tensordot/Const_2:output:07multi_head_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/concat_1?
 multi_head_att/dense_9/TensordotReshape1multi_head_att/dense_9/Tensordot/MatMul:product:02multi_head_att/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_9/Tensordots
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/dropout/Const?
dropout/dropout/MulMul)multi_head_att/dense_9/Tensordot:output:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????(2
dropout/dropout/Mul?
dropout/dropout/ShapeShape)multi_head_att/dense_9/Tensordot:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????(*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????(2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????(2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????(2
dropout/dropout/Mul_1m
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????(2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:??????????2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/add_1?
,sequential/dense_10/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02.
,sequential/dense_10/Tensordot/ReadVariableOp?
"sequential/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_10/Tensordot/axes?
"sequential/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_10/Tensordot/free?
#sequential/dense_10/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/Shape?
+sequential/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/GatherV2/axis?
&sequential/dense_10/Tensordot/GatherV2GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/free:output:04sequential/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/GatherV2?
-sequential/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_10/Tensordot/GatherV2_1/axis?
(sequential/dense_10/Tensordot/GatherV2_1GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/axes:output:06sequential/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_10/Tensordot/GatherV2_1?
#sequential/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_10/Tensordot/Const?
"sequential/dense_10/Tensordot/ProdProd/sequential/dense_10/Tensordot/GatherV2:output:0,sequential/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_10/Tensordot/Prod?
%sequential/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_1?
$sequential/dense_10/Tensordot/Prod_1Prod1sequential/dense_10/Tensordot/GatherV2_1:output:0.sequential/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_10/Tensordot/Prod_1?
)sequential/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_10/Tensordot/concat/axis?
$sequential/dense_10/Tensordot/concatConcatV2+sequential/dense_10/Tensordot/free:output:0+sequential/dense_10/Tensordot/axes:output:02sequential/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_10/Tensordot/concat?
#sequential/dense_10/Tensordot/stackPack+sequential/dense_10/Tensordot/Prod:output:0-sequential/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/stack?
'sequential/dense_10/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0-sequential/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2)
'sequential/dense_10/Tensordot/transpose?
%sequential/dense_10/Tensordot/ReshapeReshape+sequential/dense_10/Tensordot/transpose:y:0,sequential/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_10/Tensordot/Reshape?
$sequential/dense_10/Tensordot/MatMulMatMul.sequential/dense_10/Tensordot/Reshape:output:04sequential/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$sequential/dense_10/Tensordot/MatMul?
%sequential/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_2?
+sequential/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/concat_1/axis?
&sequential/dense_10/Tensordot/concat_1ConcatV2/sequential/dense_10/Tensordot/GatherV2:output:0.sequential/dense_10/Tensordot/Const_2:output:04sequential/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/concat_1?
sequential/dense_10/TensordotReshape.sequential/dense_10/Tensordot/MatMul:product:0/sequential/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Tensordot?
*sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/dense_10/BiasAdd/ReadVariableOp?
sequential/dense_10/BiasAddBiasAdd&sequential/dense_10/Tensordot:output:02sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/BiasAdd?
sequential/dense_10/ReluRelu$sequential/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Relu?
,sequential/dense_11/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02.
,sequential/dense_11/Tensordot/ReadVariableOp?
"sequential/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_11/Tensordot/axes?
"sequential/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_11/Tensordot/free?
#sequential/dense_11/Tensordot/ShapeShape&sequential/dense_10/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/Shape?
+sequential/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/GatherV2/axis?
&sequential/dense_11/Tensordot/GatherV2GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/free:output:04sequential/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/GatherV2?
-sequential/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_11/Tensordot/GatherV2_1/axis?
(sequential/dense_11/Tensordot/GatherV2_1GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/axes:output:06sequential/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_11/Tensordot/GatherV2_1?
#sequential/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_11/Tensordot/Const?
"sequential/dense_11/Tensordot/ProdProd/sequential/dense_11/Tensordot/GatherV2:output:0,sequential/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_11/Tensordot/Prod?
%sequential/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_11/Tensordot/Const_1?
$sequential/dense_11/Tensordot/Prod_1Prod1sequential/dense_11/Tensordot/GatherV2_1:output:0.sequential/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_11/Tensordot/Prod_1?
)sequential/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_11/Tensordot/concat/axis?
$sequential/dense_11/Tensordot/concatConcatV2+sequential/dense_11/Tensordot/free:output:0+sequential/dense_11/Tensordot/axes:output:02sequential/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_11/Tensordot/concat?
#sequential/dense_11/Tensordot/stackPack+sequential/dense_11/Tensordot/Prod:output:0-sequential/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/stack?
'sequential/dense_11/Tensordot/transpose	Transpose&sequential/dense_10/Relu:activations:0-sequential/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2)
'sequential/dense_11/Tensordot/transpose?
%sequential/dense_11/Tensordot/ReshapeReshape+sequential/dense_11/Tensordot/transpose:y:0,sequential/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_11/Tensordot/Reshape?
$sequential/dense_11/Tensordot/MatMulMatMul.sequential/dense_11/Tensordot/Reshape:output:04sequential/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2&
$sequential/dense_11/Tensordot/MatMul?
%sequential/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2'
%sequential/dense_11/Tensordot/Const_2?
+sequential/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/concat_1/axis?
&sequential/dense_11/Tensordot/concat_1ConcatV2/sequential/dense_11/Tensordot/GatherV2:output:0.sequential/dense_11/Tensordot/Const_2:output:04sequential/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/concat_1?
sequential/dense_11/TensordotReshape.sequential/dense_11/Tensordot/MatMul:product:0/sequential/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/Tensordot?
*sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*sequential/dense_11/BiasAdd/ReadVariableOp?
sequential/dense_11/BiasAddBiasAdd&sequential/dense_11/Tensordot:output:02sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul$sequential/dense_11/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:??????????(2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape$sequential/dense_11/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????(*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????(2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????(2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????(2
dropout_1/dropout/Mul_1?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????(2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:??????????2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp.^multi_head_att/dense/Tensordot/ReadVariableOp0^multi_head_att/dense_1/Tensordot/ReadVariableOp0^multi_head_att/dense_2/Tensordot/ReadVariableOp0^multi_head_att/dense_3/Tensordot/ReadVariableOp0^multi_head_att/dense_4/Tensordot/ReadVariableOp0^multi_head_att/dense_5/Tensordot/ReadVariableOp0^multi_head_att/dense_6/Tensordot/ReadVariableOp0^multi_head_att/dense_7/Tensordot/ReadVariableOp0^multi_head_att/dense_8/Tensordot/ReadVariableOp0^multi_head_att/dense_9/Tensordot/ReadVariableOp+^sequential/dense_10/BiasAdd/ReadVariableOp-^sequential/dense_10/Tensordot/ReadVariableOp+^sequential/dense_11/BiasAdd/ReadVariableOp-^sequential/dense_11/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????(: : : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2^
-multi_head_att/dense/Tensordot/ReadVariableOp-multi_head_att/dense/Tensordot/ReadVariableOp2b
/multi_head_att/dense_1/Tensordot/ReadVariableOp/multi_head_att/dense_1/Tensordot/ReadVariableOp2b
/multi_head_att/dense_2/Tensordot/ReadVariableOp/multi_head_att/dense_2/Tensordot/ReadVariableOp2b
/multi_head_att/dense_3/Tensordot/ReadVariableOp/multi_head_att/dense_3/Tensordot/ReadVariableOp2b
/multi_head_att/dense_4/Tensordot/ReadVariableOp/multi_head_att/dense_4/Tensordot/ReadVariableOp2b
/multi_head_att/dense_5/Tensordot/ReadVariableOp/multi_head_att/dense_5/Tensordot/ReadVariableOp2b
/multi_head_att/dense_6/Tensordot/ReadVariableOp/multi_head_att/dense_6/Tensordot/ReadVariableOp2b
/multi_head_att/dense_7/Tensordot/ReadVariableOp/multi_head_att/dense_7/Tensordot/ReadVariableOp2b
/multi_head_att/dense_8/Tensordot/ReadVariableOp/multi_head_att/dense_8/Tensordot/ReadVariableOp2b
/multi_head_att/dense_9/Tensordot/ReadVariableOp/multi_head_att/dense_9/Tensordot/ReadVariableOp2X
*sequential/dense_10/BiasAdd/ReadVariableOp*sequential/dense_10/BiasAdd/ReadVariableOp2\
,sequential/dense_10/Tensordot/ReadVariableOp,sequential/dense_10/Tensordot/ReadVariableOp2X
*sequential/dense_11/BiasAdd/ReadVariableOp*sequential/dense_11/BiasAdd/ReadVariableOp2\
,sequential/dense_11/Tensordot/ReadVariableOp,sequential/dense_11/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_5849

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_7603

inputs
unknown:
??(
	unknown_0:((
	unknown_1:((
	unknown_2:((
	unknown_3:((
	unknown_4:((
	unknown_5:((
	unknown_6:((
	unknown_7:((
	unknown_8:((
	unknown_9:x(

unknown_10:(

unknown_11:(

unknown_12:( 

unknown_13: 

unknown_14: (

unknown_15:(

unknown_16:(

unknown_17:(

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_64542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_dense_13_layer_call_and_return_conditional_losses_5763

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_7552

inputs
unknown:
??(
	unknown_0:((
	unknown_1:((
	unknown_2:((
	unknown_3:((
	unknown_4:((
	unknown_5:((
	unknown_6:((
	unknown_7:((
	unknown_8:((
	unknown_9:x(

unknown_10:(

unknown_11:(

unknown_12:( 

unknown_13: 

unknown_14: (

unknown_15:(

unknown_16:(

unknown_17:(

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_57702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
K__inference_transformer_block_layer_call_and_return_conditional_losses_8332

inputsH
6multi_head_att_dense_tensordot_readvariableop_resource:((J
8multi_head_att_dense_3_tensordot_readvariableop_resource:((J
8multi_head_att_dense_6_tensordot_readvariableop_resource:((J
8multi_head_att_dense_1_tensordot_readvariableop_resource:((J
8multi_head_att_dense_4_tensordot_readvariableop_resource:((J
8multi_head_att_dense_7_tensordot_readvariableop_resource:((J
8multi_head_att_dense_2_tensordot_readvariableop_resource:((J
8multi_head_att_dense_5_tensordot_readvariableop_resource:((J
8multi_head_att_dense_8_tensordot_readvariableop_resource:((J
8multi_head_att_dense_9_tensordot_readvariableop_resource:x(G
9layer_normalization_batchnorm_mul_readvariableop_resource:(C
5layer_normalization_batchnorm_readvariableop_resource:(G
5sequential_dense_10_tensordot_readvariableop_resource:( A
3sequential_dense_10_biasadd_readvariableop_resource: G
5sequential_dense_11_tensordot_readvariableop_resource: (A
3sequential_dense_11_biasadd_readvariableop_resource:(I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:(E
7layer_normalization_1_batchnorm_readvariableop_resource:(
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?-multi_head_att/dense/Tensordot/ReadVariableOp?/multi_head_att/dense_1/Tensordot/ReadVariableOp?/multi_head_att/dense_2/Tensordot/ReadVariableOp?/multi_head_att/dense_3/Tensordot/ReadVariableOp?/multi_head_att/dense_4/Tensordot/ReadVariableOp?/multi_head_att/dense_5/Tensordot/ReadVariableOp?/multi_head_att/dense_6/Tensordot/ReadVariableOp?/multi_head_att/dense_7/Tensordot/ReadVariableOp?/multi_head_att/dense_8/Tensordot/ReadVariableOp?/multi_head_att/dense_9/Tensordot/ReadVariableOp?*sequential/dense_10/BiasAdd/ReadVariableOp?,sequential/dense_10/Tensordot/ReadVariableOp?*sequential/dense_11/BiasAdd/ReadVariableOp?,sequential/dense_11/Tensordot/ReadVariableOpb
multi_head_att/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_att/Shape?
"multi_head_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"multi_head_att/strided_slice/stack?
$multi_head_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_1?
$multi_head_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_2?
multi_head_att/strided_sliceStridedSlicemulti_head_att/Shape:output:0+multi_head_att/strided_slice/stack:output:0-multi_head_att/strided_slice/stack_1:output:0-multi_head_att/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
multi_head_att/strided_slice?
-multi_head_att/dense/Tensordot/ReadVariableOpReadVariableOp6multi_head_att_dense_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02/
-multi_head_att/dense/Tensordot/ReadVariableOp?
#multi_head_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#multi_head_att/dense/Tensordot/axes?
#multi_head_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#multi_head_att/dense/Tensordot/free?
$multi_head_att/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/Shape?
,multi_head_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/GatherV2/axis?
'multi_head_att/dense/Tensordot/GatherV2GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/free:output:05multi_head_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/GatherV2?
.multi_head_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense/Tensordot/GatherV2_1/axis?
)multi_head_att/dense/Tensordot/GatherV2_1GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/axes:output:07multi_head_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense/Tensordot/GatherV2_1?
$multi_head_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$multi_head_att/dense/Tensordot/Const?
#multi_head_att/dense/Tensordot/ProdProd0multi_head_att/dense/Tensordot/GatherV2:output:0-multi_head_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#multi_head_att/dense/Tensordot/Prod?
&multi_head_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense/Tensordot/Const_1?
%multi_head_att/dense/Tensordot/Prod_1Prod2multi_head_att/dense/Tensordot/GatherV2_1:output:0/multi_head_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense/Tensordot/Prod_1?
*multi_head_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*multi_head_att/dense/Tensordot/concat/axis?
%multi_head_att/dense/Tensordot/concatConcatV2,multi_head_att/dense/Tensordot/free:output:0,multi_head_att/dense/Tensordot/axes:output:03multi_head_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%multi_head_att/dense/Tensordot/concat?
$multi_head_att/dense/Tensordot/stackPack,multi_head_att/dense/Tensordot/Prod:output:0.multi_head_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/stack?
(multi_head_att/dense/Tensordot/transpose	Transposeinputs.multi_head_att/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2*
(multi_head_att/dense/Tensordot/transpose?
&multi_head_att/dense/Tensordot/ReshapeReshape,multi_head_att/dense/Tensordot/transpose:y:0-multi_head_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&multi_head_att/dense/Tensordot/Reshape?
%multi_head_att/dense/Tensordot/MatMulMatMul/multi_head_att/dense/Tensordot/Reshape:output:05multi_head_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2'
%multi_head_att/dense/Tensordot/MatMul?
&multi_head_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2(
&multi_head_att/dense/Tensordot/Const_2?
,multi_head_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/concat_1/axis?
'multi_head_att/dense/Tensordot/concat_1ConcatV20multi_head_att/dense/Tensordot/GatherV2:output:0/multi_head_att/dense/Tensordot/Const_2:output:05multi_head_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/concat_1?
multi_head_att/dense/TensordotReshape/multi_head_att/dense/Tensordot/MatMul:product:00multi_head_att/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2 
multi_head_att/dense/Tensordot?
/multi_head_att/dense_3/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_3/Tensordot/ReadVariableOp?
%multi_head_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_3/Tensordot/axes?
%multi_head_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_3/Tensordot/free?
&multi_head_att/dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/Shape?
.multi_head_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/GatherV2/axis?
)multi_head_att/dense_3/Tensordot/GatherV2GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/free:output:07multi_head_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/GatherV2?
0multi_head_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_3/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_3/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/axes:output:09multi_head_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_3/Tensordot/GatherV2_1?
&multi_head_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_3/Tensordot/Const?
%multi_head_att/dense_3/Tensordot/ProdProd2multi_head_att/dense_3/Tensordot/GatherV2:output:0/multi_head_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_3/Tensordot/Prod?
(multi_head_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_3/Tensordot/Const_1?
'multi_head_att/dense_3/Tensordot/Prod_1Prod4multi_head_att/dense_3/Tensordot/GatherV2_1:output:01multi_head_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_3/Tensordot/Prod_1?
,multi_head_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_3/Tensordot/concat/axis?
'multi_head_att/dense_3/Tensordot/concatConcatV2.multi_head_att/dense_3/Tensordot/free:output:0.multi_head_att/dense_3/Tensordot/axes:output:05multi_head_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_3/Tensordot/concat?
&multi_head_att/dense_3/Tensordot/stackPack.multi_head_att/dense_3/Tensordot/Prod:output:00multi_head_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/stack?
*multi_head_att/dense_3/Tensordot/transpose	Transposeinputs0multi_head_att/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_3/Tensordot/transpose?
(multi_head_att/dense_3/Tensordot/ReshapeReshape.multi_head_att/dense_3/Tensordot/transpose:y:0/multi_head_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_3/Tensordot/Reshape?
'multi_head_att/dense_3/Tensordot/MatMulMatMul1multi_head_att/dense_3/Tensordot/Reshape:output:07multi_head_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_3/Tensordot/MatMul?
(multi_head_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_3/Tensordot/Const_2?
.multi_head_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/concat_1/axis?
)multi_head_att/dense_3/Tensordot/concat_1ConcatV22multi_head_att/dense_3/Tensordot/GatherV2:output:01multi_head_att/dense_3/Tensordot/Const_2:output:07multi_head_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/concat_1?
 multi_head_att/dense_3/TensordotReshape1multi_head_att/dense_3/Tensordot/MatMul:product:02multi_head_att/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_3/Tensordot?
/multi_head_att/dense_6/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_6/Tensordot/ReadVariableOp?
%multi_head_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_6/Tensordot/axes?
%multi_head_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_6/Tensordot/free?
&multi_head_att/dense_6/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/Shape?
.multi_head_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/GatherV2/axis?
)multi_head_att/dense_6/Tensordot/GatherV2GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/free:output:07multi_head_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/GatherV2?
0multi_head_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_6/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_6/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/axes:output:09multi_head_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_6/Tensordot/GatherV2_1?
&multi_head_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_6/Tensordot/Const?
%multi_head_att/dense_6/Tensordot/ProdProd2multi_head_att/dense_6/Tensordot/GatherV2:output:0/multi_head_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_6/Tensordot/Prod?
(multi_head_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_6/Tensordot/Const_1?
'multi_head_att/dense_6/Tensordot/Prod_1Prod4multi_head_att/dense_6/Tensordot/GatherV2_1:output:01multi_head_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_6/Tensordot/Prod_1?
,multi_head_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_6/Tensordot/concat/axis?
'multi_head_att/dense_6/Tensordot/concatConcatV2.multi_head_att/dense_6/Tensordot/free:output:0.multi_head_att/dense_6/Tensordot/axes:output:05multi_head_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_6/Tensordot/concat?
&multi_head_att/dense_6/Tensordot/stackPack.multi_head_att/dense_6/Tensordot/Prod:output:00multi_head_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/stack?
*multi_head_att/dense_6/Tensordot/transpose	Transposeinputs0multi_head_att/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_6/Tensordot/transpose?
(multi_head_att/dense_6/Tensordot/ReshapeReshape.multi_head_att/dense_6/Tensordot/transpose:y:0/multi_head_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_6/Tensordot/Reshape?
'multi_head_att/dense_6/Tensordot/MatMulMatMul1multi_head_att/dense_6/Tensordot/Reshape:output:07multi_head_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_6/Tensordot/MatMul?
(multi_head_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_6/Tensordot/Const_2?
.multi_head_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/concat_1/axis?
)multi_head_att/dense_6/Tensordot/concat_1ConcatV22multi_head_att/dense_6/Tensordot/GatherV2:output:01multi_head_att/dense_6/Tensordot/Const_2:output:07multi_head_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/concat_1?
 multi_head_att/dense_6/TensordotReshape1multi_head_att/dense_6/Tensordot/MatMul:product:02multi_head_att/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_6/Tensordot?
multi_head_att/MatMulBatchMatMulV2'multi_head_att/dense/Tensordot:output:0)multi_head_att/dense_3/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMuly
multi_head_att/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv/y?
multi_head_att/truedivRealDivmulti_head_att/MatMul:output:0!multi_head_att/truediv/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv?
multi_head_att/SoftmaxSoftmaxmulti_head_att/truediv:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax?
multi_head_att/MatMul_1BatchMatMulV2 multi_head_att/Softmax:softmax:0)multi_head_att/dense_6/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_1?
/multi_head_att/dense_1/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_1/Tensordot/ReadVariableOp?
%multi_head_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_1/Tensordot/axes?
%multi_head_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_1/Tensordot/free?
&multi_head_att/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/Shape?
.multi_head_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/GatherV2/axis?
)multi_head_att/dense_1/Tensordot/GatherV2GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/free:output:07multi_head_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/GatherV2?
0multi_head_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_1/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_1/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/axes:output:09multi_head_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_1/Tensordot/GatherV2_1?
&multi_head_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_1/Tensordot/Const?
%multi_head_att/dense_1/Tensordot/ProdProd2multi_head_att/dense_1/Tensordot/GatherV2:output:0/multi_head_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_1/Tensordot/Prod?
(multi_head_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_1/Tensordot/Const_1?
'multi_head_att/dense_1/Tensordot/Prod_1Prod4multi_head_att/dense_1/Tensordot/GatherV2_1:output:01multi_head_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_1/Tensordot/Prod_1?
,multi_head_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_1/Tensordot/concat/axis?
'multi_head_att/dense_1/Tensordot/concatConcatV2.multi_head_att/dense_1/Tensordot/free:output:0.multi_head_att/dense_1/Tensordot/axes:output:05multi_head_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_1/Tensordot/concat?
&multi_head_att/dense_1/Tensordot/stackPack.multi_head_att/dense_1/Tensordot/Prod:output:00multi_head_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/stack?
*multi_head_att/dense_1/Tensordot/transpose	Transposeinputs0multi_head_att/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_1/Tensordot/transpose?
(multi_head_att/dense_1/Tensordot/ReshapeReshape.multi_head_att/dense_1/Tensordot/transpose:y:0/multi_head_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_1/Tensordot/Reshape?
'multi_head_att/dense_1/Tensordot/MatMulMatMul1multi_head_att/dense_1/Tensordot/Reshape:output:07multi_head_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_1/Tensordot/MatMul?
(multi_head_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_1/Tensordot/Const_2?
.multi_head_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/concat_1/axis?
)multi_head_att/dense_1/Tensordot/concat_1ConcatV22multi_head_att/dense_1/Tensordot/GatherV2:output:01multi_head_att/dense_1/Tensordot/Const_2:output:07multi_head_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/concat_1?
 multi_head_att/dense_1/TensordotReshape1multi_head_att/dense_1/Tensordot/MatMul:product:02multi_head_att/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_1/Tensordot?
/multi_head_att/dense_4/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_4/Tensordot/ReadVariableOp?
%multi_head_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_4/Tensordot/axes?
%multi_head_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_4/Tensordot/free?
&multi_head_att/dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/Shape?
.multi_head_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/GatherV2/axis?
)multi_head_att/dense_4/Tensordot/GatherV2GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/free:output:07multi_head_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/GatherV2?
0multi_head_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_4/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_4/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/axes:output:09multi_head_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_4/Tensordot/GatherV2_1?
&multi_head_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_4/Tensordot/Const?
%multi_head_att/dense_4/Tensordot/ProdProd2multi_head_att/dense_4/Tensordot/GatherV2:output:0/multi_head_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_4/Tensordot/Prod?
(multi_head_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_4/Tensordot/Const_1?
'multi_head_att/dense_4/Tensordot/Prod_1Prod4multi_head_att/dense_4/Tensordot/GatherV2_1:output:01multi_head_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_4/Tensordot/Prod_1?
,multi_head_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_4/Tensordot/concat/axis?
'multi_head_att/dense_4/Tensordot/concatConcatV2.multi_head_att/dense_4/Tensordot/free:output:0.multi_head_att/dense_4/Tensordot/axes:output:05multi_head_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_4/Tensordot/concat?
&multi_head_att/dense_4/Tensordot/stackPack.multi_head_att/dense_4/Tensordot/Prod:output:00multi_head_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/stack?
*multi_head_att/dense_4/Tensordot/transpose	Transposeinputs0multi_head_att/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_4/Tensordot/transpose?
(multi_head_att/dense_4/Tensordot/ReshapeReshape.multi_head_att/dense_4/Tensordot/transpose:y:0/multi_head_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_4/Tensordot/Reshape?
'multi_head_att/dense_4/Tensordot/MatMulMatMul1multi_head_att/dense_4/Tensordot/Reshape:output:07multi_head_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_4/Tensordot/MatMul?
(multi_head_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_4/Tensordot/Const_2?
.multi_head_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/concat_1/axis?
)multi_head_att/dense_4/Tensordot/concat_1ConcatV22multi_head_att/dense_4/Tensordot/GatherV2:output:01multi_head_att/dense_4/Tensordot/Const_2:output:07multi_head_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/concat_1?
 multi_head_att/dense_4/TensordotReshape1multi_head_att/dense_4/Tensordot/MatMul:product:02multi_head_att/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_4/Tensordot?
/multi_head_att/dense_7/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_7_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_7/Tensordot/ReadVariableOp?
%multi_head_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_7/Tensordot/axes?
%multi_head_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_7/Tensordot/free?
&multi_head_att/dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/Shape?
.multi_head_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/GatherV2/axis?
)multi_head_att/dense_7/Tensordot/GatherV2GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/free:output:07multi_head_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/GatherV2?
0multi_head_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_7/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_7/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/axes:output:09multi_head_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_7/Tensordot/GatherV2_1?
&multi_head_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_7/Tensordot/Const?
%multi_head_att/dense_7/Tensordot/ProdProd2multi_head_att/dense_7/Tensordot/GatherV2:output:0/multi_head_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_7/Tensordot/Prod?
(multi_head_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_7/Tensordot/Const_1?
'multi_head_att/dense_7/Tensordot/Prod_1Prod4multi_head_att/dense_7/Tensordot/GatherV2_1:output:01multi_head_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_7/Tensordot/Prod_1?
,multi_head_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_7/Tensordot/concat/axis?
'multi_head_att/dense_7/Tensordot/concatConcatV2.multi_head_att/dense_7/Tensordot/free:output:0.multi_head_att/dense_7/Tensordot/axes:output:05multi_head_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_7/Tensordot/concat?
&multi_head_att/dense_7/Tensordot/stackPack.multi_head_att/dense_7/Tensordot/Prod:output:00multi_head_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/stack?
*multi_head_att/dense_7/Tensordot/transpose	Transposeinputs0multi_head_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_7/Tensordot/transpose?
(multi_head_att/dense_7/Tensordot/ReshapeReshape.multi_head_att/dense_7/Tensordot/transpose:y:0/multi_head_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_7/Tensordot/Reshape?
'multi_head_att/dense_7/Tensordot/MatMulMatMul1multi_head_att/dense_7/Tensordot/Reshape:output:07multi_head_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_7/Tensordot/MatMul?
(multi_head_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_7/Tensordot/Const_2?
.multi_head_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/concat_1/axis?
)multi_head_att/dense_7/Tensordot/concat_1ConcatV22multi_head_att/dense_7/Tensordot/GatherV2:output:01multi_head_att/dense_7/Tensordot/Const_2:output:07multi_head_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/concat_1?
 multi_head_att/dense_7/TensordotReshape1multi_head_att/dense_7/Tensordot/MatMul:product:02multi_head_att/dense_7/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_7/Tensordot?
multi_head_att/MatMul_2BatchMatMulV2)multi_head_att/dense_1/Tensordot:output:0)multi_head_att/dense_4/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_2}
multi_head_att/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_1/y?
multi_head_att/truediv_1RealDiv multi_head_att/MatMul_2:output:0#multi_head_att/truediv_1/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_1?
multi_head_att/Softmax_1Softmaxmulti_head_att/truediv_1:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_1?
multi_head_att/MatMul_3BatchMatMulV2"multi_head_att/Softmax_1:softmax:0)multi_head_att/dense_7/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_3?
/multi_head_att/dense_2/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_2/Tensordot/ReadVariableOp?
%multi_head_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_2/Tensordot/axes?
%multi_head_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_2/Tensordot/free?
&multi_head_att/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/Shape?
.multi_head_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/GatherV2/axis?
)multi_head_att/dense_2/Tensordot/GatherV2GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/free:output:07multi_head_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/GatherV2?
0multi_head_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_2/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_2/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/axes:output:09multi_head_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_2/Tensordot/GatherV2_1?
&multi_head_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_2/Tensordot/Const?
%multi_head_att/dense_2/Tensordot/ProdProd2multi_head_att/dense_2/Tensordot/GatherV2:output:0/multi_head_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_2/Tensordot/Prod?
(multi_head_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_2/Tensordot/Const_1?
'multi_head_att/dense_2/Tensordot/Prod_1Prod4multi_head_att/dense_2/Tensordot/GatherV2_1:output:01multi_head_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_2/Tensordot/Prod_1?
,multi_head_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_2/Tensordot/concat/axis?
'multi_head_att/dense_2/Tensordot/concatConcatV2.multi_head_att/dense_2/Tensordot/free:output:0.multi_head_att/dense_2/Tensordot/axes:output:05multi_head_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_2/Tensordot/concat?
&multi_head_att/dense_2/Tensordot/stackPack.multi_head_att/dense_2/Tensordot/Prod:output:00multi_head_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/stack?
*multi_head_att/dense_2/Tensordot/transpose	Transposeinputs0multi_head_att/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_2/Tensordot/transpose?
(multi_head_att/dense_2/Tensordot/ReshapeReshape.multi_head_att/dense_2/Tensordot/transpose:y:0/multi_head_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_2/Tensordot/Reshape?
'multi_head_att/dense_2/Tensordot/MatMulMatMul1multi_head_att/dense_2/Tensordot/Reshape:output:07multi_head_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_2/Tensordot/MatMul?
(multi_head_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_2/Tensordot/Const_2?
.multi_head_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/concat_1/axis?
)multi_head_att/dense_2/Tensordot/concat_1ConcatV22multi_head_att/dense_2/Tensordot/GatherV2:output:01multi_head_att/dense_2/Tensordot/Const_2:output:07multi_head_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/concat_1?
 multi_head_att/dense_2/TensordotReshape1multi_head_att/dense_2/Tensordot/MatMul:product:02multi_head_att/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_2/Tensordot?
/multi_head_att/dense_5/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_5_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_5/Tensordot/ReadVariableOp?
%multi_head_att/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_5/Tensordot/axes?
%multi_head_att/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_5/Tensordot/free?
&multi_head_att/dense_5/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/Shape?
.multi_head_att/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/GatherV2/axis?
)multi_head_att/dense_5/Tensordot/GatherV2GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/free:output:07multi_head_att/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/GatherV2?
0multi_head_att/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_5/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_5/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/axes:output:09multi_head_att/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_5/Tensordot/GatherV2_1?
&multi_head_att/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_5/Tensordot/Const?
%multi_head_att/dense_5/Tensordot/ProdProd2multi_head_att/dense_5/Tensordot/GatherV2:output:0/multi_head_att/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_5/Tensordot/Prod?
(multi_head_att/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_5/Tensordot/Const_1?
'multi_head_att/dense_5/Tensordot/Prod_1Prod4multi_head_att/dense_5/Tensordot/GatherV2_1:output:01multi_head_att/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_5/Tensordot/Prod_1?
,multi_head_att/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_5/Tensordot/concat/axis?
'multi_head_att/dense_5/Tensordot/concatConcatV2.multi_head_att/dense_5/Tensordot/free:output:0.multi_head_att/dense_5/Tensordot/axes:output:05multi_head_att/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_5/Tensordot/concat?
&multi_head_att/dense_5/Tensordot/stackPack.multi_head_att/dense_5/Tensordot/Prod:output:00multi_head_att/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/stack?
*multi_head_att/dense_5/Tensordot/transpose	Transposeinputs0multi_head_att/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_5/Tensordot/transpose?
(multi_head_att/dense_5/Tensordot/ReshapeReshape.multi_head_att/dense_5/Tensordot/transpose:y:0/multi_head_att/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_5/Tensordot/Reshape?
'multi_head_att/dense_5/Tensordot/MatMulMatMul1multi_head_att/dense_5/Tensordot/Reshape:output:07multi_head_att/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_5/Tensordot/MatMul?
(multi_head_att/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_5/Tensordot/Const_2?
.multi_head_att/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/concat_1/axis?
)multi_head_att/dense_5/Tensordot/concat_1ConcatV22multi_head_att/dense_5/Tensordot/GatherV2:output:01multi_head_att/dense_5/Tensordot/Const_2:output:07multi_head_att/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/concat_1?
 multi_head_att/dense_5/TensordotReshape1multi_head_att/dense_5/Tensordot/MatMul:product:02multi_head_att/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_5/Tensordot?
/multi_head_att/dense_8/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_8/Tensordot/ReadVariableOp?
%multi_head_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_8/Tensordot/axes?
%multi_head_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_8/Tensordot/free?
&multi_head_att/dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/Shape?
.multi_head_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/GatherV2/axis?
)multi_head_att/dense_8/Tensordot/GatherV2GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/free:output:07multi_head_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/GatherV2?
0multi_head_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_8/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_8/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/axes:output:09multi_head_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_8/Tensordot/GatherV2_1?
&multi_head_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_8/Tensordot/Const?
%multi_head_att/dense_8/Tensordot/ProdProd2multi_head_att/dense_8/Tensordot/GatherV2:output:0/multi_head_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_8/Tensordot/Prod?
(multi_head_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_8/Tensordot/Const_1?
'multi_head_att/dense_8/Tensordot/Prod_1Prod4multi_head_att/dense_8/Tensordot/GatherV2_1:output:01multi_head_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_8/Tensordot/Prod_1?
,multi_head_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_8/Tensordot/concat/axis?
'multi_head_att/dense_8/Tensordot/concatConcatV2.multi_head_att/dense_8/Tensordot/free:output:0.multi_head_att/dense_8/Tensordot/axes:output:05multi_head_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_8/Tensordot/concat?
&multi_head_att/dense_8/Tensordot/stackPack.multi_head_att/dense_8/Tensordot/Prod:output:00multi_head_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/stack?
*multi_head_att/dense_8/Tensordot/transpose	Transposeinputs0multi_head_att/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_8/Tensordot/transpose?
(multi_head_att/dense_8/Tensordot/ReshapeReshape.multi_head_att/dense_8/Tensordot/transpose:y:0/multi_head_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_8/Tensordot/Reshape?
'multi_head_att/dense_8/Tensordot/MatMulMatMul1multi_head_att/dense_8/Tensordot/Reshape:output:07multi_head_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_8/Tensordot/MatMul?
(multi_head_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_8/Tensordot/Const_2?
.multi_head_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/concat_1/axis?
)multi_head_att/dense_8/Tensordot/concat_1ConcatV22multi_head_att/dense_8/Tensordot/GatherV2:output:01multi_head_att/dense_8/Tensordot/Const_2:output:07multi_head_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/concat_1?
 multi_head_att/dense_8/TensordotReshape1multi_head_att/dense_8/Tensordot/MatMul:product:02multi_head_att/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_8/Tensordot?
multi_head_att/MatMul_4BatchMatMulV2)multi_head_att/dense_2/Tensordot:output:0)multi_head_att/dense_5/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_4}
multi_head_att/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_2/y?
multi_head_att/truediv_2RealDiv multi_head_att/MatMul_4:output:0#multi_head_att/truediv_2/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_2?
multi_head_att/Softmax_2Softmaxmulti_head_att/truediv_2:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_2?
multi_head_att/MatMul_5BatchMatMulV2"multi_head_att/Softmax_2:softmax:0)multi_head_att/dense_8/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_5z
multi_head_att/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
multi_head_att/concat/axis?
multi_head_att/concatConcatV2 multi_head_att/MatMul_1:output:0 multi_head_att/MatMul_3:output:0 multi_head_att/MatMul_5:output:0#multi_head_att/concat/axis:output:0*
N*
T0*,
_output_shapes
:??????????x2
multi_head_att/concat?
/multi_head_att/dense_9/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

:x(*
dtype021
/multi_head_att/dense_9/Tensordot/ReadVariableOp?
%multi_head_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_9/Tensordot/axes?
%multi_head_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_9/Tensordot/free?
&multi_head_att/dense_9/Tensordot/ShapeShapemulti_head_att/concat:output:0*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/Shape?
.multi_head_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/GatherV2/axis?
)multi_head_att/dense_9/Tensordot/GatherV2GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/free:output:07multi_head_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/GatherV2?
0multi_head_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_9/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_9/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/axes:output:09multi_head_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_9/Tensordot/GatherV2_1?
&multi_head_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_9/Tensordot/Const?
%multi_head_att/dense_9/Tensordot/ProdProd2multi_head_att/dense_9/Tensordot/GatherV2:output:0/multi_head_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_9/Tensordot/Prod?
(multi_head_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_9/Tensordot/Const_1?
'multi_head_att/dense_9/Tensordot/Prod_1Prod4multi_head_att/dense_9/Tensordot/GatherV2_1:output:01multi_head_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_9/Tensordot/Prod_1?
,multi_head_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_9/Tensordot/concat/axis?
'multi_head_att/dense_9/Tensordot/concatConcatV2.multi_head_att/dense_9/Tensordot/free:output:0.multi_head_att/dense_9/Tensordot/axes:output:05multi_head_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_9/Tensordot/concat?
&multi_head_att/dense_9/Tensordot/stackPack.multi_head_att/dense_9/Tensordot/Prod:output:00multi_head_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/stack?
*multi_head_att/dense_9/Tensordot/transpose	Transposemulti_head_att/concat:output:00multi_head_att/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????x2,
*multi_head_att/dense_9/Tensordot/transpose?
(multi_head_att/dense_9/Tensordot/ReshapeReshape.multi_head_att/dense_9/Tensordot/transpose:y:0/multi_head_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_9/Tensordot/Reshape?
'multi_head_att/dense_9/Tensordot/MatMulMatMul1multi_head_att/dense_9/Tensordot/Reshape:output:07multi_head_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_9/Tensordot/MatMul?
(multi_head_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_9/Tensordot/Const_2?
.multi_head_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/concat_1/axis?
)multi_head_att/dense_9/Tensordot/concat_1ConcatV22multi_head_att/dense_9/Tensordot/GatherV2:output:01multi_head_att/dense_9/Tensordot/Const_2:output:07multi_head_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/concat_1?
 multi_head_att/dense_9/TensordotReshape1multi_head_att/dense_9/Tensordot/MatMul:product:02multi_head_att/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_9/Tensordots
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout/dropout/Const?
dropout/dropout/MulMul)multi_head_att/dense_9/Tensordot:output:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:??????????(2
dropout/dropout/Mul?
dropout/dropout/ShapeShape)multi_head_att/dense_9/Tensordot:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????(*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????(2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????(2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????(2
dropout/dropout/Mul_1m
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????(2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:??????????2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/add_1?
,sequential/dense_10/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02.
,sequential/dense_10/Tensordot/ReadVariableOp?
"sequential/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_10/Tensordot/axes?
"sequential/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_10/Tensordot/free?
#sequential/dense_10/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/Shape?
+sequential/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/GatherV2/axis?
&sequential/dense_10/Tensordot/GatherV2GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/free:output:04sequential/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/GatherV2?
-sequential/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_10/Tensordot/GatherV2_1/axis?
(sequential/dense_10/Tensordot/GatherV2_1GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/axes:output:06sequential/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_10/Tensordot/GatherV2_1?
#sequential/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_10/Tensordot/Const?
"sequential/dense_10/Tensordot/ProdProd/sequential/dense_10/Tensordot/GatherV2:output:0,sequential/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_10/Tensordot/Prod?
%sequential/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_1?
$sequential/dense_10/Tensordot/Prod_1Prod1sequential/dense_10/Tensordot/GatherV2_1:output:0.sequential/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_10/Tensordot/Prod_1?
)sequential/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_10/Tensordot/concat/axis?
$sequential/dense_10/Tensordot/concatConcatV2+sequential/dense_10/Tensordot/free:output:0+sequential/dense_10/Tensordot/axes:output:02sequential/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_10/Tensordot/concat?
#sequential/dense_10/Tensordot/stackPack+sequential/dense_10/Tensordot/Prod:output:0-sequential/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/stack?
'sequential/dense_10/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0-sequential/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2)
'sequential/dense_10/Tensordot/transpose?
%sequential/dense_10/Tensordot/ReshapeReshape+sequential/dense_10/Tensordot/transpose:y:0,sequential/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_10/Tensordot/Reshape?
$sequential/dense_10/Tensordot/MatMulMatMul.sequential/dense_10/Tensordot/Reshape:output:04sequential/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$sequential/dense_10/Tensordot/MatMul?
%sequential/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_2?
+sequential/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/concat_1/axis?
&sequential/dense_10/Tensordot/concat_1ConcatV2/sequential/dense_10/Tensordot/GatherV2:output:0.sequential/dense_10/Tensordot/Const_2:output:04sequential/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/concat_1?
sequential/dense_10/TensordotReshape.sequential/dense_10/Tensordot/MatMul:product:0/sequential/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Tensordot?
*sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/dense_10/BiasAdd/ReadVariableOp?
sequential/dense_10/BiasAddBiasAdd&sequential/dense_10/Tensordot:output:02sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/BiasAdd?
sequential/dense_10/ReluRelu$sequential/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Relu?
,sequential/dense_11/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02.
,sequential/dense_11/Tensordot/ReadVariableOp?
"sequential/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_11/Tensordot/axes?
"sequential/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_11/Tensordot/free?
#sequential/dense_11/Tensordot/ShapeShape&sequential/dense_10/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/Shape?
+sequential/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/GatherV2/axis?
&sequential/dense_11/Tensordot/GatherV2GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/free:output:04sequential/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/GatherV2?
-sequential/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_11/Tensordot/GatherV2_1/axis?
(sequential/dense_11/Tensordot/GatherV2_1GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/axes:output:06sequential/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_11/Tensordot/GatherV2_1?
#sequential/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_11/Tensordot/Const?
"sequential/dense_11/Tensordot/ProdProd/sequential/dense_11/Tensordot/GatherV2:output:0,sequential/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_11/Tensordot/Prod?
%sequential/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_11/Tensordot/Const_1?
$sequential/dense_11/Tensordot/Prod_1Prod1sequential/dense_11/Tensordot/GatherV2_1:output:0.sequential/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_11/Tensordot/Prod_1?
)sequential/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_11/Tensordot/concat/axis?
$sequential/dense_11/Tensordot/concatConcatV2+sequential/dense_11/Tensordot/free:output:0+sequential/dense_11/Tensordot/axes:output:02sequential/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_11/Tensordot/concat?
#sequential/dense_11/Tensordot/stackPack+sequential/dense_11/Tensordot/Prod:output:0-sequential/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/stack?
'sequential/dense_11/Tensordot/transpose	Transpose&sequential/dense_10/Relu:activations:0-sequential/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2)
'sequential/dense_11/Tensordot/transpose?
%sequential/dense_11/Tensordot/ReshapeReshape+sequential/dense_11/Tensordot/transpose:y:0,sequential/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_11/Tensordot/Reshape?
$sequential/dense_11/Tensordot/MatMulMatMul.sequential/dense_11/Tensordot/Reshape:output:04sequential/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2&
$sequential/dense_11/Tensordot/MatMul?
%sequential/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2'
%sequential/dense_11/Tensordot/Const_2?
+sequential/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/concat_1/axis?
&sequential/dense_11/Tensordot/concat_1ConcatV2/sequential/dense_11/Tensordot/GatherV2:output:0.sequential/dense_11/Tensordot/Const_2:output:04sequential/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/concat_1?
sequential/dense_11/TensordotReshape.sequential/dense_11/Tensordot/MatMul:product:0/sequential/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/Tensordot?
*sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*sequential/dense_11/BiasAdd/ReadVariableOp?
sequential/dense_11/BiasAddBiasAdd&sequential/dense_11/Tensordot:output:02sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul$sequential/dense_11/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:??????????(2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape$sequential/dense_11/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????(*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????(2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????(2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????(2
dropout_1/dropout/Mul_1?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????(2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:??????????2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp.^multi_head_att/dense/Tensordot/ReadVariableOp0^multi_head_att/dense_1/Tensordot/ReadVariableOp0^multi_head_att/dense_2/Tensordot/ReadVariableOp0^multi_head_att/dense_3/Tensordot/ReadVariableOp0^multi_head_att/dense_4/Tensordot/ReadVariableOp0^multi_head_att/dense_5/Tensordot/ReadVariableOp0^multi_head_att/dense_6/Tensordot/ReadVariableOp0^multi_head_att/dense_7/Tensordot/ReadVariableOp0^multi_head_att/dense_8/Tensordot/ReadVariableOp0^multi_head_att/dense_9/Tensordot/ReadVariableOp+^sequential/dense_10/BiasAdd/ReadVariableOp-^sequential/dense_10/Tensordot/ReadVariableOp+^sequential/dense_11/BiasAdd/ReadVariableOp-^sequential/dense_11/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????(: : : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2^
-multi_head_att/dense/Tensordot/ReadVariableOp-multi_head_att/dense/Tensordot/ReadVariableOp2b
/multi_head_att/dense_1/Tensordot/ReadVariableOp/multi_head_att/dense_1/Tensordot/ReadVariableOp2b
/multi_head_att/dense_2/Tensordot/ReadVariableOp/multi_head_att/dense_2/Tensordot/ReadVariableOp2b
/multi_head_att/dense_3/Tensordot/ReadVariableOp/multi_head_att/dense_3/Tensordot/ReadVariableOp2b
/multi_head_att/dense_4/Tensordot/ReadVariableOp/multi_head_att/dense_4/Tensordot/ReadVariableOp2b
/multi_head_att/dense_5/Tensordot/ReadVariableOp/multi_head_att/dense_5/Tensordot/ReadVariableOp2b
/multi_head_att/dense_6/Tensordot/ReadVariableOp/multi_head_att/dense_6/Tensordot/ReadVariableOp2b
/multi_head_att/dense_7/Tensordot/ReadVariableOp/multi_head_att/dense_7/Tensordot/ReadVariableOp2b
/multi_head_att/dense_8/Tensordot/ReadVariableOp/multi_head_att/dense_8/Tensordot/ReadVariableOp2b
/multi_head_att/dense_9/Tensordot/ReadVariableOp/multi_head_att/dense_9/Tensordot/ReadVariableOp2X
*sequential/dense_10/BiasAdd/ReadVariableOp*sequential/dense_10/BiasAdd/ReadVariableOp2\
,sequential/dense_10/Tensordot/ReadVariableOp,sequential/dense_10/Tensordot/ReadVariableOp2X
*sequential/dense_11/BiasAdd/ReadVariableOp*sequential/dense_11/BiasAdd/ReadVariableOp2\
,sequential/dense_11/Tensordot/ReadVariableOp,sequential/dense_11/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
??
?
K__inference_transformer_block_layer_call_and_return_conditional_losses_7969

inputsH
6multi_head_att_dense_tensordot_readvariableop_resource:((J
8multi_head_att_dense_3_tensordot_readvariableop_resource:((J
8multi_head_att_dense_6_tensordot_readvariableop_resource:((J
8multi_head_att_dense_1_tensordot_readvariableop_resource:((J
8multi_head_att_dense_4_tensordot_readvariableop_resource:((J
8multi_head_att_dense_7_tensordot_readvariableop_resource:((J
8multi_head_att_dense_2_tensordot_readvariableop_resource:((J
8multi_head_att_dense_5_tensordot_readvariableop_resource:((J
8multi_head_att_dense_8_tensordot_readvariableop_resource:((J
8multi_head_att_dense_9_tensordot_readvariableop_resource:x(G
9layer_normalization_batchnorm_mul_readvariableop_resource:(C
5layer_normalization_batchnorm_readvariableop_resource:(G
5sequential_dense_10_tensordot_readvariableop_resource:( A
3sequential_dense_10_biasadd_readvariableop_resource: G
5sequential_dense_11_tensordot_readvariableop_resource: (A
3sequential_dense_11_biasadd_readvariableop_resource:(I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:(E
7layer_normalization_1_batchnorm_readvariableop_resource:(
identity??,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?-multi_head_att/dense/Tensordot/ReadVariableOp?/multi_head_att/dense_1/Tensordot/ReadVariableOp?/multi_head_att/dense_2/Tensordot/ReadVariableOp?/multi_head_att/dense_3/Tensordot/ReadVariableOp?/multi_head_att/dense_4/Tensordot/ReadVariableOp?/multi_head_att/dense_5/Tensordot/ReadVariableOp?/multi_head_att/dense_6/Tensordot/ReadVariableOp?/multi_head_att/dense_7/Tensordot/ReadVariableOp?/multi_head_att/dense_8/Tensordot/ReadVariableOp?/multi_head_att/dense_9/Tensordot/ReadVariableOp?*sequential/dense_10/BiasAdd/ReadVariableOp?,sequential/dense_10/Tensordot/ReadVariableOp?*sequential/dense_11/BiasAdd/ReadVariableOp?,sequential/dense_11/Tensordot/ReadVariableOpb
multi_head_att/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_att/Shape?
"multi_head_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"multi_head_att/strided_slice/stack?
$multi_head_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_1?
$multi_head_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$multi_head_att/strided_slice/stack_2?
multi_head_att/strided_sliceStridedSlicemulti_head_att/Shape:output:0+multi_head_att/strided_slice/stack:output:0-multi_head_att/strided_slice/stack_1:output:0-multi_head_att/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
multi_head_att/strided_slice?
-multi_head_att/dense/Tensordot/ReadVariableOpReadVariableOp6multi_head_att_dense_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype02/
-multi_head_att/dense/Tensordot/ReadVariableOp?
#multi_head_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#multi_head_att/dense/Tensordot/axes?
#multi_head_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#multi_head_att/dense/Tensordot/free?
$multi_head_att/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/Shape?
,multi_head_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/GatherV2/axis?
'multi_head_att/dense/Tensordot/GatherV2GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/free:output:05multi_head_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/GatherV2?
.multi_head_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense/Tensordot/GatherV2_1/axis?
)multi_head_att/dense/Tensordot/GatherV2_1GatherV2-multi_head_att/dense/Tensordot/Shape:output:0,multi_head_att/dense/Tensordot/axes:output:07multi_head_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense/Tensordot/GatherV2_1?
$multi_head_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$multi_head_att/dense/Tensordot/Const?
#multi_head_att/dense/Tensordot/ProdProd0multi_head_att/dense/Tensordot/GatherV2:output:0-multi_head_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#multi_head_att/dense/Tensordot/Prod?
&multi_head_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense/Tensordot/Const_1?
%multi_head_att/dense/Tensordot/Prod_1Prod2multi_head_att/dense/Tensordot/GatherV2_1:output:0/multi_head_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense/Tensordot/Prod_1?
*multi_head_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*multi_head_att/dense/Tensordot/concat/axis?
%multi_head_att/dense/Tensordot/concatConcatV2,multi_head_att/dense/Tensordot/free:output:0,multi_head_att/dense/Tensordot/axes:output:03multi_head_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%multi_head_att/dense/Tensordot/concat?
$multi_head_att/dense/Tensordot/stackPack,multi_head_att/dense/Tensordot/Prod:output:0.multi_head_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_att/dense/Tensordot/stack?
(multi_head_att/dense/Tensordot/transpose	Transposeinputs.multi_head_att/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2*
(multi_head_att/dense/Tensordot/transpose?
&multi_head_att/dense/Tensordot/ReshapeReshape,multi_head_att/dense/Tensordot/transpose:y:0-multi_head_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&multi_head_att/dense/Tensordot/Reshape?
%multi_head_att/dense/Tensordot/MatMulMatMul/multi_head_att/dense/Tensordot/Reshape:output:05multi_head_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2'
%multi_head_att/dense/Tensordot/MatMul?
&multi_head_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2(
&multi_head_att/dense/Tensordot/Const_2?
,multi_head_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense/Tensordot/concat_1/axis?
'multi_head_att/dense/Tensordot/concat_1ConcatV20multi_head_att/dense/Tensordot/GatherV2:output:0/multi_head_att/dense/Tensordot/Const_2:output:05multi_head_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense/Tensordot/concat_1?
multi_head_att/dense/TensordotReshape/multi_head_att/dense/Tensordot/MatMul:product:00multi_head_att/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2 
multi_head_att/dense/Tensordot?
/multi_head_att/dense_3/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_3/Tensordot/ReadVariableOp?
%multi_head_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_3/Tensordot/axes?
%multi_head_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_3/Tensordot/free?
&multi_head_att/dense_3/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/Shape?
.multi_head_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/GatherV2/axis?
)multi_head_att/dense_3/Tensordot/GatherV2GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/free:output:07multi_head_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/GatherV2?
0multi_head_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_3/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_3/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_3/Tensordot/Shape:output:0.multi_head_att/dense_3/Tensordot/axes:output:09multi_head_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_3/Tensordot/GatherV2_1?
&multi_head_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_3/Tensordot/Const?
%multi_head_att/dense_3/Tensordot/ProdProd2multi_head_att/dense_3/Tensordot/GatherV2:output:0/multi_head_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_3/Tensordot/Prod?
(multi_head_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_3/Tensordot/Const_1?
'multi_head_att/dense_3/Tensordot/Prod_1Prod4multi_head_att/dense_3/Tensordot/GatherV2_1:output:01multi_head_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_3/Tensordot/Prod_1?
,multi_head_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_3/Tensordot/concat/axis?
'multi_head_att/dense_3/Tensordot/concatConcatV2.multi_head_att/dense_3/Tensordot/free:output:0.multi_head_att/dense_3/Tensordot/axes:output:05multi_head_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_3/Tensordot/concat?
&multi_head_att/dense_3/Tensordot/stackPack.multi_head_att/dense_3/Tensordot/Prod:output:00multi_head_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_3/Tensordot/stack?
*multi_head_att/dense_3/Tensordot/transpose	Transposeinputs0multi_head_att/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_3/Tensordot/transpose?
(multi_head_att/dense_3/Tensordot/ReshapeReshape.multi_head_att/dense_3/Tensordot/transpose:y:0/multi_head_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_3/Tensordot/Reshape?
'multi_head_att/dense_3/Tensordot/MatMulMatMul1multi_head_att/dense_3/Tensordot/Reshape:output:07multi_head_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_3/Tensordot/MatMul?
(multi_head_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_3/Tensordot/Const_2?
.multi_head_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_3/Tensordot/concat_1/axis?
)multi_head_att/dense_3/Tensordot/concat_1ConcatV22multi_head_att/dense_3/Tensordot/GatherV2:output:01multi_head_att/dense_3/Tensordot/Const_2:output:07multi_head_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_3/Tensordot/concat_1?
 multi_head_att/dense_3/TensordotReshape1multi_head_att/dense_3/Tensordot/MatMul:product:02multi_head_att/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_3/Tensordot?
/multi_head_att/dense_6/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_6/Tensordot/ReadVariableOp?
%multi_head_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_6/Tensordot/axes?
%multi_head_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_6/Tensordot/free?
&multi_head_att/dense_6/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/Shape?
.multi_head_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/GatherV2/axis?
)multi_head_att/dense_6/Tensordot/GatherV2GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/free:output:07multi_head_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/GatherV2?
0multi_head_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_6/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_6/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_6/Tensordot/Shape:output:0.multi_head_att/dense_6/Tensordot/axes:output:09multi_head_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_6/Tensordot/GatherV2_1?
&multi_head_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_6/Tensordot/Const?
%multi_head_att/dense_6/Tensordot/ProdProd2multi_head_att/dense_6/Tensordot/GatherV2:output:0/multi_head_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_6/Tensordot/Prod?
(multi_head_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_6/Tensordot/Const_1?
'multi_head_att/dense_6/Tensordot/Prod_1Prod4multi_head_att/dense_6/Tensordot/GatherV2_1:output:01multi_head_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_6/Tensordot/Prod_1?
,multi_head_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_6/Tensordot/concat/axis?
'multi_head_att/dense_6/Tensordot/concatConcatV2.multi_head_att/dense_6/Tensordot/free:output:0.multi_head_att/dense_6/Tensordot/axes:output:05multi_head_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_6/Tensordot/concat?
&multi_head_att/dense_6/Tensordot/stackPack.multi_head_att/dense_6/Tensordot/Prod:output:00multi_head_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_6/Tensordot/stack?
*multi_head_att/dense_6/Tensordot/transpose	Transposeinputs0multi_head_att/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_6/Tensordot/transpose?
(multi_head_att/dense_6/Tensordot/ReshapeReshape.multi_head_att/dense_6/Tensordot/transpose:y:0/multi_head_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_6/Tensordot/Reshape?
'multi_head_att/dense_6/Tensordot/MatMulMatMul1multi_head_att/dense_6/Tensordot/Reshape:output:07multi_head_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_6/Tensordot/MatMul?
(multi_head_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_6/Tensordot/Const_2?
.multi_head_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_6/Tensordot/concat_1/axis?
)multi_head_att/dense_6/Tensordot/concat_1ConcatV22multi_head_att/dense_6/Tensordot/GatherV2:output:01multi_head_att/dense_6/Tensordot/Const_2:output:07multi_head_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_6/Tensordot/concat_1?
 multi_head_att/dense_6/TensordotReshape1multi_head_att/dense_6/Tensordot/MatMul:product:02multi_head_att/dense_6/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_6/Tensordot?
multi_head_att/MatMulBatchMatMulV2'multi_head_att/dense/Tensordot:output:0)multi_head_att/dense_3/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMuly
multi_head_att/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv/y?
multi_head_att/truedivRealDivmulti_head_att/MatMul:output:0!multi_head_att/truediv/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv?
multi_head_att/SoftmaxSoftmaxmulti_head_att/truediv:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax?
multi_head_att/MatMul_1BatchMatMulV2 multi_head_att/Softmax:softmax:0)multi_head_att/dense_6/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_1?
/multi_head_att/dense_1/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_1/Tensordot/ReadVariableOp?
%multi_head_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_1/Tensordot/axes?
%multi_head_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_1/Tensordot/free?
&multi_head_att/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/Shape?
.multi_head_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/GatherV2/axis?
)multi_head_att/dense_1/Tensordot/GatherV2GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/free:output:07multi_head_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/GatherV2?
0multi_head_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_1/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_1/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_1/Tensordot/Shape:output:0.multi_head_att/dense_1/Tensordot/axes:output:09multi_head_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_1/Tensordot/GatherV2_1?
&multi_head_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_1/Tensordot/Const?
%multi_head_att/dense_1/Tensordot/ProdProd2multi_head_att/dense_1/Tensordot/GatherV2:output:0/multi_head_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_1/Tensordot/Prod?
(multi_head_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_1/Tensordot/Const_1?
'multi_head_att/dense_1/Tensordot/Prod_1Prod4multi_head_att/dense_1/Tensordot/GatherV2_1:output:01multi_head_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_1/Tensordot/Prod_1?
,multi_head_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_1/Tensordot/concat/axis?
'multi_head_att/dense_1/Tensordot/concatConcatV2.multi_head_att/dense_1/Tensordot/free:output:0.multi_head_att/dense_1/Tensordot/axes:output:05multi_head_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_1/Tensordot/concat?
&multi_head_att/dense_1/Tensordot/stackPack.multi_head_att/dense_1/Tensordot/Prod:output:00multi_head_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_1/Tensordot/stack?
*multi_head_att/dense_1/Tensordot/transpose	Transposeinputs0multi_head_att/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_1/Tensordot/transpose?
(multi_head_att/dense_1/Tensordot/ReshapeReshape.multi_head_att/dense_1/Tensordot/transpose:y:0/multi_head_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_1/Tensordot/Reshape?
'multi_head_att/dense_1/Tensordot/MatMulMatMul1multi_head_att/dense_1/Tensordot/Reshape:output:07multi_head_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_1/Tensordot/MatMul?
(multi_head_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_1/Tensordot/Const_2?
.multi_head_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_1/Tensordot/concat_1/axis?
)multi_head_att/dense_1/Tensordot/concat_1ConcatV22multi_head_att/dense_1/Tensordot/GatherV2:output:01multi_head_att/dense_1/Tensordot/Const_2:output:07multi_head_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_1/Tensordot/concat_1?
 multi_head_att/dense_1/TensordotReshape1multi_head_att/dense_1/Tensordot/MatMul:product:02multi_head_att/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_1/Tensordot?
/multi_head_att/dense_4/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_4/Tensordot/ReadVariableOp?
%multi_head_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_4/Tensordot/axes?
%multi_head_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_4/Tensordot/free?
&multi_head_att/dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/Shape?
.multi_head_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/GatherV2/axis?
)multi_head_att/dense_4/Tensordot/GatherV2GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/free:output:07multi_head_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/GatherV2?
0multi_head_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_4/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_4/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_4/Tensordot/Shape:output:0.multi_head_att/dense_4/Tensordot/axes:output:09multi_head_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_4/Tensordot/GatherV2_1?
&multi_head_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_4/Tensordot/Const?
%multi_head_att/dense_4/Tensordot/ProdProd2multi_head_att/dense_4/Tensordot/GatherV2:output:0/multi_head_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_4/Tensordot/Prod?
(multi_head_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_4/Tensordot/Const_1?
'multi_head_att/dense_4/Tensordot/Prod_1Prod4multi_head_att/dense_4/Tensordot/GatherV2_1:output:01multi_head_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_4/Tensordot/Prod_1?
,multi_head_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_4/Tensordot/concat/axis?
'multi_head_att/dense_4/Tensordot/concatConcatV2.multi_head_att/dense_4/Tensordot/free:output:0.multi_head_att/dense_4/Tensordot/axes:output:05multi_head_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_4/Tensordot/concat?
&multi_head_att/dense_4/Tensordot/stackPack.multi_head_att/dense_4/Tensordot/Prod:output:00multi_head_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_4/Tensordot/stack?
*multi_head_att/dense_4/Tensordot/transpose	Transposeinputs0multi_head_att/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_4/Tensordot/transpose?
(multi_head_att/dense_4/Tensordot/ReshapeReshape.multi_head_att/dense_4/Tensordot/transpose:y:0/multi_head_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_4/Tensordot/Reshape?
'multi_head_att/dense_4/Tensordot/MatMulMatMul1multi_head_att/dense_4/Tensordot/Reshape:output:07multi_head_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_4/Tensordot/MatMul?
(multi_head_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_4/Tensordot/Const_2?
.multi_head_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_4/Tensordot/concat_1/axis?
)multi_head_att/dense_4/Tensordot/concat_1ConcatV22multi_head_att/dense_4/Tensordot/GatherV2:output:01multi_head_att/dense_4/Tensordot/Const_2:output:07multi_head_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_4/Tensordot/concat_1?
 multi_head_att/dense_4/TensordotReshape1multi_head_att/dense_4/Tensordot/MatMul:product:02multi_head_att/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_4/Tensordot?
/multi_head_att/dense_7/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_7_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_7/Tensordot/ReadVariableOp?
%multi_head_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_7/Tensordot/axes?
%multi_head_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_7/Tensordot/free?
&multi_head_att/dense_7/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/Shape?
.multi_head_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/GatherV2/axis?
)multi_head_att/dense_7/Tensordot/GatherV2GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/free:output:07multi_head_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/GatherV2?
0multi_head_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_7/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_7/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_7/Tensordot/Shape:output:0.multi_head_att/dense_7/Tensordot/axes:output:09multi_head_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_7/Tensordot/GatherV2_1?
&multi_head_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_7/Tensordot/Const?
%multi_head_att/dense_7/Tensordot/ProdProd2multi_head_att/dense_7/Tensordot/GatherV2:output:0/multi_head_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_7/Tensordot/Prod?
(multi_head_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_7/Tensordot/Const_1?
'multi_head_att/dense_7/Tensordot/Prod_1Prod4multi_head_att/dense_7/Tensordot/GatherV2_1:output:01multi_head_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_7/Tensordot/Prod_1?
,multi_head_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_7/Tensordot/concat/axis?
'multi_head_att/dense_7/Tensordot/concatConcatV2.multi_head_att/dense_7/Tensordot/free:output:0.multi_head_att/dense_7/Tensordot/axes:output:05multi_head_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_7/Tensordot/concat?
&multi_head_att/dense_7/Tensordot/stackPack.multi_head_att/dense_7/Tensordot/Prod:output:00multi_head_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_7/Tensordot/stack?
*multi_head_att/dense_7/Tensordot/transpose	Transposeinputs0multi_head_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_7/Tensordot/transpose?
(multi_head_att/dense_7/Tensordot/ReshapeReshape.multi_head_att/dense_7/Tensordot/transpose:y:0/multi_head_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_7/Tensordot/Reshape?
'multi_head_att/dense_7/Tensordot/MatMulMatMul1multi_head_att/dense_7/Tensordot/Reshape:output:07multi_head_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_7/Tensordot/MatMul?
(multi_head_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_7/Tensordot/Const_2?
.multi_head_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_7/Tensordot/concat_1/axis?
)multi_head_att/dense_7/Tensordot/concat_1ConcatV22multi_head_att/dense_7/Tensordot/GatherV2:output:01multi_head_att/dense_7/Tensordot/Const_2:output:07multi_head_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_7/Tensordot/concat_1?
 multi_head_att/dense_7/TensordotReshape1multi_head_att/dense_7/Tensordot/MatMul:product:02multi_head_att/dense_7/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_7/Tensordot?
multi_head_att/MatMul_2BatchMatMulV2)multi_head_att/dense_1/Tensordot:output:0)multi_head_att/dense_4/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_2}
multi_head_att/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_1/y?
multi_head_att/truediv_1RealDiv multi_head_att/MatMul_2:output:0#multi_head_att/truediv_1/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_1?
multi_head_att/Softmax_1Softmaxmulti_head_att/truediv_1:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_1?
multi_head_att/MatMul_3BatchMatMulV2"multi_head_att/Softmax_1:softmax:0)multi_head_att/dense_7/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_3?
/multi_head_att/dense_2/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_2/Tensordot/ReadVariableOp?
%multi_head_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_2/Tensordot/axes?
%multi_head_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_2/Tensordot/free?
&multi_head_att/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/Shape?
.multi_head_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/GatherV2/axis?
)multi_head_att/dense_2/Tensordot/GatherV2GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/free:output:07multi_head_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/GatherV2?
0multi_head_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_2/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_2/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_2/Tensordot/Shape:output:0.multi_head_att/dense_2/Tensordot/axes:output:09multi_head_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_2/Tensordot/GatherV2_1?
&multi_head_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_2/Tensordot/Const?
%multi_head_att/dense_2/Tensordot/ProdProd2multi_head_att/dense_2/Tensordot/GatherV2:output:0/multi_head_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_2/Tensordot/Prod?
(multi_head_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_2/Tensordot/Const_1?
'multi_head_att/dense_2/Tensordot/Prod_1Prod4multi_head_att/dense_2/Tensordot/GatherV2_1:output:01multi_head_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_2/Tensordot/Prod_1?
,multi_head_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_2/Tensordot/concat/axis?
'multi_head_att/dense_2/Tensordot/concatConcatV2.multi_head_att/dense_2/Tensordot/free:output:0.multi_head_att/dense_2/Tensordot/axes:output:05multi_head_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_2/Tensordot/concat?
&multi_head_att/dense_2/Tensordot/stackPack.multi_head_att/dense_2/Tensordot/Prod:output:00multi_head_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_2/Tensordot/stack?
*multi_head_att/dense_2/Tensordot/transpose	Transposeinputs0multi_head_att/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_2/Tensordot/transpose?
(multi_head_att/dense_2/Tensordot/ReshapeReshape.multi_head_att/dense_2/Tensordot/transpose:y:0/multi_head_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_2/Tensordot/Reshape?
'multi_head_att/dense_2/Tensordot/MatMulMatMul1multi_head_att/dense_2/Tensordot/Reshape:output:07multi_head_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_2/Tensordot/MatMul?
(multi_head_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_2/Tensordot/Const_2?
.multi_head_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_2/Tensordot/concat_1/axis?
)multi_head_att/dense_2/Tensordot/concat_1ConcatV22multi_head_att/dense_2/Tensordot/GatherV2:output:01multi_head_att/dense_2/Tensordot/Const_2:output:07multi_head_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_2/Tensordot/concat_1?
 multi_head_att/dense_2/TensordotReshape1multi_head_att/dense_2/Tensordot/MatMul:product:02multi_head_att/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_2/Tensordot?
/multi_head_att/dense_5/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_5_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_5/Tensordot/ReadVariableOp?
%multi_head_att/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_5/Tensordot/axes?
%multi_head_att/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_5/Tensordot/free?
&multi_head_att/dense_5/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/Shape?
.multi_head_att/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/GatherV2/axis?
)multi_head_att/dense_5/Tensordot/GatherV2GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/free:output:07multi_head_att/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/GatherV2?
0multi_head_att/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_5/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_5/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_5/Tensordot/Shape:output:0.multi_head_att/dense_5/Tensordot/axes:output:09multi_head_att/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_5/Tensordot/GatherV2_1?
&multi_head_att/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_5/Tensordot/Const?
%multi_head_att/dense_5/Tensordot/ProdProd2multi_head_att/dense_5/Tensordot/GatherV2:output:0/multi_head_att/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_5/Tensordot/Prod?
(multi_head_att/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_5/Tensordot/Const_1?
'multi_head_att/dense_5/Tensordot/Prod_1Prod4multi_head_att/dense_5/Tensordot/GatherV2_1:output:01multi_head_att/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_5/Tensordot/Prod_1?
,multi_head_att/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_5/Tensordot/concat/axis?
'multi_head_att/dense_5/Tensordot/concatConcatV2.multi_head_att/dense_5/Tensordot/free:output:0.multi_head_att/dense_5/Tensordot/axes:output:05multi_head_att/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_5/Tensordot/concat?
&multi_head_att/dense_5/Tensordot/stackPack.multi_head_att/dense_5/Tensordot/Prod:output:00multi_head_att/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_5/Tensordot/stack?
*multi_head_att/dense_5/Tensordot/transpose	Transposeinputs0multi_head_att/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_5/Tensordot/transpose?
(multi_head_att/dense_5/Tensordot/ReshapeReshape.multi_head_att/dense_5/Tensordot/transpose:y:0/multi_head_att/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_5/Tensordot/Reshape?
'multi_head_att/dense_5/Tensordot/MatMulMatMul1multi_head_att/dense_5/Tensordot/Reshape:output:07multi_head_att/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_5/Tensordot/MatMul?
(multi_head_att/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_5/Tensordot/Const_2?
.multi_head_att/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_5/Tensordot/concat_1/axis?
)multi_head_att/dense_5/Tensordot/concat_1ConcatV22multi_head_att/dense_5/Tensordot/GatherV2:output:01multi_head_att/dense_5/Tensordot/Const_2:output:07multi_head_att/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_5/Tensordot/concat_1?
 multi_head_att/dense_5/TensordotReshape1multi_head_att/dense_5/Tensordot/MatMul:product:02multi_head_att/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_5/Tensordot?
/multi_head_att/dense_8/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:((*
dtype021
/multi_head_att/dense_8/Tensordot/ReadVariableOp?
%multi_head_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_8/Tensordot/axes?
%multi_head_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_8/Tensordot/free?
&multi_head_att/dense_8/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/Shape?
.multi_head_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/GatherV2/axis?
)multi_head_att/dense_8/Tensordot/GatherV2GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/free:output:07multi_head_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/GatherV2?
0multi_head_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_8/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_8/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_8/Tensordot/Shape:output:0.multi_head_att/dense_8/Tensordot/axes:output:09multi_head_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_8/Tensordot/GatherV2_1?
&multi_head_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_8/Tensordot/Const?
%multi_head_att/dense_8/Tensordot/ProdProd2multi_head_att/dense_8/Tensordot/GatherV2:output:0/multi_head_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_8/Tensordot/Prod?
(multi_head_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_8/Tensordot/Const_1?
'multi_head_att/dense_8/Tensordot/Prod_1Prod4multi_head_att/dense_8/Tensordot/GatherV2_1:output:01multi_head_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_8/Tensordot/Prod_1?
,multi_head_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_8/Tensordot/concat/axis?
'multi_head_att/dense_8/Tensordot/concatConcatV2.multi_head_att/dense_8/Tensordot/free:output:0.multi_head_att/dense_8/Tensordot/axes:output:05multi_head_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_8/Tensordot/concat?
&multi_head_att/dense_8/Tensordot/stackPack.multi_head_att/dense_8/Tensordot/Prod:output:00multi_head_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_8/Tensordot/stack?
*multi_head_att/dense_8/Tensordot/transpose	Transposeinputs0multi_head_att/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2,
*multi_head_att/dense_8/Tensordot/transpose?
(multi_head_att/dense_8/Tensordot/ReshapeReshape.multi_head_att/dense_8/Tensordot/transpose:y:0/multi_head_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_8/Tensordot/Reshape?
'multi_head_att/dense_8/Tensordot/MatMulMatMul1multi_head_att/dense_8/Tensordot/Reshape:output:07multi_head_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_8/Tensordot/MatMul?
(multi_head_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_8/Tensordot/Const_2?
.multi_head_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_8/Tensordot/concat_1/axis?
)multi_head_att/dense_8/Tensordot/concat_1ConcatV22multi_head_att/dense_8/Tensordot/GatherV2:output:01multi_head_att/dense_8/Tensordot/Const_2:output:07multi_head_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_8/Tensordot/concat_1?
 multi_head_att/dense_8/TensordotReshape1multi_head_att/dense_8/Tensordot/MatMul:product:02multi_head_att/dense_8/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_8/Tensordot?
multi_head_att/MatMul_4BatchMatMulV2)multi_head_att/dense_2/Tensordot:output:0)multi_head_att/dense_5/Tensordot:output:0*
T0*-
_output_shapes
:???????????*
adj_y(2
multi_head_att/MatMul_4}
multi_head_att/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r?|A2
multi_head_att/truediv_2/y?
multi_head_att/truediv_2RealDiv multi_head_att/MatMul_4:output:0#multi_head_att/truediv_2/y:output:0*
T0*-
_output_shapes
:???????????2
multi_head_att/truediv_2?
multi_head_att/Softmax_2Softmaxmulti_head_att/truediv_2:z:0*
T0*-
_output_shapes
:???????????2
multi_head_att/Softmax_2?
multi_head_att/MatMul_5BatchMatMulV2"multi_head_att/Softmax_2:softmax:0)multi_head_att/dense_8/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
multi_head_att/MatMul_5z
multi_head_att/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
multi_head_att/concat/axis?
multi_head_att/concatConcatV2 multi_head_att/MatMul_1:output:0 multi_head_att/MatMul_3:output:0 multi_head_att/MatMul_5:output:0#multi_head_att/concat/axis:output:0*
N*
T0*,
_output_shapes
:??????????x2
multi_head_att/concat?
/multi_head_att/dense_9/Tensordot/ReadVariableOpReadVariableOp8multi_head_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

:x(*
dtype021
/multi_head_att/dense_9/Tensordot/ReadVariableOp?
%multi_head_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%multi_head_att/dense_9/Tensordot/axes?
%multi_head_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%multi_head_att/dense_9/Tensordot/free?
&multi_head_att/dense_9/Tensordot/ShapeShapemulti_head_att/concat:output:0*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/Shape?
.multi_head_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/GatherV2/axis?
)multi_head_att/dense_9/Tensordot/GatherV2GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/free:output:07multi_head_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/GatherV2?
0multi_head_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0multi_head_att/dense_9/Tensordot/GatherV2_1/axis?
+multi_head_att/dense_9/Tensordot/GatherV2_1GatherV2/multi_head_att/dense_9/Tensordot/Shape:output:0.multi_head_att/dense_9/Tensordot/axes:output:09multi_head_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+multi_head_att/dense_9/Tensordot/GatherV2_1?
&multi_head_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&multi_head_att/dense_9/Tensordot/Const?
%multi_head_att/dense_9/Tensordot/ProdProd2multi_head_att/dense_9/Tensordot/GatherV2:output:0/multi_head_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%multi_head_att/dense_9/Tensordot/Prod?
(multi_head_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(multi_head_att/dense_9/Tensordot/Const_1?
'multi_head_att/dense_9/Tensordot/Prod_1Prod4multi_head_att/dense_9/Tensordot/GatherV2_1:output:01multi_head_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'multi_head_att/dense_9/Tensordot/Prod_1?
,multi_head_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,multi_head_att/dense_9/Tensordot/concat/axis?
'multi_head_att/dense_9/Tensordot/concatConcatV2.multi_head_att/dense_9/Tensordot/free:output:0.multi_head_att/dense_9/Tensordot/axes:output:05multi_head_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_att/dense_9/Tensordot/concat?
&multi_head_att/dense_9/Tensordot/stackPack.multi_head_att/dense_9/Tensordot/Prod:output:00multi_head_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_att/dense_9/Tensordot/stack?
*multi_head_att/dense_9/Tensordot/transpose	Transposemulti_head_att/concat:output:00multi_head_att/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????x2,
*multi_head_att/dense_9/Tensordot/transpose?
(multi_head_att/dense_9/Tensordot/ReshapeReshape.multi_head_att/dense_9/Tensordot/transpose:y:0/multi_head_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(multi_head_att/dense_9/Tensordot/Reshape?
'multi_head_att/dense_9/Tensordot/MatMulMatMul1multi_head_att/dense_9/Tensordot/Reshape:output:07multi_head_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2)
'multi_head_att/dense_9/Tensordot/MatMul?
(multi_head_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2*
(multi_head_att/dense_9/Tensordot/Const_2?
.multi_head_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_att/dense_9/Tensordot/concat_1/axis?
)multi_head_att/dense_9/Tensordot/concat_1ConcatV22multi_head_att/dense_9/Tensordot/GatherV2:output:01multi_head_att/dense_9/Tensordot/Const_2:output:07multi_head_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_att/dense_9/Tensordot/concat_1?
 multi_head_att/dense_9/TensordotReshape1multi_head_att/dense_9/Tensordot/MatMul:product:02multi_head_att/dense_9/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2"
 multi_head_att/dense_9/Tensordot?
dropout/IdentityIdentity)multi_head_att/dense_9/Tensordot:output:0*
T0*,
_output_shapes
:??????????(2
dropout/Identitym
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:??????????(2
add?
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2"
 layer_normalization/moments/mean?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:??????????2*
(layer_normalization/moments/StopGradient?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(2/
-layer_normalization/moments/SquaredDifference?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2&
$layer_normalization/moments/variance?
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52%
#layer_normalization/batchnorm/add/y?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2#
!layer_normalization/batchnorm/add?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization/batchnorm/Rsqrt?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/mul?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_1?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/mul_2?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2#
!layer_normalization/batchnorm/sub?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization/batchnorm/add_1?
,sequential/dense_10/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_10_tensordot_readvariableop_resource*
_output_shapes

:( *
dtype02.
,sequential/dense_10/Tensordot/ReadVariableOp?
"sequential/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_10/Tensordot/axes?
"sequential/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_10/Tensordot/free?
#sequential/dense_10/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/Shape?
+sequential/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/GatherV2/axis?
&sequential/dense_10/Tensordot/GatherV2GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/free:output:04sequential/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/GatherV2?
-sequential/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_10/Tensordot/GatherV2_1/axis?
(sequential/dense_10/Tensordot/GatherV2_1GatherV2,sequential/dense_10/Tensordot/Shape:output:0+sequential/dense_10/Tensordot/axes:output:06sequential/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_10/Tensordot/GatherV2_1?
#sequential/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_10/Tensordot/Const?
"sequential/dense_10/Tensordot/ProdProd/sequential/dense_10/Tensordot/GatherV2:output:0,sequential/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_10/Tensordot/Prod?
%sequential/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_1?
$sequential/dense_10/Tensordot/Prod_1Prod1sequential/dense_10/Tensordot/GatherV2_1:output:0.sequential/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_10/Tensordot/Prod_1?
)sequential/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_10/Tensordot/concat/axis?
$sequential/dense_10/Tensordot/concatConcatV2+sequential/dense_10/Tensordot/free:output:0+sequential/dense_10/Tensordot/axes:output:02sequential/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_10/Tensordot/concat?
#sequential/dense_10/Tensordot/stackPack+sequential/dense_10/Tensordot/Prod:output:0-sequential/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_10/Tensordot/stack?
'sequential/dense_10/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0-sequential/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????(2)
'sequential/dense_10/Tensordot/transpose?
%sequential/dense_10/Tensordot/ReshapeReshape+sequential/dense_10/Tensordot/transpose:y:0,sequential/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_10/Tensordot/Reshape?
$sequential/dense_10/Tensordot/MatMulMatMul.sequential/dense_10/Tensordot/Reshape:output:04sequential/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$sequential/dense_10/Tensordot/MatMul?
%sequential/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_10/Tensordot/Const_2?
+sequential/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_10/Tensordot/concat_1/axis?
&sequential/dense_10/Tensordot/concat_1ConcatV2/sequential/dense_10/Tensordot/GatherV2:output:0.sequential/dense_10/Tensordot/Const_2:output:04sequential/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_10/Tensordot/concat_1?
sequential/dense_10/TensordotReshape.sequential/dense_10/Tensordot/MatMul:product:0/sequential/dense_10/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Tensordot?
*sequential/dense_10/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/dense_10/BiasAdd/ReadVariableOp?
sequential/dense_10/BiasAddBiasAdd&sequential/dense_10/Tensordot:output:02sequential/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/BiasAdd?
sequential/dense_10/ReluRelu$sequential/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:?????????? 2
sequential/dense_10/Relu?
,sequential/dense_11/Tensordot/ReadVariableOpReadVariableOp5sequential_dense_11_tensordot_readvariableop_resource*
_output_shapes

: (*
dtype02.
,sequential/dense_11/Tensordot/ReadVariableOp?
"sequential/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense_11/Tensordot/axes?
"sequential/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"sequential/dense_11/Tensordot/free?
#sequential/dense_11/Tensordot/ShapeShape&sequential/dense_10/Relu:activations:0*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/Shape?
+sequential/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/GatherV2/axis?
&sequential/dense_11/Tensordot/GatherV2GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/free:output:04sequential/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/GatherV2?
-sequential/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/dense_11/Tensordot/GatherV2_1/axis?
(sequential/dense_11/Tensordot/GatherV2_1GatherV2,sequential/dense_11/Tensordot/Shape:output:0+sequential/dense_11/Tensordot/axes:output:06sequential/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential/dense_11/Tensordot/GatherV2_1?
#sequential/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense_11/Tensordot/Const?
"sequential/dense_11/Tensordot/ProdProd/sequential/dense_11/Tensordot/GatherV2:output:0,sequential/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"sequential/dense_11/Tensordot/Prod?
%sequential/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/dense_11/Tensordot/Const_1?
$sequential/dense_11/Tensordot/Prod_1Prod1sequential/dense_11/Tensordot/GatherV2_1:output:0.sequential/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$sequential/dense_11/Tensordot/Prod_1?
)sequential/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense_11/Tensordot/concat/axis?
$sequential/dense_11/Tensordot/concatConcatV2+sequential/dense_11/Tensordot/free:output:0+sequential/dense_11/Tensordot/axes:output:02sequential/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense_11/Tensordot/concat?
#sequential/dense_11/Tensordot/stackPack+sequential/dense_11/Tensordot/Prod:output:0-sequential/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_11/Tensordot/stack?
'sequential/dense_11/Tensordot/transpose	Transpose&sequential/dense_10/Relu:activations:0-sequential/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????? 2)
'sequential/dense_11/Tensordot/transpose?
%sequential/dense_11/Tensordot/ReshapeReshape+sequential/dense_11/Tensordot/transpose:y:0,sequential/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2'
%sequential/dense_11/Tensordot/Reshape?
$sequential/dense_11/Tensordot/MatMulMatMul.sequential/dense_11/Tensordot/Reshape:output:04sequential/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2&
$sequential/dense_11/Tensordot/MatMul?
%sequential/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2'
%sequential/dense_11/Tensordot/Const_2?
+sequential/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense_11/Tensordot/concat_1/axis?
&sequential/dense_11/Tensordot/concat_1ConcatV2/sequential/dense_11/Tensordot/GatherV2:output:0.sequential/dense_11/Tensordot/Const_2:output:04sequential/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential/dense_11/Tensordot/concat_1?
sequential/dense_11/TensordotReshape.sequential/dense_11/Tensordot/MatMul:product:0/sequential/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/Tensordot?
*sequential/dense_11/BiasAdd/ReadVariableOpReadVariableOp3sequential_dense_11_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02,
*sequential/dense_11/BiasAdd/ReadVariableOp?
sequential/dense_11/BiasAddBiasAdd&sequential/dense_11/Tensordot:output:02sequential/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2
sequential/dense_11/BiasAdd?
dropout_1/IdentityIdentity$sequential/dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:??????????(2
dropout_1/Identity?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:??????????(2
add_1?
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2$
"layer_normalization_1/moments/mean?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:??????????2,
*layer_normalization_1/moments/StopGradient?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:??????????(21
/layer_normalization_1/moments/SquaredDifference?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:??????????*
	keep_dims(2(
&layer_normalization_1/moments/variance?
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_1/batchnorm/add/y?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:??????????2%
#layer_normalization_1/batchnorm/add?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:??????????2'
%layer_normalization_1/batchnorm/Rsqrt?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/mul?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_1?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/mul_2?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:??????????(2%
#layer_normalization_1/batchnorm/sub?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:??????????(2'
%layer_normalization_1/batchnorm/add_1?
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:??????????(2

Identity?
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp.^multi_head_att/dense/Tensordot/ReadVariableOp0^multi_head_att/dense_1/Tensordot/ReadVariableOp0^multi_head_att/dense_2/Tensordot/ReadVariableOp0^multi_head_att/dense_3/Tensordot/ReadVariableOp0^multi_head_att/dense_4/Tensordot/ReadVariableOp0^multi_head_att/dense_5/Tensordot/ReadVariableOp0^multi_head_att/dense_6/Tensordot/ReadVariableOp0^multi_head_att/dense_7/Tensordot/ReadVariableOp0^multi_head_att/dense_8/Tensordot/ReadVariableOp0^multi_head_att/dense_9/Tensordot/ReadVariableOp+^sequential/dense_10/BiasAdd/ReadVariableOp-^sequential/dense_10/Tensordot/ReadVariableOp+^sequential/dense_11/BiasAdd/ReadVariableOp-^sequential/dense_11/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????(: : : : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2^
-multi_head_att/dense/Tensordot/ReadVariableOp-multi_head_att/dense/Tensordot/ReadVariableOp2b
/multi_head_att/dense_1/Tensordot/ReadVariableOp/multi_head_att/dense_1/Tensordot/ReadVariableOp2b
/multi_head_att/dense_2/Tensordot/ReadVariableOp/multi_head_att/dense_2/Tensordot/ReadVariableOp2b
/multi_head_att/dense_3/Tensordot/ReadVariableOp/multi_head_att/dense_3/Tensordot/ReadVariableOp2b
/multi_head_att/dense_4/Tensordot/ReadVariableOp/multi_head_att/dense_4/Tensordot/ReadVariableOp2b
/multi_head_att/dense_5/Tensordot/ReadVariableOp/multi_head_att/dense_5/Tensordot/ReadVariableOp2b
/multi_head_att/dense_6/Tensordot/ReadVariableOp/multi_head_att/dense_6/Tensordot/ReadVariableOp2b
/multi_head_att/dense_7/Tensordot/ReadVariableOp/multi_head_att/dense_7/Tensordot/ReadVariableOp2b
/multi_head_att/dense_8/Tensordot/ReadVariableOp/multi_head_att/dense_8/Tensordot/ReadVariableOp2b
/multi_head_att/dense_9/Tensordot/ReadVariableOp/multi_head_att/dense_9/Tensordot/ReadVariableOp2X
*sequential/dense_10/BiasAdd/ReadVariableOp*sequential/dense_10/BiasAdd/ReadVariableOp2\
,sequential/dense_10/Tensordot/ReadVariableOp,sequential/dense_10/Tensordot/ReadVariableOp2X
*sequential/dense_11/BiasAdd/ReadVariableOp*sequential/dense_11/BiasAdd/ReadVariableOp2\
,sequential/dense_11/Tensordot/ReadVariableOp,sequential/dense_11/Tensordot/ReadVariableOp:T P
,
_output_shapes
:??????????(
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_6554
input_1
unknown:
??(
	unknown_0:((
	unknown_1:((
	unknown_2:((
	unknown_3:((
	unknown_4:((
	unknown_5:((
	unknown_6:((
	unknown_7:((
	unknown_8:((
	unknown_9:x(

unknown_10:(

unknown_11:(

unknown_12:( 

unknown_13: 

unknown_14: (

unknown_15:(

unknown_16:(

unknown_17:(

unknown_18:(

unknown_19:

unknown_20:

unknown_21:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_64542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:??????????: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????<
dense_130
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ش
?"
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?
_tf_keras_network?{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 250]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "w2v_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 20000, "output_dim": 40, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "w2v_embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block", "inbound_nodes": [[["w2v_embedding", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last", "keepdims": false}, "name": "global_average_pooling1d", "inbound_nodes": [[["transformer_block", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["global_average_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_13", 0, 0]]}, "shared_object_id": 12, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 250]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 250]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 250]}, "float32", "input_1"]}, "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 14}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 250]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 250]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

embeddings
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "w2v_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "w2v_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 20000, "output_dim": 40, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 1}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [1024, 1]}}
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "transformer_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "TransformerBlock", "config": {"layer was saved without config": true}}
?
trainable_variables
 	variables
!regularization_losses
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "global_average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last", "keepdims": false}, "inbound_nodes": [[["transformer_block", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 15}}
?
#trainable_variables
$	variables
%regularization_losses
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["global_average_pooling1d", 0, 0, {}]]], "shared_object_id": 4}
?

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_12", 0, 0, {}]]], "shared_object_id": 8}
?	

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_3", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratem?'m?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?v?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?"
	optimizer
?
0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
'19
(20
121
222"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
'19
(20
121
222"
trackable_list_wrapper
?
Nnon_trainable_variables

trainable_variables

Olayers
regularization_losses
Pmetrics
Qlayer_regularization_losses
	variables
Rlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
6:4
??(2"word2_vec/w2v_embedding/embeddings
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables

Slayers
regularization_losses
Tmetrics
Ulayer_regularization_losses
Vnon_trainable_variables
Wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
	Xquery
Ykey
	Zvalue
[	linOutput
\trainable_variables
]	variables
^regularization_losses
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "multi_head_att", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAtt", "config": {"layer was saved without config": true}}
?
`layer_with_weights-0
`layer-0
alayer_with_weights-1
alayer-1
btrainable_variables
cregularization_losses
d	variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 250, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 250, 40]}, "float32", "dense_10_input"]}, "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 250, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_10_input"}, "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}]}}}
?
faxis
	Jgamma
Kbeta
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 29, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?
kaxis
	Lgamma
Mbeta
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 32, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "shared_object_id": 33}
?
ttrainable_variables
u	variables
vregularization_losses
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "shared_object_id": 34}
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17"
trackable_list_wrapper
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables

xlayers
regularization_losses
ymetrics
zlayer_regularization_losses
{non_trainable_variables
|layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
 	variables

}layers
!regularization_losses
~metrics
layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#trainable_variables
$	variables
?layers
%regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:(2dense_12/kernel
:2dense_12/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)trainable_variables
*	variables
?layers
+regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-trainable_variables
.	variables
?layers
/regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_13/kernel
:2dense_13/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3trainable_variables
4	variables
?layers
5regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?:=((2-transformer_block/multi_head_att/dense/kernel
A:?((2/transformer_block/multi_head_att/dense_1/kernel
A:?((2/transformer_block/multi_head_att/dense_2/kernel
A:?((2/transformer_block/multi_head_att/dense_3/kernel
A:?((2/transformer_block/multi_head_att/dense_4/kernel
A:?((2/transformer_block/multi_head_att/dense_5/kernel
A:?((2/transformer_block/multi_head_att/dense_6/kernel
A:?((2/transformer_block/multi_head_att/dense_7/kernel
A:?((2/transformer_block/multi_head_att/dense_8/kernel
A:?x(2/transformer_block/multi_head_att/dense_9/kernel
!:( 2dense_10/kernel
: 2dense_10/bias
!: (2dense_11/kernel
:(2dense_11/bias
9:7(2+transformer_block/layer_normalization/gamma
8:6(2*transformer_block/layer_normalization/beta
;:9(2-transformer_block/layer_normalization_1/gamma
::8(2,transformer_block/layer_normalization_1/beta
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
?0
?1"
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
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?

Ekernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 120]}}
f
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9"
trackable_list_wrapper
f
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\trainable_variables
]	variables
?layers
^regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Fkernel
Gbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

Hkernel
Ibias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 32]}}
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
?
?non_trainable_variables
btrainable_variables
?layers
cregularization_losses
?metrics
 ?layer_regularization_losses
d	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gtrainable_variables
h	variables
?layers
iregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ltrainable_variables
m	variables
?layers
nregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ptrainable_variables
q	variables
?layers
rregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ttrainable_variables
u	variables
?layers
vregularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
J
0
1
2
3
4
5"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 40}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 14}
?

<kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

=kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

>kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

?kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

@kernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

Akernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

Bkernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 65}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 66}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 67, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

Ckernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 69}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 70}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 71, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
?

Dkernel
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 73}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 74}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 75, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 250, 40]}}
'
E0"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
o
?0
?1
?2
?3
?4
?5
?6
?7
?8
[9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
`0
a1"
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
=0"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
>0"
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
?0"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
A0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
B0"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
C0"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
D0"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?	variables
?layers
?regularization_losses
?metrics
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
;:9
??(2)Adam/word2_vec/w2v_embedding/embeddings/m
&:$(2Adam/dense_12/kernel/m
 :2Adam/dense_12/bias/m
&:$2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
D:B((24Adam/transformer_block/multi_head_att/dense/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_1/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_2/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_3/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_4/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_5/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_6/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_7/kernel/m
F:D((26Adam/transformer_block/multi_head_att/dense_8/kernel/m
F:Dx(26Adam/transformer_block/multi_head_att/dense_9/kernel/m
&:$( 2Adam/dense_10/kernel/m
 : 2Adam/dense_10/bias/m
&:$ (2Adam/dense_11/kernel/m
 :(2Adam/dense_11/bias/m
>:<(22Adam/transformer_block/layer_normalization/gamma/m
=:;(21Adam/transformer_block/layer_normalization/beta/m
@:>(24Adam/transformer_block/layer_normalization_1/gamma/m
?:=(23Adam/transformer_block/layer_normalization_1/beta/m
;:9
??(2)Adam/word2_vec/w2v_embedding/embeddings/v
&:$(2Adam/dense_12/kernel/v
 :2Adam/dense_12/bias/v
&:$2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
D:B((24Adam/transformer_block/multi_head_att/dense/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_1/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_2/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_3/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_4/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_5/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_6/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_7/kernel/v
F:D((26Adam/transformer_block/multi_head_att/dense_8/kernel/v
F:Dx(26Adam/transformer_block/multi_head_att/dense_9/kernel/v
&:$( 2Adam/dense_10/kernel/v
 : 2Adam/dense_10/bias/v
&:$ (2Adam/dense_11/kernel/v
 :(2Adam/dense_11/bias/v
>:<(22Adam/transformer_block/layer_normalization/gamma/v
=:;(21Adam/transformer_block/layer_normalization/beta/v
@:>(24Adam/transformer_block/layer_normalization_1/gamma/v
?:=(23Adam/transformer_block/layer_normalization_1/beta/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_7100
?__inference_model_layer_call_and_return_conditional_losses_7501
?__inference_model_layer_call_and_return_conditional_losses_6611
?__inference_model_layer_call_and_return_conditional_losses_6668?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_5819
$__inference_model_layer_call_fn_7552
$__inference_model_layer_call_fn_7603
$__inference_model_layer_call_fn_6554?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_5089?
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
annotations? *'?$
"?
input_1??????????
?2?
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_7613?
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
,__inference_w2v_embedding_layer_call_fn_7620?
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
?2?
K__inference_transformer_block_layer_call_and_return_conditional_losses_7969
K__inference_transformer_block_layer_call_and_return_conditional_losses_8332?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_transformer_block_layer_call_fn_8373
0__inference_transformer_block_layer_call_fn_8414?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8420
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8426?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_global_average_pooling1d_layer_call_fn_8431
7__inference_global_average_pooling1d_layer_call_fn_8436?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_2_layer_call_and_return_conditional_losses_8441
C__inference_dropout_2_layer_call_and_return_conditional_losses_8453?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_2_layer_call_fn_8458
(__inference_dropout_2_layer_call_fn_8463?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_12_layer_call_and_return_conditional_losses_8474?
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
'__inference_dense_12_layer_call_fn_8483?
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
?2?
C__inference_dropout_3_layer_call_and_return_conditional_losses_8488
C__inference_dropout_3_layer_call_and_return_conditional_losses_8500?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_3_layer_call_fn_8505
(__inference_dropout_3_layer_call_fn_8510?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_13_layer_call_and_return_conditional_losses_8521?
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
'__inference_dense_13_layer_call_fn_8530?
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
"__inference_signature_wrapper_6727input_1"?
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
 
?2??
???
FullArgSpec,
args$?!
jself
jquery
jkey
jvalue
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
?2??
???
FullArgSpec,
args$?!
jself
jquery
jkey
jvalue
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
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_8587
D__inference_sequential_layer_call_and_return_conditional_losses_8644
D__inference_sequential_layer_call_and_return_conditional_losses_5268
D__inference_sequential_layer_call_and_return_conditional_losses_5282?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_5181
)__inference_sequential_layer_call_fn_8657
)__inference_sequential_layer_call_fn_8670
)__inference_sequential_layer_call_fn_5254?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
B__inference_dense_10_layer_call_and_return_conditional_losses_8701?
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
'__inference_dense_10_layer_call_fn_8710?
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
B__inference_dense_11_layer_call_and_return_conditional_losses_8740?
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
'__inference_dense_11_layer_call_fn_8749?
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
 ?
__inference__wrapped_model_5089?<?B=@C>ADEJKFGHILM'(121?.
'?$
"?
input_1??????????
? "3?0
.
dense_13"?
dense_13??????????
B__inference_dense_10_layer_call_and_return_conditional_losses_8701fFG4?1
*?'
%?"
inputs??????????(
? "*?'
 ?
0?????????? 
? ?
'__inference_dense_10_layer_call_fn_8710YFG4?1
*?'
%?"
inputs??????????(
? "??????????? ?
B__inference_dense_11_layer_call_and_return_conditional_losses_8740fHI4?1
*?'
%?"
inputs?????????? 
? "*?'
 ?
0??????????(
? ?
'__inference_dense_11_layer_call_fn_8749YHI4?1
*?'
%?"
inputs?????????? 
? "???????????(?
B__inference_dense_12_layer_call_and_return_conditional_losses_8474\'(/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? z
'__inference_dense_12_layer_call_fn_8483O'(/?,
%?"
 ?
inputs?????????(
? "???????????
B__inference_dense_13_layer_call_and_return_conditional_losses_8521\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_13_layer_call_fn_8530O12/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dropout_2_layer_call_and_return_conditional_losses_8441\3?0
)?&
 ?
inputs?????????(
p 
? "%?"
?
0?????????(
? ?
C__inference_dropout_2_layer_call_and_return_conditional_losses_8453\3?0
)?&
 ?
inputs?????????(
p
? "%?"
?
0?????????(
? {
(__inference_dropout_2_layer_call_fn_8458O3?0
)?&
 ?
inputs?????????(
p 
? "??????????({
(__inference_dropout_2_layer_call_fn_8463O3?0
)?&
 ?
inputs?????????(
p
? "??????????(?
C__inference_dropout_3_layer_call_and_return_conditional_losses_8488\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
C__inference_dropout_3_layer_call_and_return_conditional_losses_8500\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? {
(__inference_dropout_3_layer_call_fn_8505O3?0
)?&
 ?
inputs?????????
p 
? "??????????{
(__inference_dropout_3_layer_call_fn_8510O3?0
)?&
 ?
inputs?????????
p
? "???????????
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8420{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_8426a8?5
.?+
%?"
inputs??????????(

 
? "%?"
?
0?????????(
? ?
7__inference_global_average_pooling1d_layer_call_fn_8431nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
7__inference_global_average_pooling1d_layer_call_fn_8436T8?5
.?+
%?"
inputs??????????(

 
? "??????????(?
?__inference_model_layer_call_and_return_conditional_losses_6611{<?B=@C>ADEJKFGHILM'(129?6
/?,
"?
input_1??????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_6668{<?B=@C>ADEJKFGHILM'(129?6
/?,
"?
input_1??????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_7100z<?B=@C>ADEJKFGHILM'(128?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_7501z<?B=@C>ADEJKFGHILM'(128?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_5819n<?B=@C>ADEJKFGHILM'(129?6
/?,
"?
input_1??????????
p 

 
? "???????????
$__inference_model_layer_call_fn_6554n<?B=@C>ADEJKFGHILM'(129?6
/?,
"?
input_1??????????
p

 
? "???????????
$__inference_model_layer_call_fn_7552m<?B=@C>ADEJKFGHILM'(128?5
.?+
!?
inputs??????????
p 

 
? "???????????
$__inference_model_layer_call_fn_7603m<?B=@C>ADEJKFGHILM'(128?5
.?+
!?
inputs??????????
p

 
? "???????????
D__inference_sequential_layer_call_and_return_conditional_losses_5268xFGHID?A
:?7
-?*
dense_10_input??????????(
p 

 
? "*?'
 ?
0??????????(
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_5282xFGHID?A
:?7
-?*
dense_10_input??????????(
p

 
? "*?'
 ?
0??????????(
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_8587pFGHI<?9
2?/
%?"
inputs??????????(
p 

 
? "*?'
 ?
0??????????(
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_8644pFGHI<?9
2?/
%?"
inputs??????????(
p

 
? "*?'
 ?
0??????????(
? ?
)__inference_sequential_layer_call_fn_5181kFGHID?A
:?7
-?*
dense_10_input??????????(
p 

 
? "???????????(?
)__inference_sequential_layer_call_fn_5254kFGHID?A
:?7
-?*
dense_10_input??????????(
p

 
? "???????????(?
)__inference_sequential_layer_call_fn_8657cFGHI<?9
2?/
%?"
inputs??????????(
p 

 
? "???????????(?
)__inference_sequential_layer_call_fn_8670cFGHI<?9
2?/
%?"
inputs??????????(
p

 
? "???????????(?
"__inference_signature_wrapper_6727?<?B=@C>ADEJKFGHILM'(12<?9
? 
2?/
-
input_1"?
input_1??????????"3?0
.
dense_13"?
dense_13??????????
K__inference_transformer_block_layer_call_and_return_conditional_losses_7969z<?B=@C>ADEJKFGHILM8?5
.?+
%?"
inputs??????????(
p 
? "*?'
 ?
0??????????(
? ?
K__inference_transformer_block_layer_call_and_return_conditional_losses_8332z<?B=@C>ADEJKFGHILM8?5
.?+
%?"
inputs??????????(
p
? "*?'
 ?
0??????????(
? ?
0__inference_transformer_block_layer_call_fn_8373m<?B=@C>ADEJKFGHILM8?5
.?+
%?"
inputs??????????(
p 
? "???????????(?
0__inference_transformer_block_layer_call_fn_8414m<?B=@C>ADEJKFGHILM8?5
.?+
%?"
inputs??????????(
p
? "???????????(?
G__inference_w2v_embedding_layer_call_and_return_conditional_losses_7613a0?-
&?#
!?
inputs??????????
? "*?'
 ?
0??????????(
? ?
,__inference_w2v_embedding_layer_call_fn_7620T0?-
&?#
!?
inputs??????????
? "???????????(