import {constructors} from '../util/taffy_constructors.js'
import * as Convolution from './convolution.js'
import {higherOrderPrimitives} from './higherOrder.js'

const {op_doc, tensor_description, tensor_shape} = constructors

/*
---------------------------------
---------- helper fns -----------
---------------------------------
*/
const isTensor = v => {
	try {return v.constructor === constructors.tensor_description}
	catch(e){return false}
}

const ensureAllTensors = tensors => tensors.forEach((t,i) => {
	if(!isTensor(t)){
		let got = ''
		try {
			got = `type "${t.constructor.name}"`
		} catch(e){ got = '"unknown"' }
		const message = `argument ${i} is not a tensor, instead got ${got}`
		throw({message, i, arg: t})
	}
})

// TODO: broadcast to most general dtype
export function broadcastDTypes(tensors){
	return tensors[0].dtype
}

export function broadcastShapes(tensors){
	const rank = Math.max(...tensors.map(t=>t.shape.length)),
		shapes = tensors
			.map(t => Array(rank-t.shape.length).fill(1).concat(t.shape)),
		res_shape = Array(rank).fill().map((_, i) => {
			const dims = shapes.map(s => s[i]),
				symbols = [...new Set(dims.filter(isNaN))],
				numbersNotOne = [...new Set(dims.filter(x=>!isNaN(x) && x!=1))]
			if(numbersNotOne.length > 1){
				const message = 'tensors are not broadcastable along ' +
					`dimension ${i}, with values ${dims}`
				throw({message, metaData: {dims, i},
					metaDataIdentifier: 'not_broadcastable'})	
			}
			if(symbols.length > 1){
				const message = 'symbolic dimensions are broadcastable, '+
					`along dimension ${i}, with values ${dims}`
				throw({message, metaData: {dims, i},
					metaDataIdentifier: 'not_broadcastable'})	
			}
			if(symbols.length == 1){
				return numbersNotOne.length == 0? symbols[0] : numbersNotOne[0]
			}
			return numbersNotOne.length == 0? 1 : numbersNotOne[0]
		}),
		res_dtype = broadcastDTypes(tensors)
	if(!tensors.every(t => t.dtype == res_dtype)){
		throw({message: 'tensors are of different dtypes'})
	}
	return {shape: new tensor_shape(res_shape), dtype: res_dtype}
}


/*
---------------------------------
---------- placeholder  ---------
---------------------------------
*/
const __placeholder__primitive = {
	name: 'placeholder',
	type: 'placeholder',
	desc_function: function(){
		throw({message: 'The placeholder desc_function shouldn\'t be called!'})
	},
	doc: new op_doc([],
		['Any value supplied to the placeholder'],
		'Forwards a single supplied value. Takes no inputs.')
}


/*
---------------------------------
------------- relu  -------------
---------------------------------
*/
function __relu__desc_func(tensor_trace, node, tensors){
	if(tensors.length<1) throw({message: 'must take >=1 tensors'})
	ensureAllTensors(tensors)
	const results = tensors.reduce((acc, tensor, i) => {
		const shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			val_ref = node.name + ':' + i,
			out = new tensor_description(shape, dtype, val_ref, 'relu',
				[tensor.val_ref], {})
		return Object.assign(acc, {[val_ref]: out})
	}, {})
	Object.assign(tensor_trace, results)
	return results
}

const __relu__primitive = {
	name: 'relu',
	type: 'tensor',
	desc_function: __relu__desc_func,
	doc: new op_doc(['tensor'], ['ReLU, ie f(x)=max(0,x)'],
		'ReLU activation function')
}


/*
---------------------------------
------------ sigmoid  -----------
---------------------------------
*/
function __sigmoid__desc_func(tensor_trace, node, tensors){
	if(tensors.length<1) throw({message: 'must take >=1 tensors'})
	ensureAllTensors(tensors)
	const results = tensors.reduce((acc, tensor, i) => {
		const shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			val_ref = node.name + ':' + i,
			out = new tensor_description(shape, dtype, val_ref, 'sigmoid',
				[tensor.val_ref], {})
		return Object.assign(acc, {[val_ref]: out})
	}, {})
	Object.assign(tensor_trace, results)
	return results
}

const __sigmoid__primitive = {
	name: 'sigmoid',
	type: 'tensor',
	desc_function: __sigmoid__desc_func,
	doc: new op_doc(['tensor'], ['sigmoid of input'],
		'sigmoid activation function')
}


/*
---------------------------------
------------- tanh  -------------
---------------------------------
*/
function __tanh__desc_func(tensor_trace, node, tensors){
	if(tensors.length<1) throw({message: 'must take >=1 tensors'})
	ensureAllTensors(tensors)
	const results = tensors.reduce((acc, tensor, i) => {
		const shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			val_ref = node.name + ':' + i,
			out = new tensor_description(shape, dtype, val_ref, 'tanh',
				[tensor.val_ref], {})
		return Object.assign(acc, {[val_ref]: out})
	}, {})
	Object.assign(tensor_trace, results)
	return results
}

const __tanh__primitive = {
	name: 'tanh',
	type: 'tensor',
	desc_function: __tanh__desc_func,
	doc: new op_doc(['tensor'], ['tanh of input'], 'tanh activation function')
}


/*
---------------------------------
-------------- exp  -------------
---------------------------------
*/
function __exp__desc_func(tensor_trace, node, tensors){
	if(tensors.length!==1) throw({message: 'must take 1 tensor'})
	ensureAllTensors(tensors)
	const tensor = tensors[0],
		shape = new tensor_shape(tensor.shape),
		dtype = tensor.dtype,
		out = new tensor_description(shape, dtype, node.name+':0', 'exp',
			[tensor.val_ref], {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __exp__primitive = {
	name: 'exp',
	type: 'tensor',
	desc_function: __exp__desc_func,
	doc: new op_doc(['tensor'], ['elementwise exp, ie f(x)=e^x'],
		'exponential function, ie f(x)=e^x')
}


/*
---------------------------------
-------------- abs  -------------
---------------------------------
*/
function __abs__desc_func(tensor_trace, node, tensors){
	if(tensors.length!==1) throw({message: 'must take 1 tensor'})
	ensureAllTensors(tensors)
	const tensor = tensors[0],
		shape = new tensor_shape(tensor.shape),
		dtype = tensor.dtype,
		out = new tensor_description(shape, dtype, node.name+':0', 'abs',
			[tensor.val_ref], {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __abs__primitive = {
	name: 'abs',
	type: 'tensor',
	desc_function: __abs__desc_func,
	doc: new op_doc(['tensor'], ['elementwise absolute value, ie f(x)=|x|'],
		'abs value function, ie f(x)=|x|')
}


/*
---------------------------------
------------- negate  -----------
---------------------------------
*/
function __negate__desc_func(tensor_trace, node, tensors){
	if(tensors.length!==1) throw({message: 'must take 1 tensor'})
	ensureAllTensors(tensors)
	const tensor = tensors[0],
		shape = new tensor_shape(tensor.shape),
		dtype = tensor.dtype,
		out = new tensor_description(shape, dtype, node.name+':0', 'negate',
			[tensor.val_ref], {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __negate__primitive = {
	name: 'negate',
	type: 'tensor',
	desc_function: __negate__desc_func,
	doc: new op_doc(['tensor'], ['negation, ie f(x)=-x'],
		'negation function, ie f(x)=-x')
}


/*
---------------------------------
------------- sqrt  -------------
---------------------------------
*/
function __sqrt__desc_func(tensor_trace, node, tensors){
	if(tensors.length!==1) throw({message: 'must take 1 tensor'})
	ensureAllTensors(tensors)
	const tensor = tensors[0],
		shape = new tensor_shape(tensor.shape),
		dtype = tensor.dtype,
		out = new tensor_description(shape, dtype, node.name+':0', 'sqrt',
			[tensor.val_ref], {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __sqrt__primitive = {
	name: 'sqrt',
	type: 'tensor',
	desc_function: __sqrt__desc_func,
	doc: new op_doc(['tensor'], ['elementwise square root'],
		'square root function')
}


/*
---------------------------------
------------- matmul  -----------
---------------------------------
*/
function __matmul__desc_func(tensor_trace, node, tensors){
	if(tensors.length!=2) throw({message: 'must take 2 tensors'})
	ensureAllTensors(tensors)
	if(tensors[0].dtype !== tensors[1].dtype){
		throw({message: 'tensors are of different dtypes'})
	}
	if(!tensors.every(t => t.shape.length >= 2)){
		throw({message: 'tensors must be of rank >=2'})
	}
	if(tensors[0].shape.length !== tensors[1].shape.length){
		throw({message: 'tensors are of different rank'})
	}
	if(!isNaN(tensors[0].shape.slice(-1)[0]) && 
		!isNaN(tensors[1].shape.slice(-2)[0]) && 
		tensors[0].shape.slice(-1)[0] !== tensors[1].shape.slice(-2)[0]){
		throw({message: 'shapes don\'t match (dimension -1 != dimension -2)'})
	}
	const prefix = tensors[0].shape.slice(0,-2).map((d1,i) => {
			const d2 = tensors[1].shape[i]
			if(!isNaN(d1) && !isNaN(d2)){
				if(d1!=d2){
					const message = 'tensors are not broadcastable '+
						`along dimension ${i}, with values ${[d1, d2]}`
					throw({message, metaData: {dims: [d1,d2], i},
						metaDataIdentifier: 'not_broadcastable'})	
				}
				return d1
			}
			if(isNaN(d1) && isNaN(d2)) return d1
			return isNaN(d1)? d1 : d2
		}),
		d1 = tensors[0].shape.slice(-2)[0],
		d2 = tensors[1].shape.slice(-1)[0],
		shape = new tensor_shape(prefix.concat([d1,d2])),
		dtype = tensors[0].dtype,
		out = new tensor_description(shape, dtype, node.name+':0', 'matmul',
			tensors.map(t => t.val_ref), {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __matmul__primitive = {
	name: 'matmul',
	type: 'tensor',
	desc_function: __matmul__desc_func,
	doc: new op_doc(['tensor 1', 'tensor 2'],
		['matrix multiplication of tensors'],
		'matrix multiplication of tensors')
}


/*
---------------------------------
-------------- add  -------------
---------------------------------
*/
function __add__desc_func(tensor_trace, node, tensors){
	if(tensors.length==0) throw({message: 'must take n>=1 tensors'})
	ensureAllTensors(tensors)
	const {shape, dtype} = broadcastShapes(tensors),
		out = new tensor_description(shape, dtype, node.name+':0', 'add',
			tensors.map(t => t.val_ref), {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __add__primitive = {
	name: 'add',
	type: 'tensor',
	desc_function: __add__desc_func,
	doc: new op_doc(['...tensor values'], ['sum of tensors'],
		'variadic function that adds n>=1 tensors')
}


/*
---------------------------------
----------- subtract  -----------
---------------------------------
*/
function __subtract__desc_func(tensor_trace, node, tensors){
	if(tensors.length!==2) throw({message: 'must take 2 tensors'})
	ensureAllTensors(tensors)
	const {shape, dtype} = broadcastShapes(tensors),
		out = new tensor_description(shape, dtype, node.name+':0', 'subtract',
			tensors.map(t => t.val_ref), {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __subtract__primitive = {
	name: 'subtract',
	type: 'tensor',
	desc_function: __subtract__desc_func,
	doc: new op_doc(['tensor 1', 'tensor 2'],
		['element-wise subtraction of tensors'],
		'subtracts 2 tensors element-wise')
}


/*
---------------------------------
------------ multiply  ----------
---------------------------------
*/
function __multiply__desc_func(tensor_trace, node, tensors){
	if(tensors.length==0) throw({message: 'must take n>=1 tensors'})
	ensureAllTensors(tensors)
	const {shape, dtype} = broadcastShapes(tensors),
		out = new tensor_description(shape, dtype, node.name+':0', 'multiply',
			tensors.map(t => t.val_ref), {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __multiply__primitive = {
	name: 'multiply',
	type: 'tensor',
	desc_function: __multiply__desc_func,
	doc: new op_doc(['...tensor values'],
		['element-wise product of tensors'],
		'variadic function that multiplies n>=1 tensors element-wise')
}


/*
---------------------------------
------------- divide  -----------
---------------------------------
*/
function __divide__desc_func(tensor_trace, node, tensors){
	if(tensors.length!==2) throw({message: 'must take 2 tensors'})
	ensureAllTensors(tensors)
	const {shape, dtype} = broadcastShapes(tensors),
		out = new tensor_description(shape, dtype, node.name+':0', 'divide',
			tensors.map(t => t.val_ref), {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __divide__primitive = {
	name: 'divide',
	type: 'tensor',
	desc_function: __divide__desc_func,
	doc: new op_doc(['tensor 1', 'tensor 2'],
		['element-wise division of tensors'],
		'divides 2 tensors element-wise')
}


/*
---------------------------------
-------------- pow  -------------
---------------------------------
*/
function __pow__desc_func(tensor_trace, node, tensors){
	if(tensors.length!==2) throw({message: 'must take 2 tensors'})
	ensureAllTensors(tensors)
	const {shape, dtype} = broadcastShapes(tensors),
		out = new tensor_description(shape, dtype, node.name+':0', 'pow',
			tensors.map(t => t.val_ref), {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __pow__primitive = {
	name: 'pow',
	type: 'tensor',
	desc_function: __pow__desc_func,
	doc: new op_doc(['tensor 1', 'tensor 2'],
		['The power of tensor 1 to tensor 2'],
		'The power of tensor 1 to tensor 2')
}


/*
---------------------------------
---------- get_tensor  ----------
---------------------------------
*/

function __get_tensor__desc_func(tensor_trace, node, inputs){
	let [shape, fill, dtype] = inputs
	if(shape == undefined) throw({message: 'shape must be defined'})
	if(fill == undefined) throw({message: 'fill must be defined'})
	dtype = dtype || 'float32'
	if(isTensor(dtype)) { dtype = dtype.dtype }
	if(isTensor(shape)){ shape = shape.shape }
	try{shape = new tensor_shape(shape)}
	catch(e){
		const message = 'Provided shape is not a valid tensor shape. ' +
			'A tensor shape must be a vector of integers or ' +
			'strings that are valid C identifiers.'
		throw({message})
	}
	if(shape.shape.some(x=> typeof(x)===typeof(''))){
		throw({message: 'Shape must not contain symbolic dimensions'})
	}
	const supported_fills = new Set(['ones', 'zeros',
		'normal', 'truncated_normal'])
	if(supported_fills.has(fill)){
		fill = {type: 'symbol', symbol: fill}
	} else if(!isNaN(+fill)){
		fill = {type: 'scalar', val: +fill}
	} else{
		const message = `Fill not supported: "${fill}". ` +
			'Must either be a number (as a string), or one of the following: '+
			[...supported_fills].map(a=>`"${a}"`).join(', ')
		throw({message})
	}
	const out = new tensor_description(shape, dtype, node.name+':0',
			'get_tensor', [], {shape, fill, dtype}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __get_tensor__primitive = {
	name: 'get_tensor',
	type: 'tensor',
	desc_function: __get_tensor__desc_func,
	doc: new op_doc(['shape, a vector or tensor whose shape will be inherited',
		'fill, one of (number, "ones", "zeros", "normal", "truncated_normal")',
		'(optional) dtype, either undefined, a string, ' +
			'or a tensor whose dtype will be inherited'],
	['tensor'], 'produces a tensor')
}

/*
---------------------------------
------------- scalar  -----------
---------------------------------
*/

function __scalar__desc_func(tensor_trace, node, inputs){
	let [number, dtype] = inputs
	dtype = dtype || 'float32'
	const shape = []
	if(isNaN(+number)){throw({message: 'First input must be a number'})}
	if(typeof(dtype) !== typeof('')){
		throw({message: 'Second input must be a string (or undefined)'})
	}
	return __get_tensor__desc_func(tensor_trace, node, [shape, number, dtype])
}

const __scalar__primitive = {
	name: 'scalar',
	type: 'control',
	desc_function: __scalar__desc_func,
	doc: new op_doc(['A number', '(optional) dtype'],
		['a scalar tensor'], 'produces a tensor from a scalar')
}


/*
---------------------------------
----------- variable  -----------
---------------------------------
*/
function __variable__desc_func(tensor_trace, node, inputs, collection_bins){
	if(!(inputs.length === 1 || inputs.length === 2)){
		throw({message: 'must take one or two inputs'})
	}
	let [tensor, collections] = inputs
	if(!isTensor(tensor)) throw({message: 'input #0 must be a tensor'})
	if(tensor.shape.some(x=> typeof(x)===typeof(''))){
		throw({message: 'Tensor must not contain symbolic dimensions'})
	}
	collections = collections || []
	collections = typeof(collections)===typeof('')? [collections] : collections
	if(!collections.every(s => typeof(s)===typeof(''))){
		throw({message: 'input #1 must be a string or list of strings'})
	}
	const name = `${node.name}:0`,
		{shape, dtype} = tensor,
		tshape = new tensor_shape(shape),
		out = new tensor_description(tshape, dtype, name, 'variable',
			[tensor.val_ref], {})
	collections.forEach(bin => {
		if(collection_bins.hasOwnProperty(bin)){
			collection_bins[bin][out.val_ref] = out
		} else {
			collection_bins[bin] = {[out.val_ref]: out}
		}
	})
	const results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __variable__primitive = {
	name: 'variable',
	type: 'tensor',
	desc_function: __variable__desc_func,
	doc: new op_doc(
		['tensor', '(optional) a bin or list of bins to add the tensor to'],
		['tensor'],
		'initializes tensor to provided value')
}


/*
---------------------------------
----------- identity  -----------
---------------------------------
*/
function __identity__desc_func(tensor_trace, node, inputs){
	return inputs.reduce((a,v,i) => {
		return Object.assign(a, {[node.name+':'+i]: v})
	}, {})
}

const __identity__primitive = {
	name: 'identity',
	type: 'control',
	desc_function: __identity__desc_func,
	doc: new op_doc(['...inputs'], ['...inputs'], 'forwards inputs, unchanged')
}



/*
---------------------------------
----------- literals  -----------
---------------------------------
*/
function __literals__desc_func(tensor_trace, node){
	return node.literal.reduce((a,v,i) => {
		return Object.assign(a, {[node.name+':'+i]: v})
	}, {}) 
}

const __literals__primitive = {
	name: 'literals',
	type: 'control',
	desc_function: __literals__desc_func,
	doc: new op_doc([], ['...literals'], 'forwards literals, unchanged')
}


/*
---------------------------------
---------- parse_json  ----------
---------------------------------
*/
function __parse_json__desc_func(tensor_trace, node, inputs){
	return inputs.reduce((a,v,i) => {
		try{return Object.assign(a, {[node.name+':'+i]: JSON.parse(v)})}
		catch(e){throw({message:`Couldn't parse JSON literal #${i}, "${v}"`})}
	}, {})
}

const __parse_json__primitive = {
	name: 'parse_json',
	type: 'control',
	desc_function: __parse_json__desc_func,
	doc: new op_doc(['...inputs (literals)'], ['...inputs'],
		'parses JSON literals')
}


/*
---------------------------------
------- parse_json_list  --------
---------------------------------
*/
function __parse_json_list__desc_func(tensor_trace, node, inputs){
	if(inputs.length != 1) throw({message: 'must take exactly 1 input'})
	if(typeof(inputs[0])!=='string') throw({message: 'input must be a string'})
	let parsed = ''
	try{parsed = JSON.parse(inputs[0])}
	catch(e){throw({message:'Couldn\'t parse JSON'})}
	if(!Array.isArray(parsed)) throw({message: 'value is not a list'})
	return parsed
		.reduce((acc,v,i) => Object.assign(acc, {[node.name+':'+i]: v}), {})
}

const __parse_json_list__primitive = {
	name: 'parse_json_list',
	type: 'control',
	desc_function: __parse_json_list__desc_func,
	doc: new op_doc(['JSON representation of a list'],
		['...parsed entries of list'],
		'parses a JSON representation of a list')
}




/*
---------------------------------
--------- if statement  ---------
---------------------------------
*/
function __if__desc_func(tensor_trace, node, inputs){
	if(typeof(inputs[0]) === 'boolean'){
		throw({message: 'first argument must be boolean'})
	}
	if(inputs.length != 3) throw({message: 'must take 3 arguments'})
	return {[node.name+':0']: inputs[0]? inputs[1] : inputs[2]}
}

const __if__primitive = {
	name: 'if',
	type: 'control',
	desc_function: __if__desc_func,
	doc: new op_doc(['boolean', 'value', 'value'], ['one of the values'],
		'forwards one of the values')
}



/*
---------------------------------
----------- pack_list  ----------
---------------------------------
*/
function __pack_list__desc_func(tensor_trace, node, inputs){
	return {[node.name+':0']: inputs}
}

const __pack_list__primitive = {
	name: 'pack_list',
	type: 'control',
	desc_function: __pack_list__desc_func,
	doc: new op_doc(['...values'], ['array containing the values'],
		'packs the input values into an array')
}

/*
---------------------------------
---------- unpack_list  ---------
---------------------------------
*/
function __unpack_list__desc_func(tensor_trace, node, inputs){
	if(!Array.isArray(inputs[0])) throw({message: 'input is not an array'})
	return inputs[0]
		.reduce((acc,v,i)=> Object.assign(acc, {[node.name+':'+i]: v}), {})
}

const __unpack_list__primitive = {
	name: 'unpack_list',
	type: 'control',
	desc_function: __unpack_list__desc_func,
	doc: new op_doc(['array of values'], ['...values'],
		'unpacks the input values from an array')
}



/*
---------------------------------
------------ softmax  -----------
---------------------------------
*/
function __softmax__desc_func(tensor_trace, node, inputs){
	if(inputs.length!==1) throw({message: 'must take 1 tensor'})
	ensureAllTensors(inputs)
	const tensor = inputs[0],
		shape = new tensor_shape(tensor.shape),
		dtype = tensor.dtype,
		out = new tensor_description(shape, dtype, node.name+':0', 'softmax',
			[tensor.val_ref], {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __softmax__primitive = {
	name: 'softmax',
	type: 'tensor',
	desc_function: __softmax__desc_func,
	doc: new op_doc(['tensor'], ['softmax of tensor'],
		'applies the softmax function to a tensor')
}



/*
---------------------------------
------------- log  --------------
---------------------------------
*/
function __log__desc_func(tensor_trace, node, inputs){
	if(inputs.length!==1) throw({message: 'must take 1 tensor'})
	ensureAllTensors(inputs)
	const tensor = inputs[0],
		shape = new tensor_shape(tensor.shape),
		dtype = tensor.dtype,
		out = new tensor_description(shape, dtype, node.name+':0', 'log',
			[tensor.val_ref], {}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __log__primitive = {
	name: 'log',
	type: 'tensor',
	desc_function: __log__desc_func,
	doc: new op_doc(['tensor'], ['natural log of tensor'],
		'applies the natural log function to a tensor')
}

/*
---------------------------------
---------- reduce_sum  ----------
---------------------------------
*/
function __reduce_sum__desc_func(tensor_trace, node, inputs){
	if(!(inputs.length == 1 || inputs.length == 2)){
		throw({message: 'must take one or two inputs'})
	}
	if(!isTensor(inputs[0])){
		throw({message: 'input must be a tensor'})	
	} 
	const tensor = inputs[0],
		default_perm = tensor.shape.map((_,i)=>i),
		axis = !isNaN(inputs[1])? [inputs[1]] :
			[...(new Set(inputs[1] || default_perm))].sort()
	if(!axis.every(x=> !isNaN(x) && 0<=x && x<tensor.shape.length)){
		throw({message: `axis out of bounds: ${axis}`})
	}
	const dtype = tensor.dtype,
		raw_shape = tensor.shape
			.reduce((acc,v,i) => axis.includes(i)? acc : [...acc,v], []),
		shape = new tensor_shape(raw_shape),
		out = new tensor_description(shape, dtype, node.name+':0', 'reduce_sum',
			[tensor.val_ref], {axis:axis}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __reduce_sum__primitive = {
	name: 'reduce_sum',
	type: 'tensor',
	desc_function: __reduce_sum__desc_func,
	doc: new op_doc(['tensor', 'axis; a integer or array of integers'],
		['a scalar'], 'sums a tensor')
}

/*
---------------------------------
---------- reduce_avg  ----------
---------------------------------
*/
function __reduce_avg__desc_func(tensor_trace, node, inputs){
	if(!(inputs.length == 1 || inputs.length == 2)){
		throw({message: 'must take one or two inputs'})
	}
	if(!isTensor(inputs[0])) throw({message: 'input must be a tensor'})
	const tensor = inputs[0],
		default_perm = tensor.shape.map((_,i)=>i),
		axis = !isNaN(inputs[1])? [inputs[1]] :
			[...(new Set(inputs[1] || default_perm))].sort()
	if(!axis.every(x=> !isNaN(x) && 0<=x && x<tensor.shape.length)){
		throw({message: `axis out of bounds: ${axis}`})
	}
	const dtype = tensor.dtype,
		raw_shape = tensor.shape
			.reduce((acc,v,i) => axis.includes(i)? acc : [...acc,v], []),
		shape = new tensor_shape(raw_shape),
		out = new tensor_description(shape, dtype, node.name+':0', 'reduce_avg',
			[tensor.val_ref], {axis:axis}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __reduce_avg__primitive = {
	name: 'reduce_avg',
	type: 'tensor',
	desc_function: __reduce_avg__desc_func,
	doc: new op_doc(['tensor', 'axis; a integer or array of integers'],
		['a scalar'],
		'averages a tensor')
}

/*
---------------------------------
----------- transpose  ----------
---------------------------------
*/
function __transpose__desc_func(tensor_trace, node, inputs){
	if(!(inputs.length == 1 || inputs.length == 2)){
		throw({message: 'must take one or two inputs'})
	}
	if(!isTensor(inputs[0])) throw({message: 'first input must be a tensor'})
	const tensor = inputs[0],
		default_perm = Array(tensor.shape.length).fill()
			.map((_,i)=>i).reverse(),
		default_perm_set = new Set(default_perm),
		perm = inputs[1] || default_perm
	if(!(default_perm.length==perm.length &&
		perm.every(v => default_perm_set.has(v)))){
		throw({message: 'permutation isn\'t a permutation of 0...n-1, ' +
			`recieved ${perm}`})
	}
	const dtype = tensor.dtype,
		shape = new tensor_shape(perm.map(i=>tensor.shape[i])),
		out = new tensor_description(shape, dtype, node.name+':0', 'transpose',
			[tensor.val_ref], {perm:perm}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __transpose__primitive = {
	name: 'transpose',
	type: 'tensor',
	desc_function: __transpose__desc_func,
	doc: new op_doc(['tensor', 'permutation (optional)'],
		['tensor with permuted dimensions'],
		'permutes the dimensions of tensor according to ' +
		'the supplied permutation')
}

/*
---------------------------------
------------ one_hot  -----------
---------------------------------
*/
function __one_hot__desc_func(tensor_trace, node, inputs){
	if(inputs.length != 2) throw({message: 'must take two inputs'})
	if(!(isTensor(inputs[0]) && inputs[0].shape.length == 1)){
		throw({message: 'first input must be a rank 1 tensor'})
	}
	if(isNaN(inputs[1]) || Math.floor(+inputs[1])<2){
		throw({message: 'second input must be a number >=2'})
	}
	const tensor = inputs[0],
		n_colls = Math.floor(+inputs[1]),
		dtype = tensor.dtype,
		shape = new tensor_shape([tensor.shape[0], n_colls]),
		out = new tensor_description(shape, dtype, node.name+':0', 'one_hot',
			[tensor.val_ref], {n_colls:n_colls}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __one_hot__primitive = {
	name: 'one_hot',
	type: 'tensor',
	desc_function: __one_hot__desc_func,
	doc: new op_doc(['indices (as rank 1 tensor)', 'number of columns'],
		['matrix with one hot vectors as rows'],
		'constructs a matrix where each row is a one hot vector, ' +
		'with n_colls columns and one row for each index')
}

/*
---------------------------------
------------- cast  -------------
---------------------------------
*/
function __cast__desc_func(tensor_trace, node, inputs){
	if(inputs.length != 2) throw({message: 'must take two inputs'})
	const [tensor, given_dtype] = inputs
	if(!isTensor(tensor)) throw({message: 'first input must be a tensor'})
	if(!(typeof(given_dtype) == 'string' || isTensor(given_dtype))){
		throw({message: 'second input must be a string or a tensor'})	
	}
	const dtype = isTensor(given_dtype)? given_dtype.dtype : given_dtype,
		shape = new tensor_shape(tensor.shape),
		out = new tensor_description(shape, dtype, node.name+':0', 'cast',
			[tensor.val_ref], {dtype:dtype}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __cast__primitive = {
	name: 'cast',
	type: 'tensor',
	desc_function: __cast__desc_func,
	doc: new op_doc(['tensor', 'dtype (a string)'],
		['tensor cast as dtype'],
		'casts a tensor to a specified dtype')
}

/*
---------------------------------
------------ gather  ------------
---------------------------------
*/
function __gather__desc_func(tensor_trace, node, inputs){
	if(!(inputs.length === 2 || inputs.length === 3)){
		throw({message: 'must take two or three inputs'})
	}
	const [tensor, indices] = inputs
	const axis = inputs[2]? inputs[2] : 0
	ensureAllTensors(inputs.slice(0,2))
	// checking tensor
	if(tensor.shape.length == 0){
		throw({message: 'first input must not be a scalar'})
	}
	// checking indices
	if(indices.dtype !== 'int32'){
		throw({message: 'second input must have dtype "int32", instead got "'+
			indices.dtype+'"'})
	}
	if(indices.shape.length !== 1){
		throw({message: 'second input must be one dimensional'})
	}
	// checking axis
	if(!(Number.isInteger(axis) && axis>=0 && axis<tensor.shape.length)){
		throw({message: 'third input must be an integer between 0-'+
			tensor.shape.length-1})
	}
	let shape = tensor.shape.slice()
	shape[axis] = indices.shape[0]
	shape = new tensor_shape(shape)
	const out = new tensor_description(shape, tensor.dtype, node.name+':0',
		'gather', [tensor.val_ref, indices.val_ref], {axis})
	const results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __gather__primitive = {
	name: 'gather',
	type: 'tensor',
	desc_function: __gather__desc_func,
	doc: new op_doc(['x', 'indices (1d tensor with dtype "int32")',
		'(optional) axis'],
	['tensor of slices from `x`'],
	'takes slices from `x` along `axis` at the specified `indices`')
}

/*
---------------------------------
--------- gather_rows  ----------
---------------------------------
*/
function __gather_rows__desc_func(tensor_trace, node, inputs){
	if(inputs.length !== 2){
		throw({message: 'must take two inputs'})
	}
	const [x, colls] = inputs
	if(!(isTensor(x) && x.shape.length>=2)){
		throw({message: 'first input must be a tensor of rank>=2'})
	}
	if(!(isTensor(colls) && colls.shape.length==1)){
		throw({message: 'second input must be a tensor of '+
			'rank 1 with dtype "int32"'})
	}
	if(x.shape[0] !== colls.shape[0]){
		throw({message: 'first dimensions must match, '+
			`(${x.shape[0]} != ${colls.shape[0]})`})
	}
	let shape = [x.shape[0], ...x.shape.slice(2)]
	shape = new tensor_shape(shape)
	const out = new tensor_description(shape, x.dtype, node.name+':0',
		'gather_rows', [x.val_ref, colls.val_ref], {})
	const results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __gather_rows__primitive = {
	name: 'gather_rows',
	type: 'tensor',
	desc_function: __gather_rows__desc_func,
	doc: new op_doc(['x', 'indices (1d tensor with dtype "int32")'],
		['tensor of slices from rows of `x` at the provided indices'],
		'takes slices from rows of `x` along at the provided `indices`')
}

/*
---------------------------------
----------- reshape  ------------
---------------------------------
*/
function __reshape__desc_func(tensor_trace, node, inputs){
	if(inputs.length != 2) throw({message: 'must take two inputs'})
	let [tensor, newShape] = inputs
	if(!isTensor(tensor)) throw({message: 'first input must be a tensor'})
	// checking shape
	newShape = Array.isArray(newShape)? newShape : [newShape]
	const oldSymbols = JSON.stringify(tensor.shape.filter(isNaN).sort())
	const newSymbols = JSON.stringify(newShape.filter(isNaN).sort())
	if(oldSymbols !== newSymbols){
		throw({message: 'Symbolic dimensions did not match.'})
	}
	if(!newShape.filter(x=>!isNaN(x)).every(x=>Number.isInteger(x)&&x>=0)){
		throw({message: 'Dimensions must be nonnegative integers.'})
	}
	const getSize = arr => arr.filter(x=>!isNaN(x)).reduce((a,b) => a*b, 1)
	if(getSize(newShape) !== getSize(tensor.shape)){
		throw({message: 'Sizes do not match'})
	}
	const shapeEncoding = newShape.map(x => !isNaN(x)? x :
		''+tensor.shape.indexOf(x))
	try {
		newShape = new tensor_shape(newShape)
	} catch(message){ throw({message}) }
	const out = new tensor_description(newShape, tensor.dtype, node.name+':0',
		'reshape', [tensor.val_ref], {shapeEncoding})
	const results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __reshape__primitive = {
	name: 'reshape',
	type: 'tensor',
	desc_function: __reshape__desc_func,
	doc: new op_doc(['x', 'shape (array of nonnegative integers)'],
		['`x` reshaped to given `shape`'],
		'reshapes `x` into given shape `shape`')
}


/*
---------------------------------
---------- js_function  ----------
---------------------------------
*/
function __js_function__desc_func(tensor_trace, node, inputs){
	let fn = undefined
	let result = undefined
	try {
		fn = eval(node.literal[0])
	} catch(e){
		throw({message: 'Could not evaluate function string, '+
			`got error: ${e.toString()}`})
	}
	if(typeof(fn) !== 'function'){
		throw({message: 'Function string did not evaluate to a function, '+
			`instead got type "${typeof(fn)}"`})
	}
	try {
		result = fn(...inputs)
	} catch(e){
		throw({message: `Error in applying function: ${e.toString()}`})
	}
	const resultsArray = Array.isArray(result)? result : [result]
	return resultsArray.reduce((acc, res, i) =>
		Object.assign(acc, {[node.name+':'+i]: res}), {})
}

const __js_function__primitive = {
	name: 'js_function',
	type: 'control',
	desc_function: __js_function__desc_func, 
	doc: new op_doc(['...arguments'],
		['the outputs of the function applied to the arguments'],
		'applies the function to the arguments, and returns the results')
}

/*
---------------------------------
-------- get_collection  --------
---------------------------------
*/
function __get_collection__desc_func(tensor_trace, node, inputs, coll_bins){
	const collections = Array.isArray(inputs[0])? inputs[0] : [inputs[0]]
	if(!collections.every(s => typeof(s)===typeof(''))){
		throw({message: 'Input must be a string or list of strings'})
	}
	const dict = collections
		.filter(name => coll_bins.hasOwnProperty(name))
		.map(name => coll_bins[name])
		.reduce((acc, coll) => Object.assign(acc,coll), {})
	const results = {[`${node.name}:0`]: Array.from(Object.values(dict))}
	return results
}

const __get_collection__primitive = {
	name: 'get_collection',
	type: 'control',
	desc_function: __get_collection__desc_func,
	doc: new op_doc(
		['collection name, or list of names, as strings',
			'...optional control edges'],
		['list of tensors in the specified collections'],
		'finds a list of tensors in the specified collection(s)')
}

/*
---------------------------------
---------- batch_norm  ----------
---------------------------------
*/
function __batch_norm__desc_func(tensor_trace, node, inputs, coll_bins){
	const tensor = inputs[0]
	if(!isTensor(tensor)){throw({message: 'First input must be tensor'})}
	if(tensor.shape.slice(1).some(x=> typeof(x)===typeof(''))){
		throw({message: 'Tensor must not contain symbolic dimensions'+
			' (except for first dimension)'})
	}
	const shape = [1, ...tensor.shape.slice(1)]
	const dtype = tensor.dtype
	const newNode = ext =>  Object.assign({}, node, {name: node.name+ext})
	const bins = ['trainable', 'batchNorm']
	const getValue = (name, fill) => {
		const nodeInit = newNode(`/${name}/init`)
		const nodeVar = newNode(`/${name}/variable`)
		const init = Object.values(__get_tensor__desc_func(
			tensor_trace, nodeInit, [shape, fill, dtype]))[0]
		return Object.values(__variable__desc_func(
			tensor_trace, nodeVar, [init, bins], coll_bins))[0]
	}
	const mean = getValue('mean', 0)
	const variance = getValue('variance', 1)
	const scale = getValue('scale', 1)
	const offset = getValue('offset', 0)
	// batch norm
	const newShape = new tensor_shape(tensor.shape)
	const valRefs = [tensor, mean, variance, scale, offset].map(t=>t.val_ref)
	const out = new tensor_description(newShape, dtype, node.name+':0',
		'batch_norm', valRefs, {})
	const results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __batch_norm__primitive = {
	name: 'batch_norm',
	type: 'tensor',
	desc_function: __batch_norm__desc_func,
	doc: new op_doc(['input tensor'], ['normalized tensor'],
		'applies batch normalization to the input')
}


/*
---------------------------------
--------- convolution  ----------
---------------------------------
*/
const __convolution__primitive = {
	name: 'convolution',
	type: 'tensor',
	desc_function: Convolution.__convolution__desc_func,
	doc: new op_doc(['x', 'filter', '(optional) stride', '(optional) padding'],
		['x convolved with filter'],
		'convolves x with filter')
}

/*
---------------------------------
----------- max_pool  -----------
---------------------------------
*/
const __max_pool__primitive = {
	name: 'max_pool',
	type: 'tensor',
	desc_function: Convolution.__max_pool__desc_func,
	doc: new op_doc(['x', '(optional) filterSize',
		'(optional) stride', '(optional) padding'],
	['max pooling of x'],
	'applies max pooling to x')
}

/*
---------------------------------
----------- avg_pool  -----------
---------------------------------
*/
const __avg_pool__primitive = {
	name: 'avg_pool',
	type: 'tensor',
	desc_function: Convolution.__avg_pool__desc_func,
	doc: new op_doc(['x', '(optional) filterSize',
		'(optional) stride', '(optional) padding'],
	['average pooling of x'],
	'applies average pooling to x')
}


export const primitives = [
	__placeholder__primitive,
	__relu__primitive,
	__sigmoid__primitive,
	__tanh__primitive,
	__exp__primitive,
	__matmul__primitive,
	__add__primitive,
	__multiply__primitive,
	__divide__primitive,
	__identity__primitive,
	__if__primitive,
	__scalar__primitive,
	__literals__primitive,
	__parse_json__primitive,
	__parse_json_list__primitive,
	__pow__primitive,
	__sqrt__primitive,
	__get_tensor__primitive,
	__variable__primitive,
	__pack_list__primitive,
	__unpack_list__primitive,
	__softmax__primitive,
	__log__primitive,
	__reduce_sum__primitive,
	__negate__primitive,
	__transpose__primitive,
	__one_hot__primitive,
	__cast__primitive,
	__reduce_avg__primitive,
	__subtract__primitive,
	__abs__primitive,
	__convolution__primitive,
	__gather__primitive,
	__reshape__primitive,
	__js_function__primitive,
	__get_collection__primitive,
	...higherOrderPrimitives,
	__batch_norm__primitive,
	__gather_rows__primitive,
	__max_pool__primitive,
	__avg_pool__primitive,
].reduce((a,p)=>Object.assign(a, {[p.name]: p}), {})

