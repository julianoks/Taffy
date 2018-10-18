import {constructors} from '../util/taffy_constructors.js'
import {__convolution__desc_func} from './convolution.js'

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
------------- scalar  -----------
---------------------------------
*/
function __scalar__desc_func(tensor_trace, node, inputs){
	if(inputs.length < 1) throw({message: 'must take at least one input'})
	if(isNaN(+inputs[0])) throw({message: 'first input must be a number'})
	const num = +inputs[0],
		dtype = typeof(inputs[1])==='string'? inputs[1] : 'float32',
		shape = new tensor_shape([]),
		out = new tensor_description(shape, dtype, node.name+':0', 'scalar',
			[], {num:num,dtype:dtype}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

const __scalar__primitive = {
	name: 'scalar',
	type: 'tensor',
	desc_function: __scalar__desc_func,
	doc: new op_doc(['scalar value', '(optional) dtype'], ['a scalar'],
		'produces a tensor scalar')
}


/*
---------------------------------
---------- get_tensor  ----------
---------------------------------
*/

function __get_tensor__desc_func(tensor_trace, node, inputs, collection_bins){
	let [shape, fill, dtype, collections] = inputs
	if(shape == undefined) throw({message: 'shape must be defined'})
	if(fill == undefined) throw({message: 'fill must be defined'})
	dtype = dtype || 'float32'
	collections = collections || []
	if(isTensor(dtype)) { dtype = dtype.dtype }
	if(isTensor(shape)){ shape = shape.shape }
	try{shape = new tensor_shape(shape)}
	catch(e){
		const message = 'Provided shape is not a valid tensor shape. ' +
			'A tensor shape must be a vector of integers or ' +
			'strings that are valid C identifiers.'
		throw({message})
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
	collections.forEach(bin => {
		if(collection_bins.hasOwnProperty(bin)){
			collection_bins[bin][out.val_ref] = out
		} else {
			collection_bins[bin] = {[out.val_ref]: out}
		}
	})
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
			'or a tensor whose dtype will be inherited',
		'(optional) a list of bins to add the tensor to'],
	['tensor'], 'produces a tensor')
}

/*
---------------------------------
----------- variable  -----------
---------------------------------
*/
function __variable__desc_func(tensor_trace, node, inputs){
	if(inputs.length != 1) throw({message: 'must take exactly 1 input'})
	if(!isTensor(inputs[0])) throw({message: 'input must be a tensor'})
	const results = inputs.reduce((acc,v,i)=> {
		const name = node.name+':' + i,
			{shape, dtype} = v,
			tshape = new tensor_shape(shape),
			new_tensor = new tensor_description(tshape, dtype, name, 'variable',
				[v.val_ref], {})
		return Object.assign(acc, {[name]: new_tensor})
	}, {})
	Object.assign(tensor_trace, results)
	return results
}

const __variable__primitive = {
	name: 'variable',
	type: 'tensor',
	desc_function: __variable__desc_func,
	doc: new op_doc(['tensor'], ['tensor'],
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
----------- reshape  ------------
---------------------------------
*/
function __reshape__desc_func(tensor_trace, node, inputs){
	if(inputs.length != 2) throw({message: 'must take two inputs'})
	const [tensor, newShape] = inputs
	if(!isTensor(tensor)) throw({message: 'first input must be a tensor'})
	// checking shape
	if(!(Array.isArray(newShape) &&
		newShape.every(x => Number.isInteger(x) && x>=0))){
		throw({message: 'second input must be an array of '+
			'nonnegative integers'})
	}
	const oldSize = tensor.shape.reduce((a,b) => a*b, 1)
	const proposedSize = newShape.reduce((a,b) => a*b, 1)
	if(oldSize !== proposedSize){
		throw({message: `Size of new shape, ${proposedSize},`+
			` must match original size, ${oldSize}.`})
	}
	const out = new tensor_description(newShape, tensor.dtype, node.name+':0',
		'reshape', [tensor.val_ref], {newShape})
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
	const [fnString, ...args] = inputs
	let fn = undefined
	let result = undefined
	try {
		fn = eval(fnString)
	} catch(e){
		throw({message: 'Could not evaluate function string, '+
			`got error: ${e.toString()}`})
	}
	if(typeof(fn) !== 'function'){
		throw({message: 'Function string did not evaluate to a function, '+
			`instead got type "${typeof(fn)}"`})
	}
	try {
		result = fn(...args)
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
	doc: new op_doc(['javascript function (a string)', '...arguments'],
		['the outputs of the function applied to the arguments'],
		'applies the function to the arguments, and returns the results')
}

/*
---------------------------------
--------- convolution  ----------
---------------------------------
*/
const __convolution__primitive = {
	name: 'convolution',
	type: 'tensor',
	desc_function: __convolution__desc_func,
	doc: new op_doc(['x', 'filter', '(optional) stride', '(optional) padding'],
		['x convolved with filter'],
		'convolves x with filter')
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
].reduce((a,p)=>Object.assign(a, {[p.name]: p}), {})

