import {primitives} from './operations.js'
import {constructors} from '../util/taffy_constructors.js'

const {op_doc} = constructors

function executeModule(moduleName, inputs, parentArgs){
	const [tensor_trace, parentNode, , collections, modules] = parentArgs
	const module = modules[moduleName]
	const prefix = parentNode.name+'/'
	let valueTrace = module.input.map(s=>prefix+s+':0').reduce((acc,k,i) =>
		Object.assign(acc, {[k]:inputs[i]}), {})
	module.nodes.forEach(nodeOrig => {
		const node = Object.assign({}, nodeOrig, {name: prefix+nodeOrig.name})
		if(node.op === 'placeholder'){return}
		const fn = nodeIn => primitives[node.op]
			.desc_function(tensor_trace, node,nodeIn, collections, modules)
		const fnOut = fn(node.input.map(ref => valueTrace[prefix+ref]))
		Object.assign(valueTrace, fnOut)
	})
	const result = module.output.map(s=>prefix+s).reduce((acc, k) =>
		Object.assign(acc, {[k]:valueTrace[k]}), {})
	return result
}

function assertListOfArrays(inputs){
	if(!inputs.every(a => Array.isArray(a))){
		throw({message: 'inputs must be arrays'})
	}
}

const makeOpFn = (op, args) => {
	const [tensor_trace, node, , colls, modules] = args
	if(primitives.hasOwnProperty(op)){
		return inputs => primitives[op]
			.desc_function(tensor_trace, node, inputs, colls, modules)
	} else if(modules.hasOwnProperty(op)){
		return inputs => executeModule(op, inputs, args)
	} else {
		throw({message: `"${op}" is not a primitive or module name`})
	}
}

/*
---------------------------------
-------------- map --------------
---------------------------------
*/
function __map__desc_func(...args){
	const [, node, inputs] = args
	const op = node.literal[0]
	const opFn = makeOpFn(op, args)
	assertListOfArrays(inputs)
	if(!inputs.every(a => a.length===inputs[0].length)){
		throw({message: 'inputs must be of same length'})
	}
	const transposed = inputs[0].map((_,i) => inputs.map(r => r[i]))
	const results = transposed.map(opFn).map(o => Object.values(o)[0])
	return {[`${node.name}:0`]: results}
}

const __map__primitive = {
	name: 'map',
	type: 'control',
	desc_function: __map__desc_func,
	doc: new op_doc(['...rows'],
		['the result of applying the operation to each column'],
		'maps the specified operation across columns')
}

/*
---------------------------------
------------- reduce ------------
---------------------------------
*/
function __reduce__desc_func(...args){
	const [, node, inputs] = args
	const op = node.literal[0]
	const opFn = makeOpFn(op, args)
	if(!Array.isArray(inputs[0])){
		throw({message: 'First input must be an array'})
	}
	const getFirst = o => Object.values(o)[0]
	const reducer = (a,b) => getFirst(opFn([a,b]))
	const results = inputs.length==1? inputs[0].reduce(reducer) :
		inputs[0].reduce(reducer, inputs[1])
	return {[`${node.name}:0`]: results}
}

const __reduce__primitive = {
	name: 'reduce',
	type: 'control',
	desc_function: __reduce__desc_func,
	doc: new op_doc(['array of values', '(optional) initial accumulator'],
		['The resulting accumulator'],
		'Executes the operation along the array, ie f(x3,f(x1,x2)). '+
        'The operation takes the accumulator as the first argument.')
}

/*
---------------------------------
----------- reductions ----------
---------------------------------
*/
function __reductions__desc_func(...args){
	const [, node, inputs] = args
	const op = node.literal[0]
	const opFn = makeOpFn(op, args)
	if(!Array.isArray(inputs[0])){
		throw({message: 'First input must be an array'})
	}
	const getFirst = o => Object.values(o)[0]
	const reducer = (a,b) => getFirst(opFn([a,b]))
	const fullReductions = (arr, init) => arr.reduce((acc,v) => 
		acc.concat([reducer(acc.slice(-1)[0], v)]), [init])
	const results = inputs.length>1? fullReductions(...inputs) :
		fullReductions(inputs[0].slice(1), inputs[0][0])
	return {[`${node.name}:0`]: results}
}

const __reductions__primitive = {
	name: 'reductions',
	type: 'control',
	desc_function: __reductions__desc_func,
	doc: new op_doc(['array of values', '(optional) initial accumulator'],
		['An array of the history of the accumulator'],
		'Executes the operation along the array, ie x1, f(x1,x2), '+
        'f(x3,f(x1,x2)),... .'+
        'The operation takes the accumulator as the first argument.')
}

/*
---------------------------------
------------- apply -------------
---------------------------------
*/
function __apply__desc_func(...args){
	const [, node, inputs] = args
	const op = node.literal[0]
	const opFn = makeOpFn(op, args)
	if(!Array.isArray(inputs[0])){
		throw({message: 'First input must be an array'})
	}
	const results = opFn(inputs[0])
	return {[`${node.name}:0`]: Object.values(results)[0]}
}

const __apply__primitive = {
	name: 'apply',
	type: 'control',
	desc_function: __apply__desc_func,
	doc: new op_doc(['array of values'],
		['The result of applying the operation to the values'],
		'Executes the operation on the values in the array. ')
}

export const higherOrderPrimitives = [__map__primitive,
	__reduce__primitive, __apply__primitive, __reductions__primitive]
