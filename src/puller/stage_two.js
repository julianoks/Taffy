import {constructors} from '../util/taffy_constructors.js'
import {primitives} from '../ops/operations.js'

const isShape = v => {
	try {return v.constructor === constructors.tensor_shape}
	catch(e){return false}
}

const isTensor = v => {
	try {return v.constructor === constructors.tensor_description}
	catch(e){return false}
}

function quasiToTensor(inputDesc){
	return Object.entries(inputDesc).reduce((a,[k,quasi]) => {
		if(!quasi.hasOwnProperty('shape')){
			throw('input description to stage two must have `shape` key')
		}
		if(!quasi.hasOwnProperty('dtype')){
			throw('input description to stage two must have `dtype` key')
		}
		const shape = isShape(quasi.shape)? quasi.shape :
				new constructors.tensor_shape(quasi.shape),
			tensorDesc = new constructors.tensor_description(shape,
				quasi.dtype, k+':0', 'placeholder', [], {})
		return Object.assign(a, {[k+':0']: tensorDesc})
	}, {})
}

// inputDescriptions must be a dictionary with
// the names of inputs as keys and value descriptions as values
// all value descriptions must be quasi-tensor descriptions,
// which are objects with shape and dtype

/**
 * Evaluates a specified module using description of its input, 
 * producing a tensor and value trace.
 * @param {Object<string, Object<string, any>>} stageOneOut The output 
 * of `stage_on`, which are flattened modules
 * @param {string} moduleName The name of the module to be compiled
 * @param {Object<string, Object<string, any>>} inputDescriptions A 
 * dictionary with `moduleName`'s input names as keys, 
 * and dictionary with {shape, dtype} as values
 * @return {Object<string, any>} A dictionary containing a tensor 
 * and value trace, and other metadata.
 */
export function stage_two(stageOneOut, moduleName, inputDescriptions){
	let valueTrace = {},
		tensorTrace = {},
		collections = {}
	Object.entries(quasiToTensor(inputDescriptions)).forEach(([valRef, val])=>{
		const pair = {[valRef]: val}
		Object.assign(valueTrace, pair)
		Object.assign(tensorTrace, pair)
	})
	const flatModule = stageOneOut.modules[moduleName],
		inputNames = new Set(flatModule.input)
	flatModule.nodes.forEach(node => {
		if(inputNames.has(node.name)){return} // inputs already recieved traces
		const fn = inputs => primitives[node.op]
			.desc_function(tensorTrace, node, inputs, collections)
		try {
			const fnOut = fn(node.input.map(ref => valueTrace[ref]))
			Object.assign(valueTrace, fnOut)
		} catch(error){ throw {error, node: node.name, valueTrace}}
	})
	const outputs = flatModule.output.map(k => valueTrace[k])
	outputs.forEach((t, i) => {if(!isTensor(t)){
		const message = `Output #${i} of module is not a tensor`,
			metaData = {i, arg:t, valueTrace}
		throw({message,  metaData, metaDataIdentifier: 'output_not_tensor'})
	}})
	return {val_trace: valueTrace,
		tensor_trace: tensorTrace,
		collections,
		output: outputs.map(t=>t.val_ref),
		output_names: flatModule.output,
		name: moduleName,
		input_descriptions: inputDescriptions,
		input_names: flatModule.input}
}
