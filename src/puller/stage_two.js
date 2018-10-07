import {constructors} from '../util/taffy_constructors.js'
import {primitives} from '../ops/operations.js'

const isShape = v => v.constructor === constructors.tensor_shape
const isTensor = v => v.constructor === constructors.tensor_description

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
export function stage_two(stageOneOut, moduleName, inputDescriptions){
	let valueTrace = {},
		tensorTrace = {}
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
			.desc_function(tensorTrace, node, inputs)
		try {
			const fnOut = fn(node.input.map(ref => valueTrace[ref]))
			Object.assign(valueTrace, fnOut)
		} catch(error){ throw {error, node: node.name}}
	})
	const outputs = flatModule.output.map(k => valueTrace[k])
	outputs.forEach((t, i) => {if(!isTensor(t)){
		const message = `Output #${i} of module is not a tensor`,
			metaData = {i, arg:t}
		throw({message,  metaData, metaDataIdentifier: 'output_not_tensor'})
	}})
	return {val_trace: valueTrace,
		tensor_trace: tensorTrace,
		output: outputs.map(t=>t.val_ref),
		output_names: flatModule.output,
		name: moduleName,
		input_descriptions: inputDescriptions}
}
