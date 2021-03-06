import {constructors} from '../util/taffy_constructors.js'

const {tensor_description, tensor_shape} = constructors

const isTensor = obj => obj.constructor === tensor_description

const zip = (...arrs) => arrs[0].map((_,i) => arrs.map(a=>a[i]))

function strideToArray(stride, filter){
	if(Array.isArray(stride)) return stride
	const nDims = filter.shape.length - 2
	return Array(nDims).fill(stride)
}

function getConvOutShape(x, outChannels, filterInDims, stride, padding){
	const batchSize = x.shape.slice(0, 1)
	const middleInDims = x.shape.slice(1, -1)
	let middleOutDims = []
	if(padding === 'same'){
		middleOutDims = zip(middleInDims, stride)
			.map(([inD, s]) => Math.ceil(inD / s))
	}
	if(padding === 'valid'){
		middleOutDims = zip(middleInDims, filterInDims, stride)
			.map(([inD, f, s]) => Math.ceil((1 + inD - f) / s))
	}
	if(Number.isInteger(+padding)){
		middleOutDims = zip(middleInDims, filterInDims, stride)
			.map(([inD, f, s]) => 1 + ((inD - f + (2 * padding)) / s))
		if(!middleOutDims.every(Number.isInteger)){
			throw({message: 'Invalid output shape (not an integer). ' +
				'Please change the stride or padding.'})
		}
	}
	const resultShape = [].concat(batchSize, middleOutDims, outChannels)
	return new tensor_shape(resultShape)
}

export function __convolution__desc_func(tensor_trace, node, inputs){
	if(inputs.length < 2) throw({message: 'must take at least two inputs'})
	if(!inputs.slice(0,2).every(isTensor)){
		throw({message: '`x` and `filter` must be tensors'})
	}
	if(inputs[0].shape.length !== inputs[1].shape.length){
		throw({message: `x (rank ${inputs[0].shape.length}) and ` +
			`filter (rank ${inputs[1].shape.length}) must be the same rank`})
	}
	if(inputs[0].shape.length < 3){
		throw({message: '`x` and `filter` must be of rank 3 or greater'})
	}
	if(inputs[0].shape.slice(-1)[0] !== inputs[1].shape.slice(-2)[0]){
		throw({message: 'The second to last dimension of x ' +
			`(shape ${inputs[0].shape}) ` +
			'should equal the last dimension of filter ' +
			`(shape ${inputs[1].shape})`})
	}
	if(inputs[2] !== undefined){
		if(!isNaN(inputs[2])){
			if(!(Number.isInteger(inputs[2]) && inputs[2]>0)){
				const message = 'if `stride` is a number, ' +
					'it must be a positive integer'
				throw({message})
			}
		} else if(Array.isArray(inputs[2])){
			if(inputs[2].shape.length !== (inputs[0].shape.length - 2)){
				throw({message: 'If `stride` is an array, ' +
					'it must have 2 fewer dimensions than `x`'})
			}
			if(!inputs[2].every(n => Number.isInteger(n) && n>0)){
				throw({message: '`stride` must only contain positive integers'})
			}
		}
	}
	if(!(inputs[3] === undefined ||
			inputs[3] === 'same' ||
			inputs[3] === 'valid' ||
			(Number.isInteger(+inputs[3]) && (+inputs[3] >= 0)))){
		const message = '`padding` must either be "same", "valid", ' +
			'or a non-negative integer'
		throw({message})
	}
	const [x, filter] = inputs.slice(0,2),
		stride = strideToArray(inputs[2] || 1, filter),
		padding = inputs[3] || 'same'
	const dtype = x.dtype,
		shape = getConvOutShape(x,
			filter.shape.slice(-1), filter.shape.slice(1, -1),
			stride, padding),
		out = new tensor_description(shape, dtype, node.name+':0',
			'convolution',
			[x.val_ref, filter.val_ref],
			{stride, padding, shape: shape.shape}),
		results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}



const poolGenericDescFunc = opName => (tensor_trace, node, inputs) => {
	if(inputs.length < 1) throw({message: 'must take at least one input'})
	let [x, filterSize, stride, padding] = inputs
	filterSize = filterSize || 2
	stride = stride || 1
	padding = padding || 'valid'
	if(!(isTensor(x) && x.shape.length >= 3)){
		throw({message: 'first input must be a tensor of rank 3 or greater'})
	}
	if(!(Number.isInteger(filterSize) && filterSize>0)){
		throw({message: 'second input must be a positive integer'})
	}
	if(!(Number.isInteger(stride) && stride>0)){
		throw({message: 'third input must be a positive integer'})
	}
	if(!(padding === 'same' || padding === 'valid')){
		throw({message: 'fourth input must be "same" or "valid"'})
	}
	const vec = int => Array(x.shape.length).fill(int)
	const dtype = x.dtype
	const shape = getConvOutShape(x, x.shape.slice(-1),
		vec(filterSize), vec(stride), padding)
	const out = new tensor_description(shape, dtype, node.name+':0',
		opName,
		[x.val_ref],
		{filterSize, stride, padding, shape: shape.shape})
	const results = {[out.val_ref]: out}
	Object.assign(tensor_trace, results)
	return results
}

export const __max_pool__desc_func = poolGenericDescFunc('max_pool')
export const __avg_pool__desc_func = poolGenericDescFunc('avg_pool')
