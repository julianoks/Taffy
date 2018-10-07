import {primitives} from '../ops/operations.js'
import {constructors} from '../util/taffy_constructors.js'
import tape from 'tape'

/*
---------------------------------
-------- test operations  -------
---------------------------------
*/
let test_tensors
{
	const tensor_factory = shape => new constructors.tensor_description(
		new constructors.tensor_shape(shape),
		'float32', 'noname:0', 'N/A', [], {})
	let tensor0 = tensor_factory([1,5,6,7]),
		tensor1 = tensor_factory([1,5,7,8]),
		tensor2 = tensor_factory([1,5,1,8]),
		tensor3 = tensor_factory(['batch_size',5,666,8]),
		tensor4 = tensor_factory([1,5,8,777])
	test_tensors = [tensor0, tensor1, tensor2, tensor3, tensor4]
}

function testMatmul1(){
	const node = new constructors.node('mynode', 'matmul', [], []),
		out = primitives.matmul.desc_function({}, node, 
			test_tensors.slice(0,2)),
		expected = [1, 5, 6, 8],
		got = out['mynode:0'].shape
	return JSON.stringify(expected) == JSON.stringify(got)
}

function testMatmul2(){
	let node = new constructors.node('mynode', 'matmul', [], []),
		out = primitives.matmul.desc_function({}, node,
			test_tensors.slice(3,5)),
		expected = ['batch_size', 5 ,666, 777],
		got = out['mynode:0'].shape
	return JSON.stringify(expected) == JSON.stringify(got)
}

tape('matmul operation', t => {
	t.equal(testMatmul1(), true)
	t.equal(testMatmul2(), true)
	t.end()
})




function testMultiply1(){
	const node = new constructors.node('mynode', 'multiply', [], []),
		i = 2,
		dims = [6,7],
		message = 'tensors are not broadcastable along ' +
			`dimension ${i}, with values ${dims}`,
		expected = {message, metaData: {dims, i},
			metaDataIdentifier: 'not_broadcastable'}
	try{
		primitives.multiply.desc_function({}, node, test_tensors.slice(0,2))
	}
	catch(error){
		return JSON.stringify(error) == JSON.stringify(expected)
	}
	return false
}

function testMultiply2(){
	const node = new constructors.node('mynode', 'multiply', [], []),
		out = primitives.multiply.desc_function({}, node,
			test_tensors.slice(1,3))
	return JSON.stringify(out['mynode:0'].shape) == JSON.stringify([1,5,7,8])
}

function testMultiply3(){
	const node = new constructors.node('mynode', 'multiply', [], []),
		out = primitives.multiply.desc_function({}, node,
			test_tensors.slice(2,4)),
		expected = ['batch_size',5,666,8],
		got = out['mynode:0'].shape
	return JSON.stringify(got) == JSON.stringify(expected)
}



tape('multiply operation', t => {
	t.equal(testMultiply1(), true)
	t.equal(testMultiply2(), true)
	t.equal(testMultiply3(), true)
	t.end()
})
