import {topological_sort} from '../util/graph.js'
import {primitives} from '../util/operations.js'
import {constructors} from '../util/taffy_constructors.js'

export function test(){
	console.log('Topological sort test: ', test_topological_sort()? '✅' : '❌')
	console.log('matmul test: ', test_matmul()? '✅' : '❌')
	console.log('multiply test: ', test_multiply()? '✅' : '❌')
}

// test topological sort
function test_topological_sort(){
	function test_on_G(G){
		let keys = Object.keys(G),
			top_sort = topological_sort(G)
		for(let i in keys){
			let ix = top_sort.indexOf(keys[i])
			if(ix==-1){return false}
			for(let j in G[keys[i]].in){
				if(top_sort.indexOf(G[keys[i]].in[j]) >= ix){return false}
			}
		}
		return true
	}
	let tests = [
		{
			'a': {'in': []},
			'foo': {'in': []},
			'b': {'in': ['a', 'foo']},
			'c': {'in': ['a','b']},
			'd': {'in': ['c']},
			'e': {'in': ['d','c']},
			'f': {'in': ['e']},
			'g': {'in': ['f']},
			'y': {'in': []},
			'z': {'in': []},
		},]
	return tests.map(test_on_G).every(x=>x)
}



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

function test_matmul(){
	return testMatmul1() && testMatmul2()
}




function testMultiply1(){
	const node = new constructors.node('mynode', 'multiply', [], []),
		expected = {message: 'tensors not broadcastable', i: 2, dims: [6,7]}
	try{
		primitives.multiply.desc_function({}, node, test_tensors.slice(0,2))
	}
	catch(error){
		if(JSON.stringify(error) == JSON.stringify(expected)){
			return true
		}
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

function test_multiply(){
	return testMultiply1() && testMultiply2() && testMultiply3()
}
