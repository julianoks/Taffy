import {unwrapped_to_constructor as tfjs_string_constructor,
	opConversionMap} from '../packager/tfjs.js'
import {primitives} from '../ops/operations.js'
import {puller} from '../index.js'
import {lib_1, lib_2, lib_3, lib_4} from './sample_libs.js'
import {tf} from './deps/tf.js'
import tape from 'tape'
import {constructors} from '../util/taffy_constructors.js'

const tfjs_constructor = (...args) => eval(tfjs_string_constructor(...args))

const {node, module, library} = constructors

const tfRange = shape => tf.tensor(
	Array(shape.reduce((a,b)=>a*b,1)).fill().map((_,i)=>i),
	shape)

function deterministic_populate_variables(fn){
	fn.variables = Object.keys(fn.variables).sort().reduce((acc,k,i) => {
		const shape = fn.variables[k].shape,
			val = fn.tf.randomNormal(shape, 0, 1e-4, undefined, i+42)
		return Object.assign(acc, {[k]: val})
	}, {})
}

function arraysClose(arr1, arr2, epsilon=1e-4){
	const flatten = a => Array.isArray(a)? [].concat(...a.map(flatten)) : a,
		f1 = flatten(arr1),
		f2 = flatten(arr2)
	if(f1.length !== f2.length) return false
	for(let i in f1){
		if(Math.abs((f1[i]-f2[i])/f1[i]) > epsilon) return false
	}
	return true
}


tape('TFJS packager, library 1', t => {
	const lib = lib_1(),
		input_desc = {'a_in1':{'shape':[1,2,3],'dtype':'float32'},
			'a_in2':{'shape':[1,2,3],'dtype':'float32'}},
		unwrapped_lib = puller(lib, 'module_a', input_desc),
		constructor = tfjs_constructor(unwrapped_lib),
		fn = new constructor(tf)
	deterministic_populate_variables(fn)
	const input = {a_in1: tf.zeros([1,2,3]),
			a_in2: tf.ones([1,2,3])},
		output = fn.forward(input),
		outputVals = Object.values(output).sort()
			.map(t => Array.from(t.dataSync())),
		expected = [
			[1,1,1,1,1,1],
			[3814271.5,3814271.5,3814271.5,3814271.5,3814271.5,3814271.5]
		]
	t.ok(arraysClose(expected, outputVals))
	t.end()
})


tape('TFJS packager, library 2', t => {
	const lib = lib_2(),
		input_desc = {'x': {'shape': ['batch',784], 'dtype': 'float32'}},
		unwrapped_lib = puller(lib, 'layer', input_desc),
		constructor = tfjs_constructor(unwrapped_lib),
		fn = new constructor(tf),
		filling = Array(2).fill(Array(784).fill().map((_,i)=>i))
	deterministic_populate_variables(fn)
	const output = fn.forward({x: tf.tensor(filling)}),
		outputVals = Object.values(output).sort()
			.map(t => Array.from(t.dataSync())),
		expected = [
			[0.27040621638298035,1.2105419635772705,0,2.227431297302246,
				0.4541168808937073,0,0.4208914637565613,0,0.8518303632736206,
				0,0.27040621638298035,1.2105419635772705,0,2.227431297302246,
				0.4541168808937073,0,0.4208914637565613,0,0.8518303632736206,0]
		]
	t.ok(arraysClose(expected, outputVals))
	t.end()
})


tape('TFJS packager, library 3', t => {
	const lib = lib_3(),
		input_desc = {'X': {'shape': ['batch',784], 'dtype': 'float32'}},
		unwrapped_lib = puller(lib, 'probs', input_desc),
		constructor = tfjs_constructor(unwrapped_lib),
		fn = new constructor(tf),
		filling = Array(2).fill(Array(784).fill().map((_,i)=>i/784))
	deterministic_populate_variables(fn)
	const output = fn.forward({'X': tf.tensor(filling)}),
		outputVals = Object.values(output).sort()
			.map(t => Array.from(t.dataSync()))
	const expected = [
		[-0.00014086009468883276,-0.00006682475213892758,
			0.000011522284694365226,-0.00008573888771934435,
			-0.00002222436341980938,-0.00004664461448555812,
			-0.00009361813135910779,0.00019449916726443917,
			-0.000005609108484350145,0.00015348581655416638,
			-0.00014086009468883276,-0.00006682475213892758,
			0.000011522284694365226,-0.00008573888771934435,
			-0.00002222436341980938,-0.00004664461448555812,
			-0.00009361813135910779,0.00019449916726443917,
			-0.000005609108484350145,0.00015348581655416638],
		[0.09998691827058792,0.09999431669712067,
			0.10000215470790863,0.09999241679906845,
			0.09999877959489822,0.09999633580446243,
			0.09999162703752518,0.10002046823501587,
			0.10000043362379074,0.10001634806394577,
			0.09998691827058792,0.09999431669712067,
			0.10000215470790863,0.09999241679906845,
			0.09999877959489822,0.09999633580446243,
			0.09999162703752518,0.10002046823501587,
			0.10000043362379074,0.10001634806394577]
	]
	t.ok(arraysClose(expected, outputVals))
	t.end()
})


tape('TFJS packager, library 4', t => {
	const lib = lib_4(),
		input_desc = {
			'in1': {'shape':[], 'dtype':'float32'},
			'in2': {'shape':[], 'dtype':'float32'}
		},
		unwrapped_lib = puller(lib, 'only_module', input_desc),
		factory = tfjs_constructor(unwrapped_lib),
		fn = new factory(tf)
	deterministic_populate_variables(fn)
	const input = {
			in1: tf.zeros([]),
			in2: tf.ones([])
		},
		output = fn.forward(input),
		outputVals = Object.values(output).sort()
			.map(t => Array.from(t.dataSync())),
		expected = [[1]]
	t.ok(arraysClose(expected, outputVals))
	t.end()
})

tape('TFJS packager can convert all tensor operations', t => {
	const opsMissing = Object.values(primitives)
		.filter(({type}) => type=='tensor')
		.filter(({name}) => !opConversionMap.hasOwnProperty(name))
	t.ok(opsMissing.length == 0)
	t.end()
})

tape('TFJS packager, convolution 1D', t => {
	const lib = new library([new module('only_module',
		['x', 'filter'],
		['conv:0'],
		[
			new node('x', 'placeholder', []),
			new node('filter', 'placeholder', []),
			new node('literals', 'literals', [], ['1', '"same"']),
			new node('stride', 'parse_json', ['literals:0']),
			new node('padding', 'parse_json', ['literals:1']),
			new node('conv', 'convolution',
				['x:0', 'filter:0', 'stride:0', 'padding:0'])
		])])
	const inputDesc = {
			x: {shape: [1,10,12], dtype: 'float32'},
			filter: {shape: [2,12,15], dtype: 'float32'}
		},
		input = Object.entries(inputDesc).reduce((acc, [k,v]) => 
			Object.assign(acc, {[k]: tfRange(v.shape)}), {})
	const unwrapped = puller(lib, 'only_module', inputDesc),
		factory = tfjs_constructor(unwrapped),
		fn = new factory(tf)
	const output = fn.forward(input),
		outputVals = Object.values(output).sort()
			.map(t => Array.from(t.dataSync())),
		expected = tf.conv1d(input.x, input.filter, 1, 'same').dataSync()
	t.deepEqual(output['conv:0'].shape, [1, 10, 15])
	t.ok(arraysClose(expected, outputVals))
	t.end()
})

tape('TFJS packager, convolution 2D', t => {
	const lib = new library([new module('only_module',
		['x', 'filter'],
		['conv:0'],
		[
			new node('x', 'placeholder', []),
			new node('filter', 'placeholder', []),
			new node('literals', 'literals', [], ['1', '"same"']),
			new node('stride', 'parse_json', ['literals:0']),
			new node('padding', 'parse_json', ['literals:1']),
			new node('conv', 'convolution',
				['x:0', 'filter:0', 'stride:0', 'padding:0'])
		])])
	const inputDesc = {
			x: {shape: [1,10,11,12], dtype: 'float32'},
			filter: {shape: [2,2,12,15], dtype: 'float32'}
		},
		input = Object.entries(inputDesc).reduce((acc, [k,v]) => 
			Object.assign(acc, {[k]: tfRange(v.shape)}), {})
	const unwrapped = puller(lib, 'only_module', inputDesc),
		factory = tfjs_constructor(unwrapped),
		fn = new factory(tf)
	const output = fn.forward(input),
		outputVals = Object.values(output).sort()
			.map(t => Array.from(t.dataSync())),
		expected = tf.conv2d(input.x, input.filter, 1, 'same').dataSync()
	t.deepEqual(output['conv:0'].shape, [1, 10, 11, 15])
	t.ok(arraysClose(expected, outputVals))
	t.end()
})

