import {stage_one} from '../puller/stage_one.js'
import {stage_two} from '../puller/stage_two.js'
import {stage_three} from '../puller/stage_three.js'
import {lib_1, lib_4} from './sample_libs.js'
import {puller} from '../index.js'
import tape from 'tape'

const expectedStage2Out = [
	{
		'shape': [
			1,
			2,
			3
		],
		'dtype': 'float32',
		'val_ref': 'a_op1/b_op1/c_op1:0',
		'op': 'add',
		'input': [
			'a_in1:0',
			'a_in2:0'
		],
		'attr': {}
	},
	{
		'shape': [
			1,
			2,
			3
		],
		'dtype': 'float32',
		'val_ref': 'a_op2:0',
		'op': 'exp',
		'input': [
			'a_op1/b_op2:0'
		],
		'attr': {}
	}
]

const expectedStage3 = {
	'nodes': [
		{
			'name': 'a_in2',
			'op': 'placeholder',
			'input': [],
			'attr': {}
		},
		{
			'name': 'a_in1',
			'op': 'placeholder',
			'input': [],
			'attr': {}
		},
		{
			'name': 'a_op1/b_op1/c_op1',
			'op': 'add',
			'input': [
				'a_in1:0',
				'a_in2:0'
			],
			'attr': {}
		},
		{
			'name': 'a_op1/b_op1/c_op2',
			'op': 'exp',
			'input': [
				'a_op1/b_op1/c_op1:0'
			],
			'attr': {}
		},
		{
			'name': 'a_op1/b_op2',
			'op': 'exp',
			'input': [
				'a_op1/b_op1/c_op2:0'
			],
			'attr': {}
		},
		{
			'name': 'a_op2',
			'op': 'exp',
			'input': [
				'a_op1/b_op2:0'
			],
			'attr': {}
		}
	],
	'output': [
		'a_op1/b_op1/c_op1:0',
		'a_op2:0'
	],
	'output_names': [
		'a_op1:0',
		'a_op2:0'
	],
	'name': 'module_a'
}

// tests for stage 1
function test_stage_one(){
	const sample_lib_1 = lib_1(),
		a_nodes = stage_one(sample_lib_1)
			.modules.module_a.nodes
			.map(JSON.stringify)
	const expected = {
		'name':'a_op1/b_op1',
		'input':['a_op1/b_op1/c_op1:0', 'a_op1/b_op1/c_op2:0'],
		'op':'identity',
		'literal':[]
	}
	return a_nodes.includes(JSON.stringify(expected))
}

tape('Taffy Puller, stage one', t => {
	t.ok(test_stage_one())
	t.end()
})

// tests for stage 2
function test_stage_two(){
	const stage_one_out = stage_one(lib_1()),
		inputs = {
			'a_in1': {shape: [1,2,3], dtype: 'float32'},
			'a_in2': {shape: [1,2,3], dtype: 'float32'},
		},
		stage_two_out = stage_two(stage_one_out, 'module_a', inputs),
		{tensor_trace, output} = stage_two_out
	if(!output.every(k => tensor_trace.hasOwnProperty(k))) return false
	return JSON.stringify(expectedStage2Out) ==
		JSON.stringify(output.map(k => tensor_trace[k]))
}

tape('Taffy Puller, stage two', t => {
	t.ok(test_stage_two())
	t.end()
})

// tests for stage 3
function test_stage_three(){
	const stage_one_out = stage_one(lib_1()),
		inputs = {
			'a_in1': {shape: [1,2,3], dtype: 'float32'},
			'a_in2': {shape: [1,2,3], dtype: 'float32'},
		},
		stage_two_out = stage_two(stage_one_out, 'module_a', inputs),
		stage_three_out = stage_three(stage_two_out)
	delete stage_three_out.stage_two
	return JSON.stringify(expectedStage3) == JSON.stringify(stage_three_out)
}


tape('Taffy Puller, stage three', t => {
	t.ok(test_stage_three())
	t.end()
})


const lib4ExpectedStageThree = {
	'nodes': [
		{
			'name': 'in2',
			'op': 'placeholder',
			'input': [],
			'attr': {}
		},
		{
			'name': 'in1',
			'op': 'placeholder',
			'input': [],
			'attr': {}
		},
		{
			'name': 'added',
			'op': 'add',
			'input': [
				'in1:0',
				'in2:0'
			],
			'attr': {}
		}
	],
	'output': [
		'added:0'
	],
	'output_names': [
		'out:0'
	],
	'name': 'only_module'
}

tape('Taffy Puller, lib 4 with pruning', t => {
	const lib = lib_4()
	const input_desc = {
			'in1': {'shape':[], 'dtype':'float32'},
			'in2': {'shape':[], 'dtype':'float32'}
		},
		three = puller(lib, 'only_module', input_desc)
	delete three['stage_two']
	t.deepEqual(three, lib4ExpectedStageThree)
	t.end()
})

tape('Taffy Puller, lib 4 without pruning', t => {
	const lib = lib_4()
	const input_desc = {
		'in1': {'shape':[], 'dtype':'float32'},
		'in2': {'shape':[], 'dtype':'float32'}
	}
	let three = puller(lib, 'only_module', input_desc, false)
	delete three['stage_two']
	const subtractedNode = {'name': 'subtracted',
		'op': 'subtract', 'input': ['in1:0', 'in2:0'], 'attr': {}}
	let expected = Object.assign({}, lib4ExpectedStageThree)
	expected.nodes = lib4ExpectedStageThree.nodes.slice()
	expected.nodes.splice(2, 0, subtractedNode)
	t.deepEqual(three, expected)
	t.end()
})
