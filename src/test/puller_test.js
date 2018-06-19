import {stage_one} from '../puller/stage_one.js'
import {stage_two} from '../puller/stage_two.js'
import {stage_three} from '../puller/stage_three.js'
import {lib_1} from './sample_libs.js'
import tape from 'tape'

export function test(){
	console.log('Stage one test: ', test_stage_one()?  '✅' : '❌')
	console.log('Stage two test: ', test_stage_two()?  '✅' : '❌')
	console.log('Stage three test: ', test_stage_three()?  '✅' : '❌')
}

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
	const expected = {'name':'a_op1/b_op1',
		'input':['a_op1/b_op1/c_op1:0', 'a_op1/b_op1/c_op2:0'],
		'op':'identity', 'literal':[]}
	return a_nodes.includes(JSON.stringify(expected))
}

tape('Taffy Puller, stage one', t => {
	t.equal(test_stage_one(), true)
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
	t.equal(test_stage_two(), true)
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
	t.equal(test_stage_three(), true)
	t.end()
})

