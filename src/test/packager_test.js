import {unwrapped_to_constructor as tfjs_constructor} from '../packager/tfjs.js'
import {puller} from '../index.js'
import {lib_1, lib_2, lib_3} from './sample_libs.js'
import {tf} from '../deps/tf.js'
import tape from 'tape'

function deterministic_populate_variables(fn){
	fn.variables = Object.keys(fn.variables).sort().reduce((acc,k,i) => {
		const shape = fn.variables[k].shape,
			val = fn.tf.randomNormal(shape, 0, 1e-4, undefined, i+42)
		return Object.assign(acc, {[k]: val})
	}, {})
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
			.map(t=>t.dataSync().toString()),
		expected = ['1,1,1,1,1,1',
			'3814271.5,3814271.5,3814271.5,3814271.5,3814271.5,3814271.5']
	t.deepEqual(outputVals, expected)
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
			.map(t=>t.dataSync().toString()),
		expected = ['0.27040621638298035,1.2105419635772705,0,2.22743129730'+
		'2246,0.4541168808937073,0,0.4208914637565613,0,0.8518303632736206,0'+
		',0.27040621638298035,1.2105419635772705,0,2.227431297302246,0.45411'+
		'68808937073,0,0.4208914637565613,0,0.8518303632736206,0']
	t.deepEqual(outputVals, expected)
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
			.map(t=>t.dataSync().toString())
	const expected = [
		'-0.00014086009468883276,-0.00006682475213892758,' +
		'0.000011522284694365226,-0.00008573888771934435,-0.00002222436341980' +
		'938,-0.00004664461448555812,-0.00009361813135910779,0.00019449916726' +
		'443917,-0.000005609108484350145,0.00015348581655416638,-0.0001408600' +
		'9468883276,-0.00006682475213892758,0.000011522284694365226,-0.000085' +
		'73888771934435,-0.00002222436341980938,-0.00004664461448555812,-0.00' +
		'009361813135910779,0.00019449916726443917,-0.000005609108484350145,0' +
		'.00015348581655416638',
		'0.09998691827058792,0.09999431669712067,0.10' +
		'000215470790863,0.09999241679906845,0.09999877959489822,0.0999963358' +
		'0446243,0.09999162703752518,0.10002046823501587,0.10000043362379074,' +
		'0.10001634806394577,0.09998691827058792,0.09999431669712067,0.100002' +
		'15470790863,0.09999241679906845,0.09999877959489822,0.09999633580446' +
		'243,0.09999162703752518,0.10002046823501587,0.10000043362379074,' +
		'0.10001634806394577'
	]
	t.deepEqual(outputVals, expected)
	t.end()
})
