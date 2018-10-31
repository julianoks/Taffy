import {pull_and_package} from '../../index.js'
import {lib_3} from '../sample_libs.js'
import {get_train_data} from './mnist_download.js'

function stringify_output(out){
	return Object.entries(out)
		.map(([k,v]) => k+' - '+v.toString().slice(11))
		.join('\n')+'\n\n'
}

export function run(tf,
	batch_size=32,
	iterations=1000,
	example_size=10,
	train_size=1000,
	val_size=100){
	const lossInpDesc = {'X': {'shape': ['batch',784], 'dtype': 'float32'},
			'indices': {'shape': ['batch'], 'dtype': 'int32'}}, 
		loss_fn_factory = eval(pull_and_package('tfjs', lib_3(), 
			'total_loss', lossInpDesc)),
		loss_fn = new loss_fn_factory(tf)
	console.log('loss_fn:', loss_fn)

	get_train_data(tf, example_size).then(example_data => {
		console.log('example data:', example_data)
		console.log('example output of loss_fn',
			stringify_output(loss_fn.forward(example_data)))
		get_train_data(tf, train_size).then(training_data => {
			const loss_history = loss_fn.optimize(
				training_data, 'total_loss:0', batch_size, iterations)
			console.log('loss history:', loss_history)
			let inheritor = new loss_fn_factory(tf),
				probsConstructor = eval(pull_and_package(
					'tfjs', lib_3(), 'probs', {X: lossInpDesc['X']})),
				probs_inherit = new probsConstructor(tf)
			probs_inherit.inherit_vars(loss_fn, 'probs/', 'probs/')
			console.log('example\'s indices:', example_data.indices.toString())
			console.log('output of probs module (w/inherited values):',
				stringify_output(probs_inherit.forward(example_data)))
			console.log('example output of inheritor before inheritence',
				stringify_output(inheritor.forward(example_data)))
			inheritor.inherit_vars(loss_fn)
			console.log('example output of inheritor after inheritence',
				stringify_output(inheritor.forward(example_data)))
			get_train_data(tf, val_size).then(validation_data => {
				console.log('example output of loss_fn after training',
					stringify_output(loss_fn.forward(example_data)))
				console.log('training data output of loss_fn after training',
					stringify_output(loss_fn.forward(training_data)))
				console.log('validation output of loss_fn after training',
					stringify_output(loss_fn.forward(validation_data)))
			})
		})
	})
}