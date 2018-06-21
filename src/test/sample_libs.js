import {constructors} from '../util/taffy_constructors.js'

const {node, module, library} = constructors

export function lib_1(){
	// module a
	const nodes_a = [
			new node('a_in1', 'placeholder', []),
			new node('a_in2', 'placeholder', []),
			new node('a_op1', 'module_b', ['a_in1:0', 'a_in2:0']),
			new node('a_op2', 'exp', ['a_op1:1']),
		],
		module_a = new module('module_a', ['a_in1', 'a_in2'],
			['a_op1:0', 'a_op2:0'], nodes_a, ['module_b']),
		// module b
		nodes_b = [
			new node('b_in1', 'placeholder', []),
			new node('b_in2', 'placeholder', []),
			new node('b_op1', 'module_c', ['b_in1:0', 'b_in2:0']),
			new node('b_op2', 'exp', ['b_op1:1']),
		],
		module_b = new module('module_b', ['b_in1', 'b_in2'],
			['b_op1:0', 'b_op2:0'], nodes_b, ['module_c']),
		// module c
		nodes_c = [
			new node('c_in1', 'placeholder', []),
			new node('c_in2', 'placeholder', []),
			new node('c_op1', 'add', ['c_in1:0', 'c_in2:0']),
			new node('c_op2', 'exp', ['c_op1:0']),
		],
		module_c = new module('module_c', ['c_in1', 'c_in2'],
			['c_op1:0', 'c_op2:0'], nodes_c)
	// defining library
	return new library([module_a, module_b, module_c])
}


export function lib_2(){
	const nodes2 = [
			new node('text', 'literals', [],
				['[0.001, [784,10], "truncated_normal", "float32"]',
					'[[10], "zeros", "float32"]']),

			new node('list_w', 'parse_json_list', ['text:0']),
			new node('scale', 'scalar', ['list_w:0']),
			new node('rand', 'get_tensor', ['list_w:1', 'list_w:2','list_w:3']),
			new node('w_val', 'multiply', ['scale:0', 'rand:0']),
			new node('w', 'variable', ['w_val:0']),
		
			new node('list_b', 'parse_json_list', ['text:1']),
			new node('b_val', 'get_tensor',
				['list_b:0', 'list_b:1','list_b:2']),
			new node('b', 'variable', ['b_val:0']),

			new node('x', 'placeholder', []),
			new node('xw', 'matmul', ['x:0', 'w:0']),
			new node('xw_b', 'add', ['xw:0', 'b:0']),
			new node('relu_xw_b', 'relu', ['xw_b:0']),
		],
		module2 = new module('layer', ['x'], ['relu_xw_b:0'], nodes2),
		library2 = new library([module2])
	return library2
}


export function lib_3(){
	const he_init_nodes = [
			new node('fan_in', 'placeholder', []),
			new node('fan_out', 'placeholder', []),
			new node('literals_string', 'literals', [],
				['2', '"truncated_normal"']),
			new node('literals', 'parse_json',
				['literals_string:0', 'literals_string:1']),
			new node('shape', 'pack_list', ['fan_in:0', 'fan_out:0']),
			new node('scalar_2', 'scalar', ['literals:0']),
			new node('scalar_fan_in', 'scalar', ['fan_in:0']),
			new node('quotient', 'divide', ['scalar_2:0', 'scalar_fan_in:0']),
			new node('sqrt', 'sqrt', ['quotient:0']),
			new node('normal', 'get_tensor', ['shape:0', 'literals:1']),
			new node('out', 'multiply', ['normal:0', 'sqrt:0']),
		],
		he_init_module = new module('he_init', ['fan_in', 'fan_out'],
			['out:0'], he_init_nodes)

	const linear_nodes = [
			new node('X', 'placeholder', []),
			new node('n_in', 'placeholder', []),
			new node('n_out', 'placeholder', []),

			new node('W_val', 'he_init', ['n_in:0', 'n_out:0']),
			new node('W', 'variable', ['W_val:0']),

			new node('b_hyper', 'literals', [], ['zeros']),
			new node('b_shape', 'pack_list', ['n_out:0']),
			new node('b_val', 'get_tensor', ['b_shape:0', 'b_hyper:0']),
			new node('b', 'variable', ['b_val:0']),

			new node('dot', 'matmul', ['X:0', 'W:0']),
			new node('final', 'add', ['dot:0', 'b:0']),
		],
		linear_module = new module('linear', ['X', 'n_in', 'n_out'],
			['final:0', 'W:0'], linear_nodes, ['he_init'])

	const linear_relu_nodes = [
			new node('X', 'placeholder', []),
			new node('n_in', 'placeholder', []),
			new node('n_out', 'placeholder', []),

			new node('linear_no_relu', 'linear', ['X:0', 'n_in:0', 'n_out:0']),
			new node('relu', 'relu', ['linear_no_relu:0'])
		],
		linear_relu_module = new module('linear_relu', ['X', 'n_in', 'n_out'],
			['relu:0', 'linear_no_relu:1'], linear_relu_nodes, ['linear'])

	const logit_nodes = [
			new node('X', 'placeholder', []),

			new node('sizes_str', 'literals', [], ['[784, 500, 100, 50, 10]']),
			new node('sizes', 'parse_json_list', ['sizes_str:0']),

			new node('layer1', 'linear_relu', ['X:0','sizes:0','sizes:1']),
			new node('layer2', 'linear_relu',
				['layer1:0', 'sizes:1', 'sizes:2']),
			new node('layer3', 'linear_relu',
				['layer2:0', 'sizes:2', 'sizes:3']),
			new node('layer4', 'linear', ['layer3:0', 'sizes:3', 'sizes:4']),

			new node('weights', 'pack_list',
				['layer1:1', 'layer2:1', 'layer3:1', 'layer4:1']),
		],
		logit_module = new module('logit', ['X'],
			['layer4:0', 'weights:0'], logit_nodes, ['linear', 'linear_relu'])

	const probs_unexposed_nodes = [
			new node('X', 'placeholder', []),
			new node('logits', 'logit', ['X:0']),
			new node('softmax', 'softmax', ['logits:0']),
		],
		probs_unexposed_module = new module('probs_unexposed', ['X'],
			['softmax:0', 'logits:0', 'logits:1'],
			probs_unexposed_nodes, ['logit'])

	const probs_nodes = [
			new node('X', 'placeholder', []),
			new node('probs', 'probs_unexposed', ['X:0']),
			new node('logits', 'identity', ['probs:1']),
		],
		probs_module = new module('probs', ['X'],
			['probs:0', 'logits:0'], probs_nodes, ['probs_unexposed'])

	const log_loss_nodes = [
			new node('probs', 'placeholder', []),
			new node('indices', 'placeholder', []),

			new node('params_literals', 'literals', [], ['[10, 1, "float32"]']),
			new node('params', 'parse_json_list', ['params_literals:0']),

			new node('one_hot_int32', 'one_hot', ['indices:0', 'params:0']),
			new node('one_hot', 'cast', ['one_hot_int32:0', 'params:2']),
			new node('zerod_out', 'multiply', ['probs:0', 'one_hot:0']),
			new node('correct_probs', 'reduce_sum', ['zerod_out:0','params:1']),
			new node('log_probs', 'log', ['correct_probs:0']),
			new node('sum_probs', 'reduce_avg', ['log_probs:0']),
			new node('data_loss', 'negate', ['sum_probs:0']),
		],
		log_loss_module = new module('log_loss', ['probs', 'indices'],
			['data_loss:0'], log_loss_nodes, [])

	const total_loss_nodes = [
			new node('X', 'placeholder', []),
			new node('indices', 'placeholder', []),

			new node('probs', 'probs_unexposed', ['X:0']),
			new node('data_loss', 'log_loss', ['probs:0', 'indices:0']),

			new node('literals', 'literals', [], ['[2, 0.001]']),
			new node('nums_raw', 'parse_json_list', ['literals:0']),
			new node('num_2', 'scalar', ['nums_raw:0']),
			new node('num_weight', 'scalar', ['nums_raw:1']),
			new node('weights', 'unpack_list', ['probs:2']),
			new node('square1', 'pow', ['weights:0', 'num_2:0']),
			new node('square2', 'pow', ['weights:1', 'num_2:0']),
			new node('square3', 'pow', ['weights:2', 'num_2:0']),
			new node('square4', 'pow', ['weights:3', 'num_2:0']),
			new node('sum1', 'reduce_sum', ['square1:0']),
			new node('sum2', 'reduce_sum', ['square2:0']),
			new node('sum3', 'reduce_sum', ['square3:0']),
			new node('sum4', 'reduce_sum', ['square4:0']),
			new node('reg_loss_unscaled', 'add',
				['sum1:0','sum2:0','sum3:0','sum4:0']),
			new node('reg_loss', 'multiply',
				['reg_loss_unscaled:0', 'num_weight:0']),
			new node('total_loss', 'add', ['data_loss:0', 'reg_loss:0'])
		],
		total_loss_module = new module('total_loss', ['X','indices'],
			['total_loss:0','data_loss:0', 'reg_loss:0'],
			total_loss_nodes, ['probs_unexposed','log_loss'])

	return new library([he_init_module, linear_module, linear_relu_module,
		logit_module, probs_unexposed_module, probs_module,
		log_loss_module, total_loss_module])
}

export function lib_4(){
	const nodes = [
			new node('in1', 'placeholder', []),
			new node('in2', 'placeholder', []),
			new node('added', 'add', ['in1:0', 'in2:0']),
			// 'subtracted' should be pruned
			new node('subtracted', 'subtract', ['in1:0', 'in2:0']),
			new node('out', 'identity', ['added:0'])
		],
		onlyModule = new module('only_module', ['in1', 'in2'], ['out:0'],
			nodes, [])
	return new library([onlyModule])
}
