import {get_init_subgraphs} from '../util/graph.js'

const stringify = JSON.stringify

function convert_ref(name_map, ref){
	const idx = ref.lastIndexOf(':')
	return name_map[ref.slice(0,idx)] + `[${ref.slice(idx+1)}]`
}

function re_ref(unwrapped){
	const {nodes, output} = unwrapped,
		name_map = nodes.reduce(
			(acc,n,i) => Object.assign(acc, {[n.name]: 'n'+i}), {}),
		my_convert_ref = ref => convert_ref(name_map, ref),
		re_reffed_nodes = nodes.map(({name, op, input, attr}) => ({
			name: 	name_map[name],
			input: 	input.map(my_convert_ref),
			op: 	op, 
			attr: 	attr
		})),
		re_reffed_output = output.map(my_convert_ref)
	return {
		re_nodes: 	re_reffed_nodes,
		re_output: 	re_reffed_output,
		name_map: 	name_map,
	}
}


function convert_shape(shape){
	return shape.map(c => isNaN(c)? null : c)
}

function getInDesc(unwrapped){
	const mapInputDesc = ({dtype, shape}) =>
			({dtype: dtype, shape: convert_shape(shape)}),
		in_desc = unwrapped.stage_two.input_descriptions
	return Object.entries(in_desc)
		.reduce((acc,[k,v]) => Object.assign(acc, {[k]: mapInputDesc(v)}),{})
}

function getOutDesc(unwrapped){
	const tensor_trace = unwrapped.stage_two.tensor_trace,
		simple_tdesc = ({shape, dtype}) =>
			({dtype:dtype, shape:convert_shape(shape)}),
		{output_names} = unwrapped
	return unwrapped.output
		.reduce((acc,k,i) =>
			Object.assign(acc,
				{[output_names[i]]: simple_tdesc(tensor_trace[k])}),
		{})
}


function op_conversion_get_tensor(node){
	const {shape,fill,dtype} = node.attr,
		s_shape = stringify(convert_shape(shape.shape)),
		s_dtype = stringify(dtype)
	let out = ''
	if(fill.type == 'scalar') out = `tf.fill(${s_shape},${fill.val},${s_dtype})`
	else if(fill.type == 'symbol'){
		out = ({
			'ones': 	`tf.ones(${s_shape},${s_dtype})`,
			'zeros': 	`tf.zeros(${s_shape},${s_dtype})`,
			'normal': 	`tf.randomNormal(${s_shape},0,1,${s_dtype})`,
			'truncated_normal': `tf.truncatedNormal(${s_shape},0,1,${s_dtype})`
		})[fill.symbol]
		if(out===undefined) throw('Unsupported fill symbol')
	}
	else throw('Unsupported fill')
	return `[${out}]`
}

function op_conversion_reduce_avg(node){
	const axis = stringify(node.attr.axis),
		denom = `${axis}.map(i => ` +
			`${node.input[0]}.shape[i]).reduce((a,b)=>a*b,1)`,
		scalar = `tf.scalar(${denom})`,
		sum = `tf.sum(${node.input[0]}, ${axis})`,
		final = `[tf.div(${sum}, ${scalar})]`
	return final
}

function op_conversion_add(node){
	return'[' + node.input.slice(1).reduce((a,b) =>
		`tf.add(${a},${b})`, node.input[0]) + ']'
}

function op_conversion_mul(node){
	return'[' + node.input.slice(1).reduce((a,b) =>
		`tf.mul(${a},${b})`, node.input[0]) + ']'
}

function op_conversion_protected_pow(node){
	const [base, exp] = node.input,
		positiveBase = `${base}.mul(${base}.sign()).add(tf.scalar(1e-8))`,
		// 1 for odd, 0 for even
		expOdd = `tf.round(tf.mod(${exp}, tf.scalar(2)))`,
		// 1 for negative base, 0 otherwise
		baseNegative = `${base}.neg().step(0)`,
		// -1 for odd exp & negative base, 1 otherwise
		negTwoPlusOne = '.mul(tf.scalar(-2)).add(tf.scalar(1))',
		reSign = `${expOdd}.mul(${baseNegative})${negTwoPlusOne}`,
		result = `tf.pow(${positiveBase}, ${exp}).mul(${reSign})`
	return `[${result}]`
}

function convolutionWrapper(node){
	const [x, filter] = node.input,
		{stride, padding, shape} = node.attr
	const ND = shape.length - 2,
		availConvs = new Set([1, 2])
	if(!availConvs.has(ND)){
		throw(`${ND}D convolution not yet supported, ` +
			`only (${[...availConvs]})D supported`)
	}
	let result = ''
	if(ND === 1){
		result = `tf.conv1d(${x},${filter},${stride[0]},${stringify(padding)})`
	}else if(ND === 2){
		result = `tf.conv2d(${x},${filter},` +
			`${stringify(stride)},${stringify(padding)})`
	}
	return `[${result}]`
}

export const opConversionMap = {
	get_tensor: op_conversion_get_tensor,
	placeholder: () => {throw('placeholder shouldn\'t have been called...')},
	variable: node => '[this.variables[this.inverse_name_map[' +
		`${stringify(node.name)}]]]`,
	relu: node => `[${node.input.map(r => `tf.relu(${r})`)}]`,
	sigmoid: node => `[${node.input.map(r => `tf.sigmoid(${r})`)}]`,
	tanh: node => `[${node.input.map(r => `tf.tanh(${r})`)}]`,
	exp: node => `[tf.exp(${node.input[0]})]`,
	matmul: node => `[tf.matMul(${node.input})]`,
	add: op_conversion_add,
	multiply: op_conversion_mul,
	divide: node => `[tf.div(${node.input})]`,
	subtract: node => `[tf.sub(${node.input})]`,
	scalar: n => `[tf.scalar(${[+n.attr.num, stringify(n.attr.dtype)]})]`,
	pow: op_conversion_protected_pow,
	sqrt: node => `[tf.sqrt(${node.input[0]})]`,
	softmax: node => `[tf.softmax(${node.input[0]})]`,
	log: node => `[tf.log(${node.input[0]}.add(tf.scalar(1e-8)))]`,
	reduce_sum: n => `[tf.sum(${n.input[0]}, ${stringify(n.attr.axis)})]`,
	reduce_avg: op_conversion_reduce_avg,
	negate: node => `[tf.scalar(-1).mul(${node.input[0]})]`,
	transpose: n => `[tf.transpose(${n.input[0]}, ` +
		`${stringify(n.attr.perm)})]`,
	one_hot: node => `[tf.oneHot(${node.input[0]}, ${node.attr.n_colls})]`,
	cast: n => `[tf.cast(${n.input[0]}, ${stringify(n.attr.dtype)})]`,
	abs: node => `[tf.abs(${node.input[0]})]`,
	convolution: convolutionWrapper,
}


function get_variables(re_reffed_nodes, subgraphs){
	const {init_deps, init_nodes} = subgraphs,
		varConversion = node => `[tf.variable(${node.input[0]})]`,
		overriddenOps = Object.assign({}, opConversionMap,
			{variable: varConversion}),
		body = 'const tf = this.tf;' + 
			re_reffed_nodes
				.filter(n => n.op !== 'placeholder')
				.filter(n => init_deps.has(n.name))
				.map(n => `const ${n.name} = ${overriddenOps[n.op](n)};`)
				.join(''),
		map = '{'+init_nodes.map(s =>
			`[this.inverse_name_map[${stringify(s)}]]:${s}[0]`)
			.join(',')+'}',
		expression = `this.tf.tidy(()=>{${body}return ${map};})`
	return expression
}

const check_inputs = function(inputs){
	if(typeof(inputs)!=='object'){throw('`inputs` must be an object')}
	const input_descs = this.input_descriptions;
	Object.entries(input_descs).forEach(([k,v]) => {
		if(!inputs.hasOwnProperty(k)){
			throw(`Inputs must have value for '${k}'.`)
		}
		if(inputs[k].dtype != v.dtype){
			throw(`Incorrect dtype for ${k}. ` +
				`Expected '${v.dtype}', but got '${inputs[k].dtype}'.`)
		}
		if(inputs[k].shape.length != v.shape.length){
			throw(`Incorrect shape for ${k}. ` +
				`Expected [${v.shape}], but got [${inputs[k].shape}].`)
		}
		if(!v.shape.every((e,i)=>e===null || inputs[k].shape[i]==e)){
			throw(`Incorrect shape for ${k}. ` +
				`Expected [${v.shape}], but got [${inputs[k].shape}].`)
		}
	});
}.toString().replace(/\r|\n|\t|\s\s+/g, '')

function get_forward(unwrapped, re_reffed_nodes,
	input_descs, name_map, subgraphs){
	const input_acquisition = 'const tf = this.tf;' + 
		Object.keys(input_descs).map(k => 
			`const ${name_map[k]} = [inputs["${k}"]];`).join(''),
		body = re_reffed_nodes
			.filter(n => n.op !== 'placeholder')
			.filter(n => subgraphs.forward.has(n.name))
			.map(n => `const ${n.name} = ${opConversionMap[n.op](n)};`)
			.join(''),
		map_innards = unwrapped.output_names
			.map((name,i) => `"${name}":` +
				`${convert_ref(name_map, unwrapped.output[i])}`)
			.join(','),
		inner_tidy = `${input_acquisition}${body}return {${map_innards}};`,
		check_statement = 'if(check){this.check_inputs(inputs);}',
		return_statement = `return this.tf.tidy(()=>{${inner_tidy}})`,
		composed_fn = 'function(inputs, check=true){' +
			`${check_statement}${return_statement}}`
	return composed_fn
}

const optimize = function(loss, inputs,
	batch_size=32,
	iterations=100,
	optimizer=undefined,
	check_inputs=true){
	const tf = this.tf;
	const available_out = Object.entries(this.output_descriptions)
		.filter(([,{shape}]) => shape.length==0)
		.map(([k,])=>k);
	if(available_out.length==0){
		throw({message: 'there are no scalar outputs, thus no eligible loss'})
	}
	if(!this.output_descriptions.hasOwnProperty(loss)){
		throw({message: `"${loss}" isn't an output, outputs are ` +
			`${Object.keys(this.output_descriptions)}`})
	}
	if(!available_out.includes(loss)){
		throw({message: 'loss must be a scalar, but is of shape ' +
			stringify(this.output_descriptions[loss])})
	}
	const dataset_size = (Object.values(inputs)[0] || {shape: [0]}).shape[0];
	if(!Object.values(inputs).every(v => v.shape[0] == dataset_size)){
		throw({message: 'input columns are of different lengths'})
	}
	if(check_inputs){this.check_inputs(inputs)}
	const optimizer_obj = (optimizer || tf.train.adam(0.005));
	const loss_history = Array(iterations).fill().map(() => {
		const iteration_loss = +optimizer_obj.minimize(() => {
			/* Ideally we'd use tf.gather to better randomize batches,
			although this somehow messes with training,
			even when using the identity gather (ie when indices=[0,1,...,n-1]).
			For this reason, we're using tf.slice,
			until I can figure out how to resolve the tf.gather problem. */
			let positions = [0, dataset_size];
			if(batch_size < dataset_size){
				const startAvail = (dataset_size - batch_size),
					start = Math.floor(Math.random() * startAvail);
				positions = [start, batch_size];
			}
			const input = Object.entries(inputs).reduce(
				(acc,[k,v])=>Object.assign(acc,{[k]: tf.slice(v,...positions)}),
				{});
			const loss_val = this.forward(input,false)[loss];
			return loss_val;
		}, true).toString().slice(11);
		return iteration_loss;
	});
	return loss_history;
}.toString().replace(/\r|\n|\t|\s\s+/g, '')

const inherit_vars = function(donor, donor_path='', reciever_path=''){
	if(typeof(donor_path) != 'string'){
		throw('donor_path must be a string')
	}
	if(typeof(reciever_path) != 'string'){
		throw('reciever_path must be a string')
	}
	Object.entries(donor.variables)
		.filter(([k,]) => k.startsWith(donor_path))
		.map(([k,v]) => [reciever_path + k.slice(donor_path.length), v])
		.filter(([inheritName,]) => this.variables.hasOwnProperty(inheritName))
		.forEach(([inheritName,v]) => {this.variables[inheritName] = v;});
	return this;
}.toString().replace(/\r|\n|\t|\s\s+/g, '')


export function unwrapped_to_constructor(unwrapped){
	const {re_nodes, re_output, name_map} = re_ref(unwrapped),
		inDesc = getInDesc(unwrapped),
		outDesc = getOutDesc(unwrapped),
		subgraphs = get_init_subgraphs(re_nodes, re_output, ['variable'],
			s => s.slice(0,s.lastIndexOf('['))),
		forwardFn = get_forward(unwrapped, re_nodes, inDesc,name_map,subgraphs),
		passObj = o => `JSON.parse(${stringify(stringify(o))})`
	const fn = '(function(tfLib){"use_strict";' +

		'try{this.tf = tfLib || tf;}' +
		'catch(e){throw(' +
		'"A tf library must be supplied or be available as a global")}' +

		`this.implements_module = ${stringify(unwrapped.name)};` +
		`this.inherit_vars = ${inherit_vars};` +
		`this.name_map = ${passObj(name_map)};` +

		'this.inverse_name_map = Object.entries(this.name_map)' +
		'.reduce((acc,[k,v]) => Object.assign(acc,{[v]:k}), {});' +

		`this.input_descriptions = ${passObj(inDesc)};` +
		`this.output_descriptions = ${passObj(outDesc)};` +
		`this.variables = ${get_variables(re_nodes, subgraphs)};` +
		`this.check_inputs = ${check_inputs};` +
		`this.forward = ${forwardFn};` +
		`this.optimize = ${optimize};` +
		'})'
	return eval(fn)
}
