import {get_init_subgraphs} from '../util/graph.js'

const stringify = JSON.stringify

function convert_ref(ref){
    const idx = ref.lastIndexOf(':')
    const node = ref.slice(0,idx)
	return `graph['${node}'][${ref.slice(idx+1)}]`
}

function convert_shape(shape){
    const arr = shape.map(c => isNaN(c)? 'None' : ''+c)
	return '['+arr.join(', ')+']'
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
			({dtype:dtype, shape: convert_shape(shape)}),
		{output_names} = unwrapped
	return unwrapped.output
		.reduce((acc,k,i) =>
			Object.assign(acc,
				{[output_names[i]]: simple_tdesc(tensor_trace[k])}),
		{})
}

function op_conversion_get_tensor(node){
	const {shape,fill,dtype} = node.attr
	const s_shape = convert_shape(shape.shape)
	const s_dtype = stringify(dtype)
	let out = ''
	if(fill.type == 'scalar') out = `tf.fill(${s_shape},${fill.val},${s_dtype})`
	else if(fill.type == 'symbol'){
		out = ({
			'ones': 	`tf.ones(${s_shape},${s_dtype})`,
			'zeros': 	`tf.zeros(${s_shape},${s_dtype})`,
			'normal': 	`tf.random_normal(${s_shape},0,1,${s_dtype})`,
			'truncated_normal': `tf.truncated_normal(${s_shape},0,1,${s_dtype})`
		})[fill.symbol]
		if(out===undefined) throw('Unsupported fill symbol')
	}
	else throw('Unsupported fill')
	return `[${out}]`
}

function convolutionWrapper(node){
	const [x, filter] = node.input
	const {stride, padding, shape} = node.attr
	const ND = shape.length - 2
	const availConvs = new Set([1, 2, 3])
	if(!availConvs.has(ND)){
		throw(`${ND}D convolution not yet supported, ` +
			`only (${[...availConvs]})D supported`)
	}
	let result = ''
	if(ND === 1){
		result = `tf.nn.conv1d(${x},${filter},${stride[0]},${stringify(padding)})`
	}else if(ND === 2){
		result = `tf.nn.conv2d(${x},${filter},` +
			`${stringify(stride)},${stringify(padding)})`
	}else if(ND === 3){
		result = `tf.nn.conv3d(${x},${filter},` +
			`${stringify(stride)},${stringify(padding)})`
	}
	return `[${result}]`
}

function batchNormConversion(node){
    return `[tf.nn.batch_normalization(${node.input.slice(0,3)},`
        + `${node.input.slice(3,5).reverse()}, 1e-4)]`
}

function gatherRowsConversion(node){
    const [x, inds] = node.input
    const range = `tf.cast(tf.range(0,${inds}.shape[0]), ${inds}.dtype)`
	const positions = `tf.stack([${range},${inds}],1)`
	return `[tf.gather_nd(${x}, ${positions})]`
}

function poolingConversion(op, node){
    // `op` is 'max_pool' or 'avg_pool'
	const x = node.input[0]
    const {filterSize, stride, padding, shape} = node.attr
	if(!(shape.length == 4 || shape.length == 5)){
		throw('Pooling only supported for inputs of rank 4 or 5.')
    }
    const tfOp = shape.length == 5? `${op}3d` : op
	return `[tf.${tfOp}(${x},${filterSize},${stride},${stringify(padding)})]`
}

const unreffedOpConversionMap = {
	get_tensor: op_conversion_get_tensor,
	placeholder: () => {throw('placeholder shouldn\'t have been called...')},
	variable: node => `[self.variables[${stringify(node.name)}]]`,
	relu: node => `[${node.input.map(r => `tf.nn.relu(${r})`)}]`,
	sigmoid: node => `[${node.input.map(r => `tf.nn.sigmoid(${r})`)}]`,
	tanh: node => `[${node.input.map(r => `tf.nn.tanh(${r})`)}]`,
	exp: node => `[tf.exp(${node.input[0]})]`,
	matmul: node => `[tf.linalg.matmul(${node.input})]`,
	add: node => `[${node.input.join(' + ')}]`,
	multiply: node => `[${node.input.join(' * ')}]`,
	divide: node => `[tf.div(${node.input})]`,
	subtract: node => `[tf.subtract(${node.input})]`,
	pow: node => `[tf.pow(${node.input})]`,
	sqrt: node => `[tf.sqrt(${node.input[0]})]`,
	softmax: node => `[tf.nn.softmax(${node.input[0]})]`,
	log: node => `[tf.math.log(${node.input[0]}+1e-8)]`,
    reduce_sum: n => `[tf.reduce_sum(${n.input[0]}, `+
        `axis=${stringify(n.attr.axis)})]`,
    reduce_avg: node => `[tf.reduce_mean(${node.input[0]}, `+
        `axis=${stringify(node.attr.axis)})]`,
	negate: node => `[-1 * ${node.input[0]}]`,
	transpose: n => `[tf.transpose(${n.input[0]}, ` +
		`perm=${stringify(n.attr.perm)})]`,
	one_hot: node => `[tf.one_hot(${node.input[0]}, ${node.attr.n_colls})]`,
	cast: n => `[tf.cast(${n.input[0]}, ${stringify(n.attr.dtype)})]`,
	abs: node => `[tf.abs(${node.input[0]})]`,
	convolution: convolutionWrapper,
	gather: n => `[tf.gather(${n.input.slice(0,2)},axis=${n.attr.axis})]`,
	reshape: n => `[tf.reshape(${n.input[0]},[${n.attr.shapeEncoding
		.map(x => typeof(x)!=typeof('')? x : n.input[0]+'.shape['+x+']')}])]`,
	batch_norm: batchNormConversion,
	gather_rows: gatherRowsConversion,
	max_pool: n => poolingConversion('max_pool', n),
	avg_pool: n => poolingConversion('avg_pool', n),
}

export const opConversionMap = Object.entries(unreffedOpConversionMap)
    .reduce((acc, [k,fn]) => Object.assign(acc, {[k]: 
        node => fn(Object.assign({}, node,
            {input: node.input.map(convert_ref)}))
    }), {})

function make_init_fn(nodes, subgraphs){
    const {init_deps, init_nodes} = subgraphs
	const varConversion = n => `[tf.Variable(${convert_ref(n.input[0])})]`
    const overriddenOps = Object.assign({}, opConversionMap, 
        {variable: varConversion})
    const preamble = ['self.tf = tf', 'graph = {}']
	const main = nodes
        .filter(n => n.op !== 'placeholder')
        .filter(n => init_deps.has(n.name))
        .map(n => `graph['${n.name}'] = ${overriddenOps[n.op](n)}`)
    const assign = 'self.variables = {'+
        init_nodes
            .map(s => `"${s}": graph['${s}'][0]`)
            .join(',')
        +'}'
    const body = [...preamble, ...main, assign].map(s => `\t${s}`)
    const lines = ['def __init__(self, tf):', ...body]
    return lines
}

function get_call_fn(unwrapped, nodes, inDesc, subgraphs){
    const ingest = 'ingested = self.ingest_input(inputs)'
    const inputAcquisition = Object.keys(inDesc)
        .map(k => `graph['${k}'] = [ingested["${k}"]]`)
    const preamble = ['tf = self.tf', 'graph = {}',
        ingest, ...inputAcquisition]
	const main = nodes
        .filter(n => n.op !== 'placeholder')
        .filter(n => subgraphs.forward.has(n.name))
        .map(n => `graph['${n.name}'] = ${opConversionMap[n.op](n)}`)
    const return_value_inner = unwrapped.output_names
        .map((name,i) => `"${name}":` +
            `${convert_ref(unwrapped.output[i])}`)
        .join(',')
    const return_statement = `return {${return_value_inner}}`
	const body = [...preamble, ...main, return_statement].map(s => `\t${s}`)
    const lines = ['def __call__(self, inputs):', ...body]
	return lines
}

function makePythonClass(name, init, call, ingest_input){
    const coalesce = lines => lines.map(s => `\t${s}`).join('\n')
    const classStr = `class ${name}:\n`
        + coalesce(init) + '\n'
        + coalesce(call) + '\n'
        + coalesce(ingest_input) + '\n'
    return classStr
}

function make_ingest_input(inDesc){
    const unnamedPrefix = 'INPUT_'
    const intFromUnnamed = s => +s.slice(unnamedPrefix.length)
    const allUnnamed = arr => arr.every(s => s.startsWith(unnamedPrefix)) && 
        arr.map(intFromUnnamed).every(n => Number.isInteger(n))
    const input_names = Object.keys(inDesc)
        .sort((a,b) => (allUnnamed([a,b])?
            intFromUnnamed(a)<intFromUnnamed(b) : a<b)? -1 : 1)
    const body = [
        'if isinstance(recieved, dict):',
        '\tingested = recieved',
        'else if isinstance(recieved, list) or isinstance(recieved, tuple):',
        `\tinput_names = ${stringify(input_names)}`,
        '\tingested = {k:v for k,v in zip(input_names, recieved)}',
        'else: raise ValueError("Input is not a dict, tuple, or list")',
        'return ingested'
    ]
    const lines = ['def ingest_input(self, recieved, check=True):',
        ...body.map(s => `\t${s}`)]
    return lines
}

export function unwrapped_to_factory(unwrapped){
    const {nodes, output, name} = unwrapped
    const inDesc = getInDesc(unwrapped)
    const outDesc = getOutDesc(unwrapped)
    const subgraphs = get_init_subgraphs(nodes, output, ['variable'])
    const init = make_init_fn(nodes, subgraphs)
    const call = get_call_fn(unwrapped, nodes, inDesc, subgraphs)
    const ingest_input = make_ingest_input(inDesc)
    return makePythonClass(name, init, call, ingest_input)
}
