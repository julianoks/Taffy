(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
	typeof define === 'function' && define.amd ? define(['exports'], factory) :
	(factory((global.taffy = {})));
}(this, (function (exports) { 'use strict';

	function valid_C_identifier(str){
		return (z=>z[0]==z['input'])(str.match(/[_a-zA-Z][_a-zA-Z0-9]*/))
	}

	function list_of(arr, constructor){
		return arr.constructor === [].constructor &&
				(arr.length == 0 || arr.every(x => x.constructor === constructor))
	}

	/*
	----------------------------------------------------
	-------------- Taffy Object Constructors -----------
	----------------------------------------------------
	*/


	const constructors = {
		library: function(modules, tensors=[], doc=''){
			if(!list_of(modules, constructors.module)){
				throw('`modules` must be a list of module objects')
			}
			if(typeof tensors !== typeof {}){
				throw('`tensors` must either be an associative array or undefined')
			}

			this.modules = modules; // list of `module` objects
			this.tensors = tensors; // name/tensor pairs (as associative array)
			this.doc = doc; // free-form documentation for the collection of modules
		},

		module: function(name, input, output, nodes, module_import=[], doc=''){
			if((typeof name !== typeof '') || !valid_C_identifier(name)){
				throw('`name` must be a string that is a valid C identifier')
			}
			if(!list_of(input, ''.constructor)){
				throw('`input` must be a list of strings')
			}
			if(!list_of(output, ''.constructor)){
				throw('`output` must be a list of strings')
			}
			if(!list_of(nodes, constructors.node)){
				throw('`nodes` must be a list of node objects')
			}
			if(!list_of(module_import, ''.constructor)){
				throw('`module_import` must be a list of strings')
			}

			// a C identifier unique among primitives and modules in the library
			this.name = name;
			// ordered list of input node names (as string)
			this.input = input;
			// ordered list of output value references
			this.output = output;
			// unordered list of `node` objects
			this.nodes = nodes;
			// list of module names (as strings) to import as operations
			this.module_import = module_import;
			// module's documentation, as an `op_doc` object
			this.doc = doc;
		},

		node: function(name, op, input, literal=[]){
			if((typeof name !== typeof '') || !valid_C_identifier(name)){
				throw('`name` must be a string that is a valid C identifier')
			}
			if(typeof op !== typeof ''){
				throw('`op` must be a string')
			}
			if(!list_of(input, ''.constructor)){
				throw('`input` must be a list of strings')
			}
			const notTensor = n => n.constructor !== constructors.tensor_description;
			if((!(literal.length==0 || literal.every(notTensor)))){
				throw('`literal` must be a list of literals')
			}

			// a valid C identifier, unique among node names in the module
			this.name = name;
			// the identifier of the operation to implement
			this.op = op;
			// an ordered list of input value references
			this.input = input;
			// an ordered list of javascript literals
			this.literal = literal;
		},

		tensor_shape: function(integerVec){
			var shape = integerVec
				.map(x => Number.isSafeInteger(x)? Math.floor(x) : x);

			const entriesOK = shape.every(x => {
				return (typeof x === typeof '' && valid_C_identifier(x)) ||
					Number.isSafeInteger(x)
			});

			if((typeof shape !== typeof []) || !entriesOK){
				throw('`shape` must be a vector of integers or ' +
					'strings that are valid C identifiers')
			}

			this.shape = shape;
		},

		tensor_description: function(shape, dtype, val_ref, op, input, attr){
			if(shape.constructor !== constructors.tensor_shape){
				throw('`shape` must be a tensor_shape object')
			}
			if(typeof dtype !== typeof ''){
				throw('`dtype` must be a string')
			}
			if(typeof val_ref !== typeof ''){
				throw('`val_ref` must be a string')
			}
			if(typeof op !== typeof ''){
				throw('`op` must be a string')
			}
			if(!list_of(input, ''.constructor)){
				throw('`input` must be a list of strings')
			}
			if(typeof attr !== typeof {}){
				throw('`attr` must be an object')
			}

			this.shape = shape.shape;
			this.dtype = dtype;
			this.val_ref = val_ref;
			this.op = op;
			this.input = input;
			this.attr = attr;
		},

		op_doc: function(input, output, doc){
			if(!list_of(input, ''.constructor)){
				throw('`input` must be a list of strings')
			}
			if(!list_of(output, ''.constructor)){
				throw('`output` must be a list of strings')
			}

			this.input = input;
			this.output = output;
			this.doc = doc;
		}

	};

	const {tensor_description, tensor_shape} = constructors;

	const isTensor = obj => obj.constructor === tensor_description;

	const zip = (...arrs) => arrs[0].map((_,i) => arrs.map(a=>a[i]));

	function strideToArray(stride, filter){
		if(Array.isArray(stride)) return stride
		const nDims = filter.shape.length - 2;
		return Array(nDims).fill(stride)
	}

	function getConvOutShape(x, outChannels, filterInDims, stride, padding){
		const batchSize = x.shape.slice(0, 1);
		const middleInDims = x.shape.slice(1, -1);
		let middleOutDims = [];
		if(padding === 'same'){
			middleOutDims = zip(middleInDims, stride)
				.map(([inD, s]) => Math.ceil(inD / s));
		}
		if(padding === 'valid'){
			middleOutDims = zip(middleInDims, filterInDims, stride)
				.map(([inD, f, s]) => Math.ceil((1 + inD - f) / s));
		}
		if(Number.isInteger(+padding)){
			middleOutDims = zip(middleInDims, filterInDims, stride)
				.map(([inD, f, s]) => 1 + ((inD - f + (2 * padding)) / s));
			if(!middleOutDims.every(Number.isInteger)){
				throw({message: 'Invalid output shape (not an integer). ' +
					'Please change the stride or padding.'})
			}
		}
		const resultShape = [].concat(batchSize, middleOutDims, outChannels);
		return new tensor_shape(resultShape)
	}

	function __convolution__desc_func(tensor_trace, node, inputs){
		if(inputs.length < 2) throw({message: 'must take at least two inputs'})
		if(!inputs.slice(0,2).every(isTensor)){
			throw({message: '`x` and `filter` must be tensors'})
		}
		if(inputs[0].shape.length !== inputs[1].shape.length){
			throw({message: `x (rank ${inputs[0].shape.length}) and ` +
				`filter (rank ${inputs[1].shape.length}) must be the same rank`})
		}
		if(inputs[0].shape.length < 3){
			throw({message: '`x` and `filter` must be of rank 3 or greater'})
		}
		if(inputs[0].shape.slice(-1)[0] !== inputs[1].shape.slice(-2)[0]){
			throw({message: 'The second to last dimension of x ' +
				`(shape ${inputs[0].shape}) ` +
				'should equal the last dimension of filter ' +
				`(shape ${inputs[1].shape})`})
		}
		if(inputs[2] !== undefined){
			if(!isNaN(inputs[2])){
				if(!(Number.isInteger(inputs[2]) && inputs[2]>0)){
					const message = 'if `stride` is a number, ' +
						'it must be a positive integer';
					throw({message})
				}
			} else if(Array.isArray(inputs[2])){
				if(inputs[2].length !== (inputs[0].length - 2)){
					throw({message: 'If `stride` is an array, ' +
						'it must have 2 fewer dimensions than `x`'})
				}
				if(!inputs[2].every(n => Number.isInteger(n) && n>0)){
					throw({message: '`stride` must only contain positive integers'})
				}
			}
		}
		if(!(inputs[3] === undefined ||
				inputs[3] === 'same' ||
				inputs[3] === 'valid' ||
				(Number.isInteger(+inputs[3]) && (+inputs[3] >= 0)))){
			const message = '`padding` must either be "same", "valid", ' +
				'or a non-negative integer';
			throw({message})
		}
		const [x, filter] = inputs.slice(0,2),
			stride = strideToArray(inputs[2] || 1, filter),
			padding = inputs[3] || 'same';
		const dtype = x.dtype,
			shape = getConvOutShape(x,
				filter.shape.slice(-1), filter.shape.slice(1, -1),
				stride, padding),
			out = new tensor_description(shape, dtype, node.name+':0',
				'convolution',
				[x.val_ref, filter.val_ref],
				{stride, padding, shape: shape.shape}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}



	const poolGenericDescFunc = opName => (tensor_trace, node, inputs) => {
		if(inputs.length < 1) throw({message: 'must take at least one input'})
		let [x, filterSize, stride, padding] = inputs;
		filterSize = filterSize || 2;
		stride = stride || 1;
		padding = padding || 'valid';
		if(!(isTensor(x) && x.shape.length >= 3)){
			throw({message: 'first input must be a tensor of rank 3 or greater'})
		}
		if(!(Number.isInteger(filterSize) && filterSize>0)){
			throw({message: 'second input must be a positive integer'})
		}
		if(!(Number.isInteger(stride) && stride>0)){
			throw({message: 'third input must be a positive integer'})
		}
		if(!(padding === 'same' || padding === 'valid')){
			throw({message: 'fourth input must be "same" or "valid"'})
		}
		const vec = int => Array(x.shape.length).fill(int);
		const dtype = x.dtype;
		const shape = getConvOutShape(x, x.shape.slice(-1),
			vec(filterSize), vec(stride), padding);
		const out = new tensor_description(shape, dtype, node.name+':0',
			opName,
			[x.val_ref],
			{filterSize, stride, padding, shape: shape.shape});
		const results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	};

	const __max_pool__desc_func = poolGenericDescFunc('max_pool');
	const __avg_pool__desc_func = poolGenericDescFunc('avg_pool');

	const {op_doc} = constructors;

	function executeModule(moduleName, inputs, parentArgs, prefix){
		const [tensor_trace, , , collections, modules] = parentArgs;
		const module = modules[moduleName];
		let valueTrace = module.input.map(s=>prefix+s+':0').reduce((acc,k,i) =>
			Object.assign(acc, {[k]:inputs[i]}), {});
		module.nodes.forEach(nodeOrig => {
			const node = Object.assign({}, nodeOrig, {name: prefix+nodeOrig.name});
			if(node.op === 'placeholder'){return}
			const fn = nodeIn => primitives[node.op]
				.desc_function(tensor_trace, node,nodeIn, collections, modules);
			const fnOut = fn(node.input.map(ref => valueTrace[prefix+ref]));
			Object.assign(valueTrace, fnOut);
		});
		const result = module.output.map(s=>prefix+s).reduce((acc, k) =>
			Object.assign(acc, {[k]:valueTrace[k]}), {});
		return result
	}

	function assertListOfArrays(inputs){
		if(!inputs.every(a => Array.isArray(a))){
			throw({message: 'inputs must be arrays'})
		}
	}

	const makeOpFn = (op, args) => {
		const [tensor_trace, node, , colls, modules] = args;
		if(primitives.hasOwnProperty(op)){
			let counter = 0;
			return inputs => {
				const name = `${node.name}/ITERATION_${counter++}`;
				const newNode = Object.assign({}, node, {name});
				return primitives[op].desc_function(
					tensor_trace, newNode, inputs, colls, modules)
			}
		} else if(modules.hasOwnProperty(op)){
			let counter = 0;
			return inputs => executeModule(op, inputs, args,
				`${node.name}/ITERATION_${counter++}/`)
		} else {
			throw({message: `"${op}" is not a primitive or module name`})
		}
	};

	/*
	---------------------------------
	-------------- map --------------
	---------------------------------
	*/
	function __map__desc_func(...args){
		const [, node, inputs] = args;
		const op = node.literal[0];
		const opFn = makeOpFn(op, args);
		assertListOfArrays(inputs);
		if(!inputs.every(a => a.length===inputs[0].length)){
			throw({message: 'inputs must be of same length'})
		}
		const transposed = inputs[0].map((_,i) => inputs.map(r => r[i]));
		const results = transposed.map(opFn).map(o => Object.values(o)[0]);
		return {[`${node.name}:0`]: results}
	}

	const __map__primitive = {
		name: 'map',
		type: 'control',
		desc_function: __map__desc_func,
		doc: new op_doc(['...rows'],
			['the result of applying the operation to each column'],
			'maps the specified operation across columns')
	};

	/*
	---------------------------------
	------------- reduce ------------
	---------------------------------
	*/
	function __reduce__desc_func(...args){
		const [, node, inputs] = args;
		const op = node.literal[0];
		const opFn = makeOpFn(op, args);
		if(inputs[0].length==0){return null}
		if(!Array.isArray(inputs[0])){
			throw({message: 'First input must be an array'})
		}
		const getFirst = o => Object.values(o)[0];
		const reducer = (a,b) => getFirst(opFn([a,b]));
		const results = inputs.length==1? inputs[0].reduce(reducer) :
			inputs[0].reduce(reducer, inputs[1]);
		return {[`${node.name}:0`]: results}
	}

	const __reduce__primitive = {
		name: 'reduce',
		type: 'control',
		desc_function: __reduce__desc_func,
		doc: new op_doc(['array of values', '(optional) initial accumulator'],
			['The resulting accumulator'],
			'Executes the operation along the array, ie f(x3,f(x1,x2)). '+
	        'The operation takes the accumulator as the first argument.')
	};

	/*
	---------------------------------
	----------- reductions ----------
	---------------------------------
	*/
	function __reductions__desc_func(...args){
		const [, node, inputs] = args;
		const op = node.literal[0];
		const opFn = makeOpFn(op, args);
		if(inputs[0].length==0){return {[`${node.name}:0`]:null}}
		if(!Array.isArray(inputs[0])){
			throw({message: 'First input must be an array'})
		}
		const getFirst = o => Object.values(o)[0];
		const reducer = (a,b) => getFirst(opFn([a,b]));
		const fullReductions = (arr, init) => arr.reduce((acc,v) => 
			acc.concat([reducer(acc.slice(-1)[0], v)]), [init]);
		const results = inputs.length>1? fullReductions(...inputs) :
			fullReductions(inputs[0].slice(1), inputs[0][0]);
		return {[`${node.name}:0`]: results}
	}

	const __reductions__primitive = {
		name: 'reductions',
		type: 'control',
		desc_function: __reductions__desc_func,
		doc: new op_doc(['array of values', '(optional) initial accumulator'],
			['An array of the history of the accumulator'],
			'Executes the operation along the array, ie x1, f(x1,x2), '+
	        'f(x3,f(x1,x2)),... .'+
	        'The operation takes the accumulator as the first argument.')
	};

	/*
	---------------------------------
	------------- apply -------------
	---------------------------------
	*/
	function __apply__desc_func(...args){
		const [, node, inputs] = args;
		const op = node.literal[0];
		const opFn = makeOpFn(op, args);
		if(inputs[0].length==0){return {[`${node.name}:0`]:null}}
		if(!Array.isArray(inputs[0])){
			throw({message: 'First input must be an array'})
		}
		const results = opFn(inputs[0]);
		return {[`${node.name}:0`]: Object.values(results)[0]}
	}

	const __apply__primitive = {
		name: 'apply',
		type: 'control',
		desc_function: __apply__desc_func,
		doc: new op_doc(['array of values'],
			['The result of applying the operation to the values'],
			'Executes the operation on the values in the array. ')
	};

	const higherOrderPrimitives = [__map__primitive,
		__reduce__primitive, __apply__primitive, __reductions__primitive];

	const {op_doc: op_doc$1, tensor_description: tensor_description$1, tensor_shape: tensor_shape$1} = constructors;

	/*
	---------------------------------
	---------- helper fns -----------
	---------------------------------
	*/
	const isTensor$1 = v => {
		try {return v.constructor === constructors.tensor_description}
		catch(e){return false}
	};

	const ensureAllTensors = tensors => tensors.forEach((t,i) => {
		if(!isTensor$1(t)){
			let got = '';
			try {
				got = `type "${t.constructor.name}"`;
			} catch(e){ got = '"unknown"'; }
			const message = `argument ${i} is not a tensor, instead got ${got}`;
			throw({message, i, arg: t})
		}
	});

	// TODO: broadcast to most general dtype
	function broadcastDTypes(tensors){
		return tensors[0].dtype
	}

	function broadcastShapes(tensors){
		const rank = Math.max(...tensors.map(t=>t.shape.length)),
			shapes = tensors
				.map(t => Array(rank-t.shape.length).fill(1).concat(t.shape)),
			res_shape = Array(rank).fill().map((_, i) => {
				const dims = shapes.map(s => s[i]),
					symbols = [...new Set(dims.filter(isNaN))],
					numbersNotOne = [...new Set(dims.filter(x=>!isNaN(x) && x!=1))];
				if(numbersNotOne.length > 1){
					const message = 'tensors are not broadcastable along ' +
						`dimension ${i}, with values ${dims}`;
					throw({message, metaData: {dims, i},
						metaDataIdentifier: 'not_broadcastable'})	
				}
				if(symbols.length > 1){
					const message = 'symbolic dimensions are broadcastable, '+
						`along dimension ${i}, with values ${dims}`;
					throw({message, metaData: {dims, i},
						metaDataIdentifier: 'not_broadcastable'})	
				}
				if(symbols.length == 1){
					return numbersNotOne.length == 0? symbols[0] : numbersNotOne[0]
				}
				return numbersNotOne.length == 0? 1 : numbersNotOne[0]
			}),
			res_dtype = broadcastDTypes(tensors);
		if(!tensors.every(t => t.dtype == res_dtype)){
			throw({message: 'tensors are of different dtypes'})
		}
		return {shape: new tensor_shape$1(res_shape), dtype: res_dtype}
	}


	/*
	---------------------------------
	---------- placeholder  ---------
	---------------------------------
	*/
	const __placeholder__primitive = {
		name: 'placeholder',
		type: 'placeholder',
		desc_function: function(){
			throw({message: 'The placeholder desc_function shouldn\'t be called!'})
		},
		doc: new op_doc$1([],
			['Any value supplied to the placeholder'],
			'Forwards a single supplied value. Takes no inputs.')
	};


	/*
	---------------------------------
	------------- relu  -------------
	---------------------------------
	*/
	function __relu__desc_func(tensor_trace, node, tensors){
		if(tensors.length<1) throw({message: 'must take >=1 tensors'})
		ensureAllTensors(tensors);
		const results = tensors.reduce((acc, tensor, i) => {
			const shape = new tensor_shape$1(tensor.shape),
				dtype = tensor.dtype,
				val_ref = node.name + ':' + i,
				out = new tensor_description$1(shape, dtype, val_ref, 'relu',
					[tensor.val_ref], {});
			return Object.assign(acc, {[val_ref]: out})
		}, {});
		Object.assign(tensor_trace, results);
		return results
	}

	const __relu__primitive = {
		name: 'relu',
		type: 'tensor',
		desc_function: __relu__desc_func,
		doc: new op_doc$1(['tensor'], ['ReLU, ie f(x)=max(0,x)'],
			'ReLU activation function')
	};


	/*
	---------------------------------
	------------ sigmoid  -----------
	---------------------------------
	*/
	function __sigmoid__desc_func(tensor_trace, node, tensors){
		if(tensors.length<1) throw({message: 'must take >=1 tensors'})
		ensureAllTensors(tensors);
		const results = tensors.reduce((acc, tensor, i) => {
			const shape = new tensor_shape$1(tensor.shape),
				dtype = tensor.dtype,
				val_ref = node.name + ':' + i,
				out = new tensor_description$1(shape, dtype, val_ref, 'sigmoid',
					[tensor.val_ref], {});
			return Object.assign(acc, {[val_ref]: out})
		}, {});
		Object.assign(tensor_trace, results);
		return results
	}

	const __sigmoid__primitive = {
		name: 'sigmoid',
		type: 'tensor',
		desc_function: __sigmoid__desc_func,
		doc: new op_doc$1(['tensor'], ['sigmoid of input'],
			'sigmoid activation function')
	};


	/*
	---------------------------------
	------------- tanh  -------------
	---------------------------------
	*/
	function __tanh__desc_func(tensor_trace, node, tensors){
		if(tensors.length<1) throw({message: 'must take >=1 tensors'})
		ensureAllTensors(tensors);
		const results = tensors.reduce((acc, tensor, i) => {
			const shape = new tensor_shape$1(tensor.shape),
				dtype = tensor.dtype,
				val_ref = node.name + ':' + i,
				out = new tensor_description$1(shape, dtype, val_ref, 'tanh',
					[tensor.val_ref], {});
			return Object.assign(acc, {[val_ref]: out})
		}, {});
		Object.assign(tensor_trace, results);
		return results
	}

	const __tanh__primitive = {
		name: 'tanh',
		type: 'tensor',
		desc_function: __tanh__desc_func,
		doc: new op_doc$1(['tensor'], ['tanh of input'], 'tanh activation function')
	};


	/*
	---------------------------------
	-------------- exp  -------------
	---------------------------------
	*/
	function __exp__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		ensureAllTensors(tensors);
		const tensor = tensors[0],
			shape = new tensor_shape$1(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description$1(shape, dtype, node.name+':0', 'exp',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __exp__primitive = {
		name: 'exp',
		type: 'tensor',
		desc_function: __exp__desc_func,
		doc: new op_doc$1(['tensor'], ['elementwise exp, ie f(x)=e^x'],
			'exponential function, ie f(x)=e^x')
	};


	/*
	---------------------------------
	-------------- abs  -------------
	---------------------------------
	*/
	function __abs__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		ensureAllTensors(tensors);
		const tensor = tensors[0],
			shape = new tensor_shape$1(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description$1(shape, dtype, node.name+':0', 'abs',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __abs__primitive = {
		name: 'abs',
		type: 'tensor',
		desc_function: __abs__desc_func,
		doc: new op_doc$1(['tensor'], ['elementwise absolute value, ie f(x)=|x|'],
			'abs value function, ie f(x)=|x|')
	};


	/*
	---------------------------------
	------------- negate  -----------
	---------------------------------
	*/
	function __negate__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		ensureAllTensors(tensors);
		const tensor = tensors[0],
			shape = new tensor_shape$1(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description$1(shape, dtype, node.name+':0', 'negate',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __negate__primitive = {
		name: 'negate',
		type: 'tensor',
		desc_function: __negate__desc_func,
		doc: new op_doc$1(['tensor'], ['negation, ie f(x)=-x'],
			'negation function, ie f(x)=-x')
	};


	/*
	---------------------------------
	------------- sqrt  -------------
	---------------------------------
	*/
	function __sqrt__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		ensureAllTensors(tensors);
		const tensor = tensors[0],
			shape = new tensor_shape$1(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description$1(shape, dtype, node.name+':0', 'sqrt',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __sqrt__primitive = {
		name: 'sqrt',
		type: 'tensor',
		desc_function: __sqrt__desc_func,
		doc: new op_doc$1(['tensor'], ['elementwise square root'],
			'square root function')
	};


	/*
	---------------------------------
	------------- matmul  -----------
	---------------------------------
	*/
	function __matmul__desc_func(tensor_trace, node, tensors){
		if(tensors.length!=2) throw({message: 'must take 2 tensors'})
		ensureAllTensors(tensors);
		if(tensors[0].dtype !== tensors[1].dtype){
			throw({message: 'tensors are of different dtypes'})
		}
		if(!tensors.every(t => t.shape.length >= 2)){
			throw({message: 'tensors must be of rank >=2'})
		}
		if(tensors[0].shape.length !== tensors[1].shape.length){
			throw({message: 'tensors are of different rank'})
		}
		if(!isNaN(tensors[0].shape.slice(-1)[0]) && 
			!isNaN(tensors[1].shape.slice(-2)[0]) && 
			tensors[0].shape.slice(-1)[0] !== tensors[1].shape.slice(-2)[0]){
			throw({message: 'shapes don\'t match (dimension -1 != dimension -2)'})
		}
		const prefix = tensors[0].shape.slice(0,-2).map((d1,i) => {
				const d2 = tensors[1].shape[i];
				if(!isNaN(d1) && !isNaN(d2)){
					if(d1!=d2){
						const message = 'tensors are not broadcastable '+
							`along dimension ${i}, with values ${[d1, d2]}`;
						throw({message, metaData: {dims: [d1,d2], i},
							metaDataIdentifier: 'not_broadcastable'})	
					}
					return d1
				}
				if(isNaN(d1) && isNaN(d2)) return d1
				return isNaN(d1)? d1 : d2
			}),
			d1 = tensors[0].shape.slice(-2)[0],
			d2 = tensors[1].shape.slice(-1)[0],
			shape = new tensor_shape$1(prefix.concat([d1,d2])),
			dtype = tensors[0].dtype,
			out = new tensor_description$1(shape, dtype, node.name+':0', 'matmul',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __matmul__primitive = {
		name: 'matmul',
		type: 'tensor',
		desc_function: __matmul__desc_func,
		doc: new op_doc$1(['tensor 1', 'tensor 2'],
			['matrix multiplication of tensors'],
			'matrix multiplication of tensors')
	};


	/*
	---------------------------------
	-------------- add  -------------
	---------------------------------
	*/
	function __add__desc_func(tensor_trace, node, tensors){
		if(tensors.length==0) throw({message: 'must take n>=1 tensors'})
		ensureAllTensors(tensors);
		const {shape, dtype} = broadcastShapes(tensors),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'add',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __add__primitive = {
		name: 'add',
		type: 'tensor',
		desc_function: __add__desc_func,
		doc: new op_doc$1(['...tensor values'], ['sum of tensors'],
			'variadic function that adds n>=1 tensors')
	};


	/*
	---------------------------------
	----------- subtract  -----------
	---------------------------------
	*/
	function __subtract__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==2) throw({message: 'must take 2 tensors'})
		ensureAllTensors(tensors);
		const {shape, dtype} = broadcastShapes(tensors),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'subtract',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __subtract__primitive = {
		name: 'subtract',
		type: 'tensor',
		desc_function: __subtract__desc_func,
		doc: new op_doc$1(['tensor 1', 'tensor 2'],
			['element-wise subtraction of tensors'],
			'subtracts 2 tensors element-wise')
	};


	/*
	---------------------------------
	------------ multiply  ----------
	---------------------------------
	*/
	function __multiply__desc_func(tensor_trace, node, tensors){
		if(tensors.length==0) throw({message: 'must take n>=1 tensors'})
		ensureAllTensors(tensors);
		const {shape, dtype} = broadcastShapes(tensors),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'multiply',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __multiply__primitive = {
		name: 'multiply',
		type: 'tensor',
		desc_function: __multiply__desc_func,
		doc: new op_doc$1(['...tensor values'],
			['element-wise product of tensors'],
			'variadic function that multiplies n>=1 tensors element-wise')
	};


	/*
	---------------------------------
	------------- divide  -----------
	---------------------------------
	*/
	function __divide__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==2) throw({message: 'must take 2 tensors'})
		ensureAllTensors(tensors);
		const {shape, dtype} = broadcastShapes(tensors),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'divide',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __divide__primitive = {
		name: 'divide',
		type: 'tensor',
		desc_function: __divide__desc_func,
		doc: new op_doc$1(['tensor 1', 'tensor 2'],
			['element-wise division of tensors'],
			'divides 2 tensors element-wise')
	};


	/*
	---------------------------------
	-------------- pow  -------------
	---------------------------------
	*/
	function __pow__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==2) throw({message: 'must take 2 tensors'})
		ensureAllTensors(tensors);
		const {shape, dtype} = broadcastShapes(tensors),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'pow',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __pow__primitive = {
		name: 'pow',
		type: 'tensor',
		desc_function: __pow__desc_func,
		doc: new op_doc$1(['tensor 1', 'tensor 2'],
			['The power of tensor 1 to tensor 2'],
			'The power of tensor 1 to tensor 2')
	};


	/*
	---------------------------------
	---------- get_tensor  ----------
	---------------------------------
	*/

	function __get_tensor__desc_func(tensor_trace, node, inputs){
		let [shape, fill, dtype] = inputs;
		if(shape == undefined) throw({message: 'shape must be defined'})
		if(fill == undefined) throw({message: 'fill must be defined'})
		dtype = dtype || 'float32';
		if(isTensor$1(dtype)) { dtype = dtype.dtype; }
		if(isTensor$1(shape)){ shape = shape.shape; }
		try{shape = new tensor_shape$1(shape);}
		catch(e){
			const message = 'Provided shape is not a valid tensor shape. ' +
				'A tensor shape must be a vector of integers or ' +
				'strings that are valid C identifiers.';
			throw({message})
		}
		if(shape.shape.some(x=> typeof(x)===typeof(''))){
			throw({message: 'Shape must not contain symbolic dimensions'})
		}
		const supported_fills = new Set(['ones', 'zeros',
			'normal', 'truncated_normal']);
		if(supported_fills.has(fill)){
			fill = {type: 'symbol', symbol: fill};
		} else if(!isNaN(+fill)){
			fill = {type: 'scalar', val: +fill};
		} else{
			const message = `Fill not supported: "${fill}". ` +
				'Must either be a number (as a string), or one of the following: '+
				[...supported_fills].map(a=>`"${a}"`).join(', ');
			throw({message})
		}
		const out = new tensor_description$1(shape, dtype, node.name+':0',
				'get_tensor', [], {shape, fill, dtype}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __get_tensor__primitive = {
		name: 'get_tensor',
		type: 'tensor',
		desc_function: __get_tensor__desc_func,
		doc: new op_doc$1(['shape, a vector or tensor whose shape will be inherited',
			'fill, one of (number, "ones", "zeros", "normal", "truncated_normal")',
			'(optional) dtype, either undefined, a string, ' +
				'or a tensor whose dtype will be inherited'],
		['tensor'], 'produces a tensor')
	};

	/*
	---------------------------------
	------------- scalar  -----------
	---------------------------------
	*/

	function __scalar__desc_func(tensor_trace, node, inputs){
		let [number, dtype] = inputs;
		dtype = dtype || 'float32';
		const shape = [];
		if(isNaN(+number)){throw({message: 'First input must be a number'})}
		if(typeof(dtype) !== typeof('')){
			throw({message: 'Second input must be a string (or undefined)'})
		}
		return __get_tensor__desc_func(tensor_trace, node, [shape, number, dtype])
	}

	const __scalar__primitive = {
		name: 'scalar',
		type: 'control',
		desc_function: __scalar__desc_func,
		doc: new op_doc$1(['A number', '(optional) dtype'],
			['a scalar tensor'], 'produces a tensor from a scalar')
	};


	/*
	---------------------------------
	----------- variable  -----------
	---------------------------------
	*/
	function __variable__desc_func(tensor_trace, node, inputs, collection_bins){
		if(!(inputs.length === 1 || inputs.length === 2)){
			throw({message: 'must take one or two inputs'})
		}
		let [tensor, collections] = inputs;
		if(!isTensor$1(tensor)) throw({message: 'input #0 must be a tensor'})
		if(tensor.shape.some(x=> typeof(x)===typeof(''))){
			throw({message: 'Tensor must not contain symbolic dimensions'})
		}
		collections = collections || [];
		collections = typeof(collections)===typeof('')? [collections] : collections;
		if(!collections.every(s => typeof(s)===typeof(''))){
			throw({message: 'input #1 must be a string or list of strings'})
		}
		const name = `${node.name}:0`,
			{shape, dtype} = tensor,
			tshape = new tensor_shape$1(shape),
			out = new tensor_description$1(tshape, dtype, name, 'variable',
				[tensor.val_ref], {});
		collections.forEach(bin => {
			if(collection_bins.hasOwnProperty(bin)){
				collection_bins[bin][out.val_ref] = out;
			} else {
				collection_bins[bin] = {[out.val_ref]: out};
			}
		});
		const results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __variable__primitive = {
		name: 'variable',
		type: 'tensor',
		desc_function: __variable__desc_func,
		doc: new op_doc$1(
			['tensor', '(optional) a bin or list of bins to add the tensor to'],
			['tensor'],
			'initializes tensor to provided value')
	};


	/*
	---------------------------------
	----------- identity  -----------
	---------------------------------
	*/
	function __identity__desc_func(tensor_trace, node, inputs){
		return inputs.reduce((a,v,i) => {
			return Object.assign(a, {[node.name+':'+i]: v})
		}, {})
	}

	const __identity__primitive = {
		name: 'identity',
		type: 'control',
		desc_function: __identity__desc_func,
		doc: new op_doc$1(['...inputs'], ['...inputs'], 'forwards inputs, unchanged')
	};



	/*
	---------------------------------
	----------- literals  -----------
	---------------------------------
	*/
	function __literals__desc_func(tensor_trace, node){
		return node.literal.reduce((a,v,i) => {
			return Object.assign(a, {[node.name+':'+i]: v})
		}, {}) 
	}

	const __literals__primitive = {
		name: 'literals',
		type: 'control',
		desc_function: __literals__desc_func,
		doc: new op_doc$1([], ['...literals'], 'forwards literals, unchanged')
	};


	/*
	---------------------------------
	---------- parse_json  ----------
	---------------------------------
	*/
	function __parse_json__desc_func(tensor_trace, node, inputs){
		return inputs.reduce((a,v,i) => {
			try{return Object.assign(a, {[node.name+':'+i]: JSON.parse(v)})}
			catch(e){throw({message:`Couldn't parse JSON literal #${i}, "${v}"`})}
		}, {})
	}

	const __parse_json__primitive = {
		name: 'parse_json',
		type: 'control',
		desc_function: __parse_json__desc_func,
		doc: new op_doc$1(['...inputs (literals)'], ['...inputs'],
			'parses JSON literals')
	};


	/*
	---------------------------------
	------- parse_json_list  --------
	---------------------------------
	*/
	function __parse_json_list__desc_func(tensor_trace, node, inputs){
		if(inputs.length != 1) throw({message: 'must take exactly 1 input'})
		if(typeof(inputs[0])!=='string') throw({message: 'input must be a string'})
		let parsed = '';
		try{parsed = JSON.parse(inputs[0]);}
		catch(e){throw({message:'Couldn\'t parse JSON'})}
		if(!Array.isArray(parsed)) throw({message: 'value is not a list'})
		return parsed
			.reduce((acc,v,i) => Object.assign(acc, {[node.name+':'+i]: v}), {})
	}

	const __parse_json_list__primitive = {
		name: 'parse_json_list',
		type: 'control',
		desc_function: __parse_json_list__desc_func,
		doc: new op_doc$1(['JSON representation of a list'],
			['...parsed entries of list'],
			'parses a JSON representation of a list')
	};




	/*
	---------------------------------
	--------- if statement  ---------
	---------------------------------
	*/
	function __if__desc_func(tensor_trace, node, inputs){
		if(typeof(inputs[0]) === 'boolean'){
			throw({message: 'first argument must be boolean'})
		}
		if(inputs.length != 3) throw({message: 'must take 3 arguments'})
		return {[node.name+':0']: inputs[0]? inputs[1] : inputs[2]}
	}

	const __if__primitive = {
		name: 'if',
		type: 'control',
		desc_function: __if__desc_func,
		doc: new op_doc$1(['boolean', 'value', 'value'], ['one of the values'],
			'forwards one of the values')
	};



	/*
	---------------------------------
	----------- pack_list  ----------
	---------------------------------
	*/
	function __pack_list__desc_func(tensor_trace, node, inputs){
		return {[node.name+':0']: inputs}
	}

	const __pack_list__primitive = {
		name: 'pack_list',
		type: 'control',
		desc_function: __pack_list__desc_func,
		doc: new op_doc$1(['...values'], ['array containing the values'],
			'packs the input values into an array')
	};

	/*
	---------------------------------
	---------- unpack_list  ---------
	---------------------------------
	*/
	function __unpack_list__desc_func(tensor_trace, node, inputs){
		if(!Array.isArray(inputs[0])) throw({message: 'input is not an array'})
		return inputs[0]
			.reduce((acc,v,i)=> Object.assign(acc, {[node.name+':'+i]: v}), {})
	}

	const __unpack_list__primitive = {
		name: 'unpack_list',
		type: 'control',
		desc_function: __unpack_list__desc_func,
		doc: new op_doc$1(['array of values'], ['...values'],
			'unpacks the input values from an array')
	};



	/*
	---------------------------------
	------------ softmax  -----------
	---------------------------------
	*/
	function __softmax__desc_func(tensor_trace, node, inputs){
		if(inputs.length!==1) throw({message: 'must take 1 tensor'})
		ensureAllTensors(inputs);
		const tensor = inputs[0],
			shape = new tensor_shape$1(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description$1(shape, dtype, node.name+':0', 'softmax',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __softmax__primitive = {
		name: 'softmax',
		type: 'tensor',
		desc_function: __softmax__desc_func,
		doc: new op_doc$1(['tensor'], ['softmax of tensor'],
			'applies the softmax function to a tensor')
	};



	/*
	---------------------------------
	------------- log  --------------
	---------------------------------
	*/
	function __log__desc_func(tensor_trace, node, inputs){
		if(inputs.length!==1) throw({message: 'must take 1 tensor'})
		ensureAllTensors(inputs);
		const tensor = inputs[0],
			shape = new tensor_shape$1(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description$1(shape, dtype, node.name+':0', 'log',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __log__primitive = {
		name: 'log',
		type: 'tensor',
		desc_function: __log__desc_func,
		doc: new op_doc$1(['tensor'], ['natural log of tensor'],
			'applies the natural log function to a tensor')
	};

	/*
	---------------------------------
	---------- reduce_sum  ----------
	---------------------------------
	*/
	function __reduce_sum__desc_func(tensor_trace, node, inputs){
		if(!(inputs.length == 1 || inputs.length == 2)){
			throw({message: 'must take one or two inputs'})
		}
		if(!isTensor$1(inputs[0])){
			throw({message: 'input must be a tensor'})	
		} 
		const tensor = inputs[0],
			default_perm = tensor.shape.map((_,i)=>i),
			axis = !isNaN(inputs[1])? [inputs[1]] :
				[...(new Set(inputs[1] || default_perm))].sort();
		if(!axis.every(x=> !isNaN(x) && 0<=x && x<tensor.shape.length)){
			throw({message: `axis out of bounds: ${axis}`})
		}
		const dtype = tensor.dtype,
			raw_shape = tensor.shape
				.reduce((acc,v,i) => axis.includes(i)? acc : [...acc,v], []),
			shape = new tensor_shape$1(raw_shape),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'reduce_sum',
				[tensor.val_ref], {axis:axis}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __reduce_sum__primitive = {
		name: 'reduce_sum',
		type: 'tensor',
		desc_function: __reduce_sum__desc_func,
		doc: new op_doc$1(['tensor', 'axis; a integer or array of integers'],
			['a scalar'], 'sums a tensor')
	};

	/*
	---------------------------------
	---------- reduce_avg  ----------
	---------------------------------
	*/
	function __reduce_avg__desc_func(tensor_trace, node, inputs){
		if(!(inputs.length == 1 || inputs.length == 2)){
			throw({message: 'must take one or two inputs'})
		}
		if(!isTensor$1(inputs[0])) throw({message: 'input must be a tensor'})
		const tensor = inputs[0],
			default_perm = tensor.shape.map((_,i)=>i),
			axis = !isNaN(inputs[1])? [inputs[1]] :
				[...(new Set(inputs[1] || default_perm))].sort();
		if(!axis.every(x=> !isNaN(x) && 0<=x && x<tensor.shape.length)){
			throw({message: `axis out of bounds: ${axis}`})
		}
		const dtype = tensor.dtype,
			raw_shape = tensor.shape
				.reduce((acc,v,i) => axis.includes(i)? acc : [...acc,v], []),
			shape = new tensor_shape$1(raw_shape),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'reduce_avg',
				[tensor.val_ref], {axis:axis}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __reduce_avg__primitive = {
		name: 'reduce_avg',
		type: 'tensor',
		desc_function: __reduce_avg__desc_func,
		doc: new op_doc$1(['tensor', 'axis; a integer or array of integers'],
			['a scalar'],
			'averages a tensor')
	};

	/*
	---------------------------------
	----------- transpose  ----------
	---------------------------------
	*/
	function __transpose__desc_func(tensor_trace, node, inputs){
		if(!(inputs.length == 1 || inputs.length == 2)){
			throw({message: 'must take one or two inputs'})
		}
		if(!isTensor$1(inputs[0])) throw({message: 'first input must be a tensor'})
		const tensor = inputs[0],
			default_perm = Array(tensor.shape.length).fill()
				.map((_,i)=>i).reverse(),
			default_perm_set = new Set(default_perm),
			perm = inputs[1] || default_perm;
		if(!(default_perm.length==perm.length &&
			perm.every(v => default_perm_set.has(v)))){
			throw({message: 'permutation isn\'t a permutation of 0...n-1, ' +
				`recieved ${perm}`})
		}
		const dtype = tensor.dtype,
			shape = new tensor_shape$1(perm.map(i=>tensor.shape[i])),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'transpose',
				[tensor.val_ref], {perm:perm}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __transpose__primitive = {
		name: 'transpose',
		type: 'tensor',
		desc_function: __transpose__desc_func,
		doc: new op_doc$1(['tensor', 'permutation (optional)'],
			['tensor with permuted dimensions'],
			'permutes the dimensions of tensor according to ' +
			'the supplied permutation')
	};

	/*
	---------------------------------
	------------ one_hot  -----------
	---------------------------------
	*/
	function __one_hot__desc_func(tensor_trace, node, inputs){
		if(inputs.length != 2) throw({message: 'must take two inputs'})
		if(!(isTensor$1(inputs[0]) && inputs[0].shape.length == 1)){
			throw({message: 'first input must be a rank 1 tensor'})
		}
		if(isNaN(inputs[1]) || Math.floor(+inputs[1])<2){
			throw({message: 'second input must be a number >=2'})
		}
		const tensor = inputs[0],
			n_colls = Math.floor(+inputs[1]),
			dtype = tensor.dtype,
			shape = new tensor_shape$1([tensor.shape[0], n_colls]),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'one_hot',
				[tensor.val_ref], {n_colls:n_colls}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __one_hot__primitive = {
		name: 'one_hot',
		type: 'tensor',
		desc_function: __one_hot__desc_func,
		doc: new op_doc$1(['indices (as rank 1 tensor)', 'number of columns'],
			['matrix with one hot vectors as rows'],
			'constructs a matrix where each row is a one hot vector, ' +
			'with n_colls columns and one row for each index')
	};

	/*
	---------------------------------
	------------- cast  -------------
	---------------------------------
	*/
	function __cast__desc_func(tensor_trace, node, inputs){
		if(inputs.length != 2) throw({message: 'must take two inputs'})
		const [tensor, given_dtype] = inputs;
		if(!isTensor$1(tensor)) throw({message: 'first input must be a tensor'})
		if(!(typeof(given_dtype) == 'string' || isTensor$1(given_dtype))){
			throw({message: 'second input must be a string or a tensor'})	
		}
		const dtype = isTensor$1(given_dtype)? given_dtype.dtype : given_dtype,
			shape = new tensor_shape$1(tensor.shape),
			out = new tensor_description$1(shape, dtype, node.name+':0', 'cast',
				[tensor.val_ref], {dtype:dtype}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __cast__primitive = {
		name: 'cast',
		type: 'tensor',
		desc_function: __cast__desc_func,
		doc: new op_doc$1(['tensor', 'dtype (a string)'],
			['tensor cast as dtype'],
			'casts a tensor to a specified dtype')
	};

	/*
	---------------------------------
	------------ gather  ------------
	---------------------------------
	*/
	function __gather__desc_func(tensor_trace, node, inputs){
		if(!(inputs.length === 2 || inputs.length === 3)){
			throw({message: 'must take two or three inputs'})
		}
		const [tensor, indices] = inputs;
		const axis = inputs[2]? inputs[2] : 0;
		ensureAllTensors(inputs.slice(0,2));
		// checking tensor
		if(tensor.shape.length == 0){
			throw({message: 'first input must not be a scalar'})
		}
		// checking indices
		if(indices.dtype !== 'int32'){
			throw({message: 'second input must have dtype "int32", instead got "'+
				indices.dtype+'"'})
		}
		if(indices.shape.length !== 1){
			throw({message: 'second input must be one dimensional'})
		}
		// checking axis
		if(!(Number.isInteger(axis) && axis>=0 && axis<tensor.shape.length)){
			throw({message: 'third input must be an integer between 0-'+
				tensor.shape.length-1})
		}
		let shape = tensor.shape.slice();
		shape[axis] = indices.shape[0];
		shape = new tensor_shape$1(shape);
		const out = new tensor_description$1(shape, tensor.dtype, node.name+':0',
			'gather', [tensor.val_ref, indices.val_ref], {axis});
		const results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __gather__primitive = {
		name: 'gather',
		type: 'tensor',
		desc_function: __gather__desc_func,
		doc: new op_doc$1(['x', 'indices (1d tensor with dtype "int32")',
			'(optional) axis'],
		['tensor of slices from `x`'],
		'takes slices from `x` along `axis` at the specified `indices`')
	};

	/*
	---------------------------------
	--------- gather_rows  ----------
	---------------------------------
	*/
	function __gather_rows__desc_func(tensor_trace, node, inputs){
		if(inputs.length !== 2){
			throw({message: 'must take two inputs'})
		}
		const [x, colls] = inputs;
		if(!(isTensor$1(x) && x.shape.length>=2)){
			throw({message: 'first input must be a tensor of rank>=2'})
		}
		if(!(isTensor$1(colls) && colls.shape.length==1)){
			throw({message: 'second input must be a tensor of '+
				'rank 1 with dtype "int32"'})
		}
		if(x.shape[0] !== colls.shape[0]){
			throw({message: 'first dimensions must match, '+
				`(${x.shape[0]} != ${colls.shape[0]})`})
		}
		let shape = [x.shape[0], ...x.shape.slice(2)];
		shape = new tensor_shape$1(shape);
		const out = new tensor_description$1(shape, x.dtype, node.name+':0',
			'gather_rows', [x.val_ref, colls.val_ref], {});
		const results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __gather_rows__primitive = {
		name: 'gather_rows',
		type: 'tensor',
		desc_function: __gather_rows__desc_func,
		doc: new op_doc$1(['x', 'indices (1d tensor with dtype "int32")'],
			['tensor of slices from rows of `x` at the provided indices'],
			'takes slices from rows of `x` along at the provided `indices`')
	};

	/*
	---------------------------------
	----------- reshape  ------------
	---------------------------------
	*/
	function __reshape__desc_func(tensor_trace, node, inputs){
		if(inputs.length != 2) throw({message: 'must take two inputs'})
		let [tensor, newShape] = inputs;
		if(!isTensor$1(tensor)) throw({message: 'first input must be a tensor'})
		// checking shape
		newShape = Array.isArray(newShape)? newShape : [newShape];
		const oldSymbols = JSON.stringify(tensor.shape.filter(isNaN).sort());
		const newSymbols = JSON.stringify(newShape.filter(isNaN).sort());
		if(oldSymbols !== newSymbols){
			throw({message: 'Symbolic dimensions did not match.'})
		}
		if(!newShape.filter(x=>!isNaN(x)).every(x=>Number.isInteger(x)&&x>=0)){
			throw({message: 'Dimensions must be nonnegative integers.'})
		}
		const getSize = arr => arr.filter(x=>!isNaN(x)).reduce((a,b) => a*b, 1);
		if(getSize(newShape) !== getSize(tensor.shape)){
			throw({message: 'Sizes do not match'})
		}
		const shapeEncoding = newShape.map(x => !isNaN(x)? x :
			''+tensor.shape.indexOf(x));
		try {
			newShape = new tensor_shape$1(newShape);
		} catch(message){ throw({message}) }
		const out = new tensor_description$1(newShape, tensor.dtype, node.name+':0',
			'reshape', [tensor.val_ref], {shapeEncoding});
		const results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __reshape__primitive = {
		name: 'reshape',
		type: 'tensor',
		desc_function: __reshape__desc_func,
		doc: new op_doc$1(['x', 'shape (array of nonnegative integers)'],
			['`x` reshaped to given `shape`'],
			'reshapes `x` into given shape `shape`')
	};


	/*
	---------------------------------
	---------- js_function  ----------
	---------------------------------
	*/
	function __js_function__desc_func(tensor_trace, node, inputs){
		let fn = undefined;
		let result = undefined;
		try {
			fn = eval(node.literal[0]);
		} catch(e){
			throw({message: 'Could not evaluate function string, '+
				`got error: ${e.toString()}`})
		}
		if(typeof(fn) !== 'function'){
			throw({message: 'Function string did not evaluate to a function, '+
				`instead got type "${typeof(fn)}"`})
		}
		try {
			result = fn(...inputs);
		} catch(e){
			throw({message: `Error in applying function: ${e.toString()}`})
		}
		const resultsArray = Array.isArray(result)? result : [result];
		return resultsArray.reduce((acc, res, i) =>
			Object.assign(acc, {[node.name+':'+i]: res}), {})
	}

	const __js_function__primitive = {
		name: 'js_function',
		type: 'control',
		desc_function: __js_function__desc_func, 
		doc: new op_doc$1(['...arguments'],
			['the outputs of the function applied to the arguments'],
			'applies the function to the arguments, and returns the results')
	};

	/*
	---------------------------------
	-------- get_collection  --------
	---------------------------------
	*/
	function __get_collection__desc_func(tensor_trace, node, inputs, coll_bins){
		const collections = Array.isArray(inputs[0])? inputs[0] : [inputs[0]];
		if(!collections.every(s => typeof(s)===typeof(''))){
			throw({message: 'Input must be a string or list of strings'})
		}
		const dict = collections
			.filter(name => coll_bins.hasOwnProperty(name))
			.map(name => coll_bins[name])
			.reduce((acc, coll) => Object.assign(acc,coll), {});
		const results = {[`${node.name}:0`]: Array.from(Object.values(dict))};
		return results
	}

	const __get_collection__primitive = {
		name: 'get_collection',
		type: 'control',
		desc_function: __get_collection__desc_func,
		doc: new op_doc$1(
			['collection name, or list of names, as strings',
				'...optional control edges'],
			['list of tensors in the specified collections'],
			'finds a list of tensors in the specified collection(s)')
	};

	/*
	---------------------------------
	---------- batch_norm  ----------
	---------------------------------
	*/
	function __batch_norm__desc_func(tensor_trace, node, inputs, coll_bins){
		const tensor = inputs[0];
		if(!isTensor$1(tensor)){throw({message: 'First input must be tensor'})}
		if(tensor.shape.slice(1).some(x=> typeof(x)===typeof(''))){
			throw({message: 'Tensor must not contain symbolic dimensions'+
				' (except for first dimension)'})
		}
		const shape = [1, ...tensor.shape.slice(1)];
		const dtype = tensor.dtype;
		const newNode = ext =>  Object.assign({}, node, {name: node.name+ext});
		const bins = ['trainable', 'batchNorm'];
		const getValue = (name, fill) => {
			const nodeInit = newNode(`/${name}/init`);
			const nodeVar = newNode(`/${name}/variable`);
			const init = Object.values(__get_tensor__desc_func(
				tensor_trace, nodeInit, [shape, fill, dtype]))[0];
			return Object.values(__variable__desc_func(
				tensor_trace, nodeVar, [init, bins], coll_bins))[0]
		};
		const mean = getValue('mean', 0);
		const variance = getValue('variance', 1);
		const scale = getValue('scale', 1);
		const offset = getValue('offset', 0);
		// batch norm
		const newShape = new tensor_shape$1(tensor.shape);
		const valRefs = [tensor, mean, variance, scale, offset].map(t=>t.val_ref);
		const out = new tensor_description$1(newShape, dtype, node.name+':0',
			'batch_norm', valRefs, {});
		const results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __batch_norm__primitive = {
		name: 'batch_norm',
		type: 'tensor',
		desc_function: __batch_norm__desc_func,
		doc: new op_doc$1(['input tensor'], ['normalized tensor'],
			'applies batch normalization to the input')
	};


	/*
	---------------------------------
	--------- convolution  ----------
	---------------------------------
	*/
	const __convolution__primitive = {
		name: 'convolution',
		type: 'tensor',
		desc_function: __convolution__desc_func,
		doc: new op_doc$1(['x', 'filter', '(optional) stride', '(optional) padding'],
			['x convolved with filter'],
			'convolves x with filter')
	};

	/*
	---------------------------------
	----------- max_pool  -----------
	---------------------------------
	*/
	const __max_pool__primitive = {
		name: 'max_pool',
		type: 'tensor',
		desc_function: __max_pool__desc_func,
		doc: new op_doc$1(['x', '(optional) filterSize',
			'(optional) stride', '(optional) padding'],
		['max pooling of x'],
		'applies max pooling to x')
	};

	/*
	---------------------------------
	----------- avg_pool  -----------
	---------------------------------
	*/
	const __avg_pool__primitive = {
		name: 'avg_pool',
		type: 'tensor',
		desc_function: __avg_pool__desc_func,
		doc: new op_doc$1(['x', '(optional) filterSize',
			'(optional) stride', '(optional) padding'],
		['average pooling of x'],
		'applies average pooling to x')
	};


	const primitives = [
		__placeholder__primitive,
		__relu__primitive,
		__sigmoid__primitive,
		__tanh__primitive,
		__exp__primitive,
		__matmul__primitive,
		__add__primitive,
		__multiply__primitive,
		__divide__primitive,
		__identity__primitive,
		__if__primitive,
		__scalar__primitive,
		__literals__primitive,
		__parse_json__primitive,
		__parse_json_list__primitive,
		__pow__primitive,
		__sqrt__primitive,
		__get_tensor__primitive,
		__variable__primitive,
		__pack_list__primitive,
		__unpack_list__primitive,
		__softmax__primitive,
		__log__primitive,
		__reduce_sum__primitive,
		__negate__primitive,
		__transpose__primitive,
		__one_hot__primitive,
		__cast__primitive,
		__reduce_avg__primitive,
		__subtract__primitive,
		__abs__primitive,
		__convolution__primitive,
		__gather__primitive,
		__reshape__primitive,
		__js_function__primitive,
		__get_collection__primitive,
		...higherOrderPrimitives,
		__batch_norm__primitive,
		__gather_rows__primitive,
		__max_pool__primitive,
		__avg_pool__primitive,
	].reduce((a,p)=>Object.assign(a, {[p.name]: p}), {});

	/*
	----------------------------------------------------
	------------------ Topoological Sort ---------------
	----------------------------------------------------
	*/

	function _copy_G(G){ // written in old-school JS for speed
		const keys = Object.keys(G);
		let new_G = {};
		for(let i=0; i<keys.length; i++){
			const key = keys[i];
			new_G[key] = {in: G[key].in.slice()};
		}
		return new_G
	}

	// Gives graph an outdirection, inplace
	function _give_out_direction(G){
		Object.keys(G).forEach(k => G[k].out = []);
		Object.keys(G).forEach(k => {
			G[k].in.forEach(k_in => G[k_in].out.push(k));
		});
	}

	// Kahn's algorithm
	// https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
	// note that this modifies the graph, G
	function _side_effects_topological_sort(G){
		let L = [],
			S = Object.keys(G).filter(k => G[k].in.length == 0).sort();
		_give_out_direction(G);
		while(S.length > 0){
			let n = S.pop(),
				new_S = [];
			L.push(n);
			for(let i=G[n].out.length-1; i>=0; i--){
				let m = G[n].out[i];
				G[n].out.splice(i,1);
				if(G[m].in.length == 1){
					new_S.push(m);
					G[m].in = [];
				}	
				else{
					G[m].in.splice(G[m].in.indexOf(n),1);
				}
			}
			S.push(...new_S.sort());
		}
		if(!Object.keys(G).every(k=>G[k].in.length==0)) return false
		return L
	}

	function topological_sort(orig_G){
		let G = _copy_G(orig_G);
		return _side_effects_topological_sort(G)
	}


	function find_ancestors(graph, nodes){
		let stack = [...nodes],
			visited = new Set([]);
		while(stack.length > 0){
			const node = stack.pop();
			visited.add(node);
			stack.push(...graph[node].in.filter(p => !visited.has(p)));
		}
		return visited
	}


	function prune_and_topsort(G, nodes){
		const pruned_G = Array.from(find_ancestors(G, nodes))
			.reduce((acc,k) => Object.assign(acc, {[k]: G[k]}), {});
		return topological_sort(pruned_G)
	}


	const _defaultSliceName = s => s.slice(0,s.lastIndexOf(':'));
	function get_init_subgraphs(nodes, output, init_ops,
		slice_name=_defaultSliceName){
		const init_ops_set = new Set(init_ops),
			node_map = nodes.reduce((acc,n)=>Object.assign(acc,{[n.name]:n}), {}),
			walkable = name => !init_ops_set.has(node_map[name].op),
			graph = nodes.reduce((acc,node) => {
				const inputs = node.input.map(slice_name).filter(walkable);
				return Object.assign(acc, {[node.name]: {in: inputs}})
			}, {}),
			output_nodes = output.map(slice_name),
			init_nodes = nodes.filter(n=>init_ops_set.has(n.op)).map(n=>n.name),
			forward_ancestor = find_ancestors(graph, output_nodes),
			init_ancestor = find_ancestors(graph, init_nodes);
		return {init_deps: 	init_ancestor,
			init_nodes: init_nodes,
			forward: 	new Set([...forward_ancestor, ...init_nodes])}
	}

	function pruneTopsortNodes(nodes, outputNames, prune){
		const stripIndices = arr => arr.map(s => s.slice(0,s.lastIndexOf(':'))),
			nodeDict = nodes.reduce((a,n) => Object.assign(a,{[n.name]: n}), {}),
			nodeDeps = nodes.reduce((a,n) => 
				Object.assign(a,{[n.name]: {in: stripIndices(n.input)}}), {});
		const graph = prune?
			prune_and_topsort(nodeDeps, stripIndices(outputNames)) :
			topological_sort(nodeDeps);
		if(graph === false){
			throw({message: 'Graph contains cycle',
				metaDataIdentifier: 'cyclic_graph'})
		}
		return graph.map(k => nodeDict[k])
	}

	// node to module's list of nodes replacement rule
	// args: node object, module object
	// returns: list of node objects
	function nodeToModule(parentNode, module){
		const rename = s => parentNode.name + '/' + s,
			inputToIndex = module.input.reduce(
				(a,name,i) => Object.assign(a,{[name]:i}), {});
		return module.nodes.map(node => {
			const newNode = {
				name: 	rename(node.name),
				input: 	node.input.map(rename),
				op: 	node.op,
				literal: node.literal};
			if(!inputToIndex.hasOwnProperty(node.name)) return newNode
			return Object.assign(newNode,
				{op: 'identity',
					input: [parentNode.input[inputToIndex[node.name]]]})
		}).concat([{name: parentNode.name,
			input: module.output.map(rename),
			op: 'identity', literal: []}])
	}


	/**
	 * Flattens each module in a node such that each module 
	 * only contains primitive operations.
	 * Additionally, it puts each module's nodes in topologically 
	 * sorted order, and nodes that do not contribute to the output 
	 * are optionally pruned.
	 * @param {`taffy library`} library A taffy library
	 * @param {boolean=} prune Whether to prune nodes that 
	 * don't contribute to a module's output
	 * @return {Object<string, Object<string, any>>} Flattened modules
	 */
	function stage_one(library, prune=true){
		// build dependency graph of modules and find topological ordering
		const origModules = library.modules.reduce(
				(a,x) => Object.assign(a, {[x.name]: x}), {}),
			deps = library.modules.reduce(
				(a,x) => Object.assign(a, {[x.name]: {in: x.module_import}}),{}),
			moduleOrder = topological_sort(deps);
		if(moduleOrder === false){throw('Module dependencies contain a cycle')}
		// flatten modules
		const flattened = moduleOrder.reduce((a, modName)=> {
			const modDeps = new Set(deps[modName].in);
			const origMod = origModules[modName];
			const nodes = pruneTopsortNodes(origMod.nodes, origMod.output, prune)
				.map(node => modDeps.has(node.op)?
					nodeToModule(node, a[node.op]) :
					[node])
				.reduce((x,z) => x.concat(z), []);
			return Object.assign(a, {[modName]: {
				name: 	modName,
				input: 	origMod.input,
				output: origMod.output,
				nodes: 	nodes}})
		}, {});
		return {modules: flattened}
	}

	const isShape = v => {
		try {return v.constructor === constructors.tensor_shape}
		catch(e){return false}
	};

	const isTensor$2 = v => {
		try {return v.constructor === constructors.tensor_description}
		catch(e){return false}
	};

	function quasiToTensor(inputDesc){
		return Object.entries(inputDesc).reduce((a,[k,quasi]) => {
			if(!quasi.hasOwnProperty('shape')){
				throw('input description to stage two must have `shape` key')
			}
			if(!quasi.hasOwnProperty('dtype')){
				throw('input description to stage two must have `dtype` key')
			}
			const shape = isShape(quasi.shape)? quasi.shape :
					new constructors.tensor_shape(quasi.shape),
				tensorDesc = new constructors.tensor_description(shape,
					quasi.dtype, k+':0', 'placeholder', [], {});
			return Object.assign(a, {[k+':0']: tensorDesc})
		}, {})
	}

	// inputDescriptions must be a dictionary with
	// the names of inputs as keys and value descriptions as values
	// all value descriptions must be quasi-tensor descriptions,
	// which are objects with shape and dtype

	/**
	 * Evaluates a specified module using description of its input, 
	 * producing a tensor and value trace.
	 * @param {Object<string, Object<string, any>>} stageOneOut The output 
	 * of `stage_on`, which are flattened modules
	 * @param {string} moduleName The name of the module to be compiled
	 * @param {Object<string, Object<string, any>>} inputDescriptions A 
	 * dictionary with `moduleName`'s input names as keys, 
	 * and dictionary with {shape, dtype} as values
	 * @return {Object<string, any>} A dictionary containing a tensor 
	 * and value trace, and other metadata.
	 */
	function stage_two(stageOneOut, moduleName, inputDescriptions){
		let valueTrace = {},
			tensorTrace = {},
			collections = {};
		Object.entries(quasiToTensor(inputDescriptions)).forEach(([valRef, val])=>{
			const pair = {[valRef]: val};
			Object.assign(valueTrace, pair);
			Object.assign(tensorTrace, pair);
		});
		const flatModule = stageOneOut.modules[moduleName],
			inputNames = new Set(flatModule.input);
		flatModule.nodes.forEach(node => {
			if(inputNames.has(node.name)){return} // inputs already recieved traces
			const fn = inputs => primitives[node.op].desc_function(
				tensorTrace, node, inputs, collections, stageOneOut.modules);
			try {
				const fnOut = fn(node.input.map(ref => valueTrace[ref]));
				Object.assign(valueTrace, fnOut);
			} catch(error){ throw {error, node: node.name, valueTrace}}
		});
		const outputs = flatModule.output.map(k => valueTrace[k]);
		outputs.forEach((t, i) => {if(!isTensor$2(t)){
			const message = `Output #${i} of module is not a tensor`,
				metaData = {i, arg:t, valueTrace};
			throw({message,  metaData, metaDataIdentifier: 'output_not_tensor'})
		}});
		return {val_trace: valueTrace,
			tensor_trace: tensorTrace,
			collections,
			output: outputs.map(t=>t.val_ref),
			output_names: flatModule.output,
			name: moduleName,
			input_descriptions: inputDescriptions,
			input_names: flatModule.input}
	}

	const stripIndex = s => s.slice(0,s.lastIndexOf(':'));

	/**
	 * Transforms the graph to only contain tensor and placeholder operations
	 * @param {Object<string, any>} stageTwoOut The output of `stage_two`
	 * @param {boolean} prune Whether to prune nodes that don't 
	 * contribute to a module's output
	 * @return {Object<string, any>} A dictionary containing 
	 * nodes that implement tensor or placeholder operations, 
	 * and other metadata
	 */
	function stage_three(stageTwoOut, prune=true){
		const {tensor_trace, output, output_names} = stageTwoOut,
			depGraph = Object.entries(tensor_trace)
				.reduce((a, [k, v]) => {
					if(a.hasOwnProperty(stripIndex(k))) return a
					const nodeInput = v.input.map(stripIndex);
					return Object.assign(a, {[stripIndex(k)]: {in: nodeInput}})
				}, {}),
			order = prune?
				prune_and_topsort(depGraph, output.map(stripIndex)) :
				topological_sort(depGraph),
			nodesDict = Object.entries(tensor_trace)
				.reduce((a, [k, {op, input, attr}]) => {
					if(a.hasOwnProperty(stripIndex(k))) return a
					return Object.assign(a, {[stripIndex(k)]: {
						name: stripIndex(k),
						op: op, input: input, attr: attr
					}})
				}, {});
		return {nodes: order.map(k => nodesDict[k]),
			output: output,
			output_names: output_names,
			name: stageTwoOut.name,
			stage_two: stageTwoOut}
	}

	const stringify = JSON.stringify;

	function convert_ref(name_map, ref){
		const idx = ref.lastIndexOf(':');
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
			re_reffed_output = output.map(my_convert_ref);
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
			in_desc = unwrapped.stage_two.input_descriptions;
		return Object.entries(in_desc)
			.reduce((acc,[k,v]) => Object.assign(acc, {[k]: mapInputDesc(v)}),{})
	}

	function getOutDesc(unwrapped){
		const tensor_trace = unwrapped.stage_two.tensor_trace,
			simple_tdesc = ({shape, dtype}) =>
				({dtype:dtype, shape:convert_shape(shape)}),
			{output_names} = unwrapped;
		return unwrapped.output
			.reduce((acc,k,i) =>
				Object.assign(acc,
					{[output_names[i]]: simple_tdesc(tensor_trace[k])}),
			{})
	}


	function op_conversion_get_tensor(node){
		const {shape,fill,dtype} = node.attr,
			s_shape = stringify(convert_shape(shape.shape)),
			s_dtype = stringify(dtype);
		let out = '';
		if(fill.type == 'scalar') out = `tf.fill(${s_shape},${fill.val},${s_dtype})`;
		else if(fill.type == 'symbol'){
			out = ({
				'ones': 	`tf.ones(${s_shape},${s_dtype})`,
				'zeros': 	`tf.zeros(${s_shape},${s_dtype})`,
				'normal': 	`tf.randomNormal(${s_shape},0,1,${s_dtype})`,
				'truncated_normal': `tf.truncatedNormal(${s_shape},0,1,${s_dtype})`
			})[fill.symbol];
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
			final = `[tf.div(${sum}, ${scalar})]`;
		return final
	}

	function op_conversion_add(node){
		return '[' + node.input.slice(1).reduce((a,b) =>
			`tf.add(${a},${b})`, node.input[0]) + ']'
	}

	function op_conversion_mul(node){
		return '[' + node.input.slice(1).reduce((a,b) =>
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
			result = `tf.pow(${positiveBase}, ${exp}).mul(${reSign})`;
		return `[${result}]`
	}

	function convolutionWrapper(node){
		const [x, filter] = node.input,
			{stride, padding, shape} = node.attr;
		const ND = shape.length - 2,
			availConvs = new Set([1, 2]);
		if(!availConvs.has(ND)){
			throw(`${ND}D convolution not yet supported, ` +
				`only (${[...availConvs]})D supported`)
		}
		let result = '';
		if(ND === 1){
			result = `tf.conv1d(${x},${filter},${stride[0]},${stringify(padding)})`;
		}else if(ND === 2){
			result = `tf.conv2d(${x},${filter},` +
				`${stringify(stride)},${stringify(padding)})`;
		}
		return `[${result}]`
	}

	function batchNormConversion(node){
		let [x, ...rest] = node.input;
		rest = rest.map(t => `${t}.gather(tf.zeros([${x}.shape[0]], 'int32'))`);
		return `[tf.batchNormalization(${x}, ${rest.slice(0,2)},0.0001,`+
			`${rest.slice(2)})]`
	}

	function gatherRowsConversion(node){
		const [x, inds] = node.input;
		/* // Should be used when tf.gatherND has gradient function
		const positions = `tf.stack([tf.range(0,${inds}.shape[0])` +
			`.cast(${inds}.dtype),${inds}],1)`
		return `[tf.gatherND(${x}, ${positions})]`
		*/
		const pos = `tf.range(0,${inds}.shape[0]).mul(${x}.shape[1])`+
			`.cast(${inds}.dtype).add(${inds})`;
		return `[${x}.flatten().gather(${pos})]`
	}

	function poolingConversion(op, node){
		const x = node.input[0];
		const {filterSize, stride, padding, shape} = node.attr;
		if(!(shape.length == 3 || shape.length == 4)){
			throw('Pooling only supported for inputs of rank 3 or 4.')
		}
		return `[tf.${op}(${x},${filterSize},${stride},${stringify(padding)})]`
	}

	const opConversionMap = {
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
		gather: n => `[tf.gather(${n.input.slice(0,2)},${n.attr.axis})]`,
		reshape: n => `[tf.reshape(${n.input[0]},[${n.attr.shapeEncoding
		.map(x => typeof(x)!=typeof('')? x : n.input[0]+'.shape['+x+']')}])]`,
		batch_norm: batchNormConversion,
		gather_rows: gatherRowsConversion,
		max_pool: n => poolingConversion('maxPool', n),
		avg_pool: n => poolingConversion('avgPool', n),
	};


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
			expression = `this.tf.tidy(()=>{${body}return ${map};})`;
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
	}.toString().replace(/\r|\n|\t|\s\s+/g, '');

	const inputObjectToInputs = 'const inputs = !Array.isArray(inputObject)? '+
		'inputObject : this.input_names.reduce('+
		'(acc,name,i)=>Object.assign(acc,{[name]:inputObject[i]}),{});';

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
			check_statement = inputObjectToInputs +
				'if(check){this.check_inputs(inputs);}',
			return_statement = `return this.tf.tidy(()=>{${inner_tidy}})`,
			composed_fn = 'function(inputObject, check=true){' +
				`${check_statement}${return_statement}}`;
		return composed_fn
	}

	const optimize = function(inputObject,
		lossName=undefined,
		batch_size=32,
		iterations=100,
		optimizer=undefined,
		check_inputs=true){
		const inputs = !Array.isArray(inputObject)? inputObject :
			this.input_names.reduce((acc,name,i) => 
				Object.assign(acc,{[name]:inputObject[i]}),{});
		const tf = this.tf;
		const available_out = Object.entries(this.output_descriptions)
			.filter(([,{shape}]) => shape.length==0)
			.map(([k,])=>k);
		if(available_out.length==0){
			throw({message: 'there are no scalar outputs, thus no eligible loss'})
		}
		if(available_out.length>1 && lossName === undefined){
			console.log(`Warning: defaulting to "${available_out[0]}" as loss.`);
		}
		const loss = lossName? lossName : available_out[0];
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
		if(check_inputs){this.check_inputs(inputs);}
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
	}.toString().replace(/\r|\n|\t|\s\s+/g, '');

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
	}.toString().replace(/\r|\n|\t|\s\s+/g, '');


	function unwrapped_to_constructor(unwrapped){
		const {re_nodes, re_output, name_map} = re_ref(unwrapped),
			inDesc = getInDesc(unwrapped),
			outDesc = getOutDesc(unwrapped),
			subgraphs = get_init_subgraphs(re_nodes, re_output, ['variable'],
				s => s.slice(0,s.lastIndexOf('['))),
			forwardFn = get_forward(unwrapped, re_nodes,inDesc,name_map,subgraphs),
			passObj = o => `JSON.parse(${stringify(stringify(o))})`,
			collections = Object.entries(unwrapped.stage_two.collections).reduce(
				(acc,[k,v]) => Object.assign(acc,{[k]:Array.from(Object.keys(v))}),
				{});
		const fn_string = '(function(tfLib){"use_strict";' +

			'try{this.tf = tfLib || tf;}' +
			'catch(e){throw(' +
			'"A tf library must be supplied or be available as a global")}' +

			`this.implements_module = ${stringify(unwrapped.name)};` +
			`this.collections = ${passObj(collections)};` +
			`this.inherit_vars = ${inherit_vars};` +
			`this.input_names = ${stringify(unwrapped.stage_two.input_names)};` +
			`this.name_map = ${passObj(name_map)};` +

			'this.inverse_name_map = Object.entries(this.name_map)' +
			'.reduce((acc,[k,v]) => Object.assign(acc,{[v]:k}), {});' +

			`this.input_descriptions = ${passObj(inDesc)};` +
			`this.output_descriptions = ${passObj(outDesc)};` +
			`this.variables = ${get_variables(re_nodes, subgraphs)};` +
			`this.check_inputs = ${check_inputs};` +
			`this.forward = ${forwardFn};` +
			`this.optimize = ${optimize};` +
			'})';
		return fn_string
	}

	const stringify$1 = JSON.stringify;

	function convert_ref$1(ref){
		const idx = ref.lastIndexOf(':');
		const node = ref.slice(0,idx);
		return `graph['${node}'][${ref.slice(idx+1)}]`
	}

	function convert_shape$1(shape){
		const arr = shape.map(c => isNaN(c)? 'None' : ''+c);
		return '['+arr.join(', ')+']'
	}

	function getInDesc$1(unwrapped){
		const mapInputDesc = ({dtype, shape}) =>
				({dtype: dtype, shape: convert_shape$1(shape)}),
			in_desc = unwrapped.stage_two.input_descriptions;
		return Object.entries(in_desc)
			.reduce((acc,[k,v]) => Object.assign(acc, {[k]: mapInputDesc(v)}),{})
	}

	function op_conversion_get_tensor$1(node){
		const {shape,fill,dtype} = node.attr;
		const s_shape = convert_shape$1(shape.shape);
		const s_dtype = stringify$1(dtype);
		let out = '';
		if(fill.type == 'scalar'){
			out = `tf.cast(tf.fill(${s_shape},${fill.val}), ${s_dtype})`;
		}
		else if(fill.type == 'symbol'){
			out = ({
				'ones': 	`tf.ones(${s_shape},${s_dtype})`,
				'zeros': 	`tf.zeros(${s_shape},${s_dtype})`,
				'normal': 	`tf.random_normal(${s_shape},0,1,${s_dtype})`,
				'truncated_normal': `tf.truncated_normal(${s_shape},0,1,${s_dtype})`
			})[fill.symbol];
			if(out===undefined) throw('Unsupported fill symbol')
		}
		else throw('Unsupported fill')
		return `[${out}]`
	}

	const convertTFStride = arr => stringify$1([1, ...arr, 1]);

	function convolutionWrapper$1(node){
		const [x, filter] = node.input;
		const {stride, padding, shape} = node.attr;
		const tfPad = typeof(padding)==typeof('')? padding.toUpperCase() : padding;
		const ND = shape.length - 2;
		const availConvs = new Set([1, 2, 3]);
		if(!availConvs.has(ND)){
			throw(`${ND}D convolution not yet supported, ` +
				`only (${[...availConvs]})D supported`)
		}
		let result = '';
		if(ND == 1){
			result = `tf.nn.conv1d(${x}, ${filter}, ` +
	            `${stride[0]}, ${stringify$1(tfPad)})`;
		} else {
			result = `tf.nn.conv${ND}d(${x}, ${filter}, ` +
	            `${convertTFStride(stride)}, ${stringify$1(tfPad)})`;
		}
		return `[${result}]`
	}

	function batchNormConversion$1(node){
		return `[tf.nn.batch_normalization(${node.input.slice(0,3)},`
	        + `${node.input.slice(3,5).reverse()}, tf.constant(1e-4))]`
	}

	function gatherRowsConversion$1(node){
		const [x, inds] = node.input;
		const range = `tf.cast(tf.range(0,${inds}.shape[0]), ${inds}.dtype)`;
		const positions = `tf.stack([${range},${inds}],1)`;
		return `[tf.gather_nd(${x}, ${positions})]`
	}

	function poolingConversion$1(op, node){
		// `op` is 'max_pool' or 'avg_pool'
		const x = node.input[0];
		const {filterSize, stride, padding, shape} = node.attr;
		const tfPad = typeof(padding)==typeof('')? padding.toUpperCase() : padding;
		const tfStride = convertTFStride(Array(shape.length-2).fill(stride));
		const tfFilter = convertTFStride(Array(shape.length-2).fill(filterSize));
		if(!(shape.length == 4 || shape.length == 5)){
			throw('Pooling only supported for inputs of rank 4 or 5.')
		}
		const tfOp = shape.length == 5? `${op}3d` : op;
		return `[tf.nn.${tfOp}(${x},${tfFilter},${tfStride},${stringify$1(tfPad)})]`
	}

	function convertPow(node){
		const dtype = `${node.input[0]}.dtype`; // TODO: pick higher of the dtypes
		const casted = node.input.map(s => `tf.cast(${s}, ${dtype})`);
		return `[tf.pow(${casted})]`
	}

	const unreffedOpConversionMap = {
		get_tensor: op_conversion_get_tensor$1,
		placeholder: () => {throw('placeholder shouldn\'t have been called...')},
		variable: node => `[self.variables[${stringify$1(node.name)}]]`,
		relu: node => `[${node.input.map(r => `tf.nn.relu(${r})`)}]`,
		sigmoid: node => `[${node.input.map(r => `tf.nn.sigmoid(${r})`)}]`,
		tanh: node => `[${node.input.map(r => `tf.nn.tanh(${r})`)}]`,
		exp: node => `[tf.exp(${node.input[0]})]`,
		matmul: node => `[tf.linalg.matmul(${node.input})]`,
		add: node => `[${node.input.join(' + ')}]`,
		multiply: node => `[${node.input.join(' * ')}]`,
		divide: node => `[tf.div(${node.input})]`,
		subtract: node => `[tf.subtract(${node.input})]`,
		pow: convertPow,
		sqrt: node => `[tf.sqrt(${node.input[0]})]`,
		softmax: node => `[tf.nn.softmax(${node.input[0]})]`,
		log: node => `[tf.math.log(${node.input[0]}+tf.constant(1e-8))]`,
		reduce_sum: n => `[tf.reduce_sum(${n.input[0]}, `+
	        `axis=${stringify$1(n.attr.axis)})]`,
		reduce_avg: node => `[tf.reduce_mean(${node.input[0]}, `+
	        `axis=${stringify$1(node.attr.axis)})]`,
		negate: node => `[-1 * ${node.input[0]}]`,
		transpose: n => `[tf.transpose(${n.input[0]}, ` +
			`perm=${stringify$1(n.attr.perm)})]`,
		one_hot: node => `[tf.one_hot(${node.input[0]}, ${node.attr.n_colls})]`,
		cast: n => `[tf.cast(${n.input[0]}, ${stringify$1(n.attr.dtype)})]`,
		abs: node => `[tf.abs(${node.input[0]})]`,
		convolution: convolutionWrapper$1,
		gather: n => `[tf.gather(${n.input.slice(0,2)},axis=${n.attr.axis})]`,
		reshape: n => `[tf.reshape(${n.input[0]},[${n.attr.shapeEncoding
		.map(x => typeof(x)!=typeof('')? x : n.input[0]+'.shape['+x+']')}])]`,
		batch_norm: batchNormConversion$1,
		gather_rows: gatherRowsConversion$1,
		max_pool: n => poolingConversion$1('max_pool', n),
		avg_pool: n => poolingConversion$1('avg_pool', n),
	};

	const opConversionMap$1 = Object.entries(unreffedOpConversionMap)
		.reduce((acc, [k,fn]) => Object.assign(acc, {[k]: 
			node => fn(Object.assign({}, node, 
				{input: node.input.map(convert_ref$1)}))
		}), {});

	function make_init_fn(nodes, subgraphs){
		const {init_deps, init_nodes} = subgraphs;
		const varConversion = n => `[tf.Variable(${convert_ref$1(n.input[0])})]`;
		const overriddenOps = Object.assign({}, opConversionMap$1, 
			{variable: varConversion});
		const preamble = ['self.tf = tf', 'graph = {}'];
		const main = nodes
			.filter(n => n.op !== 'placeholder')
			.filter(n => init_deps.has(n.name))
			.map(n => `graph['${n.name}'] = ${overriddenOps[n.op](n)}`);
		const assign = 'self.variables = {'+
			init_nodes
				.map(s => `"${s}": graph['${s}'][0]`)
				.join(',')
	        +'}';
		const body = [...preamble, ...main, assign].map(s => `\t${s}`);
		const lines = ['def __init__(self, tf):', ...body];
		return lines
	}

	function get_call_fn(unwrapped, nodes, inDesc, subgraphs){
		const ingest = 'ingested = self.ingest_input(inputs)';
		const inputAcquisition = Object.keys(inDesc)
			.map(k => `graph['${k}'] = [ingested["${k}"]]`);
		const preamble = ['tf = self.tf', 'graph = {}',
			ingest, ...inputAcquisition];
		const main = nodes
			.filter(n => n.op !== 'placeholder')
			.filter(n => subgraphs.forward.has(n.name))
			.map(n => `graph['${n.name}'] = ${opConversionMap$1[n.op](n)}`);
		const return_value_inner = unwrapped.output_names
			.map((name,i) => `"${name}":` +
	            `${convert_ref$1(unwrapped.output[i])}`)
			.join(',');
		const return_statement = `return {${return_value_inner}}`;
		const body = [...preamble, ...main, return_statement].map(s => `\t${s}`);
		const lines = ['def __call__(self, inputs):', ...body];
		return lines
	}

	function makePythonClass(name, init, call, ingest_input){
		const coalesce = lines => lines.map(s => `\t${s}`).join('\n');
		const classStr = `class ${name}:\n`
	        + coalesce(init) + '\n'
	        + coalesce(call) + '\n'
	        + coalesce(ingest_input) + '\n';
		return classStr
	}

	function make_ingest_input(inDesc){
		const unnamedPrefix = 'INPUT_';
		const intFromUnnamed = s => +s.slice(unnamedPrefix.length);
		const allUnnamed = arr => arr.every(s => s.startsWith(unnamedPrefix)) && 
	        arr.map(intFromUnnamed).every(n => Number.isInteger(n));
		const input_names = Object.keys(inDesc)
			.sort((a,b) => (allUnnamed([a,b])?
				intFromUnnamed(a)<intFromUnnamed(b) : a<b)? -1 : 1);
		const body = [
			'if isinstance(recieved, dict):',
			'\tingested = recieved',
			'elif isinstance(recieved, list) or isinstance(recieved, tuple):',
			`\tinput_names = ${stringify$1(input_names)}`,
			'\tingested = {k:v for k,v in zip(input_names, recieved)}',
			'else: raise ValueError("Input is not a dict, tuple, or list")',
			'return ingested'
		];
		const lines = ['def ingest_input(self, recieved, check=True):',
			...body.map(s => `\t${s}`)];
		return lines
	}

	function unwrapped_to_factory(unwrapped){
		const {nodes, output, name} = unwrapped;
		const inDesc = getInDesc$1(unwrapped);
		const subgraphs = get_init_subgraphs(nodes, output, ['variable']);
		const init = make_init_fn(nodes, subgraphs);
		const call = get_call_fn(unwrapped, nodes, inDesc, subgraphs);
		const ingest_input = make_ingest_input(inDesc);
		return makePythonClass(name, init, call, ingest_input)
	}

	const stages = {
		one: stage_one, 
		two: stage_two,
		three: stage_three
	};

	const packagers = {
		'TensorFlow.js': unwrapped_to_constructor,
		'TensorFlow Python': unwrapped_to_factory
	};

	function puller(library, module_name, input_descriptions, prune=true){
		const one_out = stage_one(library, prune),
			two_out = stage_two(one_out, module_name, input_descriptions),
			three_out = stage_three(two_out, prune);
		return three_out
	}

	function pull_and_package(packager_name,
		library, module_name, input_descriptions, prune=true){
		const pulled = puller(library, module_name, input_descriptions, prune);
		return packagers[packager_name](pulled)
	}

	exports.stages = stages;
	exports.packagers = packagers;
	exports.puller = puller;
	exports.pull_and_package = pull_and_package;
	exports.constructors = constructors;
	exports.primitives = primitives;

	Object.defineProperty(exports, '__esModule', { value: true });

})));
