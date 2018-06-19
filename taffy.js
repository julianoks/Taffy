(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
	typeof define === 'function' && define.amd ? define(['exports'], factory) :
	(factory((global.taffy = {})));
}(this, (function (exports) { 'use strict';

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

	function pruneAndTopsortNodes(nodes, outputNames){
		const stripIndices = arr => arr.map(s => s.slice(0,s.lastIndexOf(':'))),
			nodeDict = nodes.reduce((a,n) => Object.assign(a,{[n.name]: n}), {}),
			nodeDeps = nodes.reduce(
				(a,n) => Object.assign(a,{[n.name]: {in: stripIndices(n.input)}}),
				{});
		return prune_and_topsort(nodeDeps, stripIndices(outputNames))
			.map(k => nodeDict[k])
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

	function stage_one(library){
		// build dependency graph of modules and find topological ordering
		const origModules = library.modules.reduce(
				(a,x) => Object.assign(a, {[x.name]: x}), {}),
			deps = library.modules.reduce(
				(a,x) => Object.assign(a, {[x.name]: {in: x.module_import}}),{}),
			moduleOrder = topological_sort(deps);
		if(moduleOrder === false){throw('Module dependencies contain a cycle')}
		// flatten modules
		const flattened = moduleOrder.reduce((a, modName)=> {
			const modDeps = new Set(deps[modName].in),
				origModule = origModules[modName],
				nodes = pruneAndTopsortNodes(origModule.nodes, origModule.output)
					.map(node => modDeps.has(node.op)?
						nodeToModule(node, a[node.op]) :
						[node])
					.reduce((x,z) => x.concat(z), []);
			return Object.assign(a, {[modName]: {
				name: 	modName,
				input: 	origModule.input,
				output: origModule.output,
				nodes: 	nodes}})
		}, {});
		return {modules: flattened}
	}

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
			this.output;
			this.doc = doc;
		}

	};

	const {op_doc, tensor_description, tensor_shape} = constructors;

	/*
	---------------------------------
	---------- helper fns -----------
	---------------------------------
	*/
	const is_tensor = obj => obj.constructor === tensor_description;

	function broadcast_shapes(tensors){
		const rank = Math.max(...tensors.map(t=>t.shape.length)),
			shapes = tensors
				.map(t => Array(rank-t.shape.length).fill(1).concat(t.shape)),
			res_shape = Array(rank).fill().map((_, i) => {
				const dims = shapes.map(s => s[i]),
					symbols = [...new Set(dims.filter(isNaN))],
					numbersNotOne = [...new Set(dims.filter(x=>!isNaN(x) && x!=1))];
				if(numbersNotOne.length > 1){
					throw({message: 'tensors not broadcastable', i: i, dims: dims})	
				}
				if(symbols.length > 1){
					throw({message: 'cannot broadcast tensors with >1 collated ' +
						'symbolic dimensions', i:i, dims: dims})
				}
				if(symbols.length == 1){
					return numbersNotOne.length == 0? symbols[0] : numbersNotOne[0]
				}
				return numbersNotOne.length == 0? 1 : numbersNotOne[0]
			}),
			res_dtype = tensors[0].dtype;
		if(!tensors.every(t => t.dtype == res_dtype)){
			throw({message: 'tensors are of different dtypes'})
		}
		return {shape: new tensor_shape(res_shape), dtype: res_dtype}
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
			throw('The placeholder desc_function shouldn\'t be called!')
		},
		doc: new op_doc([],
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
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const results = tensors.reduce((acc, tensor, i) => {
			const shape = new tensor_shape(tensor.shape),
				dtype = tensor.dtype,
				val_ref = node.name + ':' + i,
				out = new tensor_description(shape, dtype, val_ref, 'relu',
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
		doc: new op_doc(['tensor'], ['ReLU, ie f(x)=max(0,x)'],
			'ReLU activation function')
	};


	/*
	---------------------------------
	------------ sigmoid  -----------
	---------------------------------
	*/
	function __sigmoid__desc_func(tensor_trace, node, tensors){
		if(tensors.length<1) throw({message: 'must take >=1 tensors'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const results = tensors.reduce((acc, tensor, i) => {
			const shape = new tensor_shape(tensor.shape),
				dtype = tensor.dtype,
				val_ref = node.name + ':' + i,
				out = new tensor_description(shape, dtype, val_ref, 'sigmoid',
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
		doc: new op_doc(['tensor'], ['sigmoid of input'],
			'sigmoid activation function')
	};


	/*
	---------------------------------
	------------- tanh  -------------
	---------------------------------
	*/
	function __tanh__desc_func(tensor_trace, node, tensors){
		if(tensors.length<1) throw({message: 'must take >=1 tensors'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const results = tensors.reduce((acc, tensor, i) => {
			const shape = new tensor_shape(tensor.shape),
				dtype = tensor.dtype,
				val_ref = node.name + ':' + i,
				out = new tensor_description(shape, dtype, val_ref, 'tanh',
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
		doc: new op_doc(['tensor'], ['tanh of input'], 'tanh activation function')
	};


	/*
	---------------------------------
	-------------- exp  -------------
	---------------------------------
	*/
	function __exp__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const tensor = tensors[0],
			shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description(shape, dtype, node.name+':0', 'exp',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __exp__primitive = {
		name: 'exp',
		type: 'tensor',
		desc_function: __exp__desc_func,
		doc: new op_doc(['tensor'], ['elementwise exp, ie f(x)=e^x'],
			'exponential function, ie f(x)=e^x')
	};


	/*
	---------------------------------
	-------------- abs  -------------
	---------------------------------
	*/
	function __abs__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const tensor = tensors[0],
			shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description(shape, dtype, node.name+':0', 'abs',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __abs__primitive = {
		name: 'abs',
		type: 'tensor',
		desc_function: __abs__desc_func,
		doc: new op_doc(['tensor'], ['elementwise absolute value, ie f(x)=|x|'],
			'abs value function, ie f(x)=|x|')
	};


	/*
	---------------------------------
	------------- negate  -----------
	---------------------------------
	*/
	function __negate__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const tensor = tensors[0],
			shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description(shape, dtype, node.name+':0', 'negate',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __negate__primitive = {
		name: 'negate',
		type: 'tensor',
		desc_function: __negate__desc_func,
		doc: new op_doc(['tensor'], ['negation, ie f(x)=-x'],
			'negation function, ie f(x)=-x')
	};


	/*
	---------------------------------
	------------- sqrt  -------------
	---------------------------------
	*/
	function __sqrt__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==1) throw({message: 'must take 1 tensor'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const tensor = tensors[0],
			shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description(shape, dtype, node.name+':0', 'sqrt',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __sqrt__primitive = {
		name: 'sqrt',
		type: 'tensor',
		desc_function: __sqrt__desc_func,
		doc: new op_doc(['tensor'], ['elementwise square root'],
			'square root function')
	};


	/*
	---------------------------------
	------------- matmul  -----------
	---------------------------------
	*/
	function __matmul__desc_func(tensor_trace, node, tensors){
		if(tensors.length!=2) throw({message: 'must take 2 tensors'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
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
						throw({message: 'tensors not broadcastable',
							i: i, dims: [d1,d2]})
					}
					return d1
				}
				if(isNaN(d1) && isNaN(d2)) return d1
				return isNaN(d1)? d1 : d2
			}),
			d1 = tensors[0].shape.slice(-2)[0],
			d2 = tensors[1].shape.slice(-1)[0],
			shape = new tensor_shape(prefix.concat([d1,d2])),
			dtype = tensors[0].dtype,
			out = new tensor_description(shape, dtype, node.name+':0', 'matmul',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __matmul__primitive = {
		name: 'matmul',
		type: 'tensor',
		desc_function: __matmul__desc_func,
		doc: new op_doc(['tensor 1', 'tensor 2'],
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
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const {shape, dtype} = broadcast_shapes(tensors),
			out = new tensor_description(shape, dtype, node.name+':0', 'add',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __add__primitive = {
		name: 'add',
		type: 'tensor',
		desc_function: __add__desc_func,
		doc: new op_doc(['...tensor values'], ['sum of tensors'],
			'variadic function that adds n>=1 tensors')
	};


	/*
	---------------------------------
	----------- subtract  -----------
	---------------------------------
	*/
	function __subtract__desc_func(tensor_trace, node, tensors){
		if(tensors.length!==2) throw({message: 'must take 2 tensors'})
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const {shape, dtype} = broadcast_shapes(tensors),
			out = new tensor_description(shape, dtype, node.name+':0', 'subtract',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __subtract__primitive = {
		name: 'subtract',
		type: 'tensor',
		desc_function: __subtract__desc_func,
		doc: new op_doc(['tensor 1', 'tensor 2'],
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
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const {shape, dtype} = broadcast_shapes(tensors),
			out = new tensor_description(shape, dtype, node.name+':0', 'multiply',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __multiply__primitive = {
		name: 'multiply',
		type: 'tensor',
		desc_function: __multiply__desc_func,
		doc: new op_doc(['...tensor values'],
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
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const {shape, dtype} = broadcast_shapes(tensors),
			out = new tensor_description(shape, dtype, node.name+':0', 'divide',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __divide__primitive = {
		name: 'divide',
		type: 'tensor',
		desc_function: __divide__desc_func,
		doc: new op_doc(['tensor 1', 'tensor 2'],
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
		tensors.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const {shape, dtype} = broadcast_shapes(tensors),
			out = new tensor_description(shape, dtype, node.name+':0', 'pow',
				tensors.map(t => t.val_ref), {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __pow__primitive = {
		name: 'pow',
		type: 'tensor',
		desc_function: __pow__desc_func,
		doc: new op_doc(['tensor 1', 'tensor 2'],
			['The power of tensor 1 to tensor 2'],
			'The power of tensor 1 to tensor 2')
	};


	/*
	---------------------------------
	------------- scalar  -----------
	---------------------------------
	*/
	function __scalar__desc_func(tensor_trace, node, inputs){
		if(inputs.length < 1) throw({message: 'must take at least one input'})
		if(isNaN(+inputs[0])) throw({message: 'first input must be a number'})
		const num = +inputs[0],
			dtype = typeof(inputs[1])==='string'? inputs[1] : 'float32',
			shape = new tensor_shape([]),
			out = new tensor_description(shape, dtype, node.name+':0', 'scalar',
				[], {num:num,dtype:dtype}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __scalar__primitive = {
		name: 'scalar',
		type: 'tensor',
		desc_function: __scalar__desc_func,
		doc: new op_doc(['scalar value', '(optional) dtype'], ['a scalar'],
			'produces a tensor scalar')
	};


	/*
	---------------------------------
	---------- get_tensor  ----------
	---------------------------------
	*/

	function __get_tensor__desc_func(tensor_trace, node, inputs){
		const [given_shape, given_fill, given_dtype] = inputs,
			dtype = given_dtype || 'float32';
		if(given_shape == undefined) throw({message: 'shape must be defined'})
		if(given_fill == undefined) throw({message: 'fill must be defined'})
		let shape, fill;
		try{shape = new tensor_shape(given_shape);}
		catch(e){
			throw({message: 'couldn\'t convert provided shape to tensor shape',
				error: e})
		}
		const supported_fills = new Set(['ones', 'zeros',
			'normal', 'truncated_normal']);
		if(supported_fills.has(given_fill)){
			fill = {type: 'symbol', symbol: given_fill};
		} else if(!isNaN(+given_fill)){
			fill = {type: 'scalar', val: +given_fill};
		}else{
			throw({message: 'fill not supported', val: given_fill})
		}
		const attr = {shape: shape, fill: fill, dtype: dtype},
			out = new tensor_description(shape, dtype, node.name+':0', 'get_tensor',
				[], attr),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __get_tensor__primitive = {
		name: 'get_tensor',
		type: 'tensor',
		desc_function: __get_tensor__desc_func,
		doc: new op_doc(['shape', 'fill', '(optional) dtype'],
			['tensor'], 'produces a tensor')
	};

	/*
	---------------------------------
	----------- variable  -----------
	---------------------------------
	*/
	function __variable__desc_func(tensor_trace, node, inputs){
		if(inputs.length != 1) throw({message: 'must take exactly 1 input'})
		if(!is_tensor(inputs[0])) throw({message: 'input must be a tensor'})
		const results = inputs.reduce((acc,v,i)=> {
			const name = node.name+':' + i,
				{shape, dtype} = v,
				tshape = new tensor_shape(shape),
				new_tensor = new tensor_description(tshape, dtype, name, 'variable',
					[v.val_ref], {});
			return Object.assign(acc, {[name]: new_tensor})
		}, {});
		Object.assign(tensor_trace, results);
		return results
	}

	const __variable__primitive = {
		name: 'variable',
		type: 'tensor',
		desc_function: __variable__desc_func,
		doc: new op_doc(['tensor'], ['tensor'],
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
		doc: new op_doc(['...inputs'], ['...inputs'], 'forwards inputs, unchanged')
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
		doc: new op_doc([], ['...literals'], 'forwards literals, unchanged')
	};


	/*
	---------------------------------
	---------- parse_json  ----------
	---------------------------------
	*/
	function __parse_json__desc_func(tensor_trace, node, inputs){
		return inputs.reduce((a,v,i) => {
			try{return Object.assign(a, {[node.name+':'+i]: JSON.parse(v)})}
			catch(e){throw({message:'Couldn\'t parse JSON literal', i:i,val:v})}
		}, {})
	}

	const __parse_json__primitive = {
		name: 'parse_json',
		type: 'control',
		desc_function: __parse_json__desc_func,
		doc: new op_doc(['...inputs (literals)'], ['...inputs'],
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
		doc: new op_doc(['JSON representation of a list'],
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
		doc: new op_doc(['boolean', 'value', 'value'], ['one of the values'],
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
		doc: new op_doc(['...values'], ['array containing the values'],
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
		doc: new op_doc(['array of values'], ['...values'],
			'unpacks the input values from an array')
	};



	/*
	---------------------------------
	------------ softmax  -----------
	---------------------------------
	*/
	function __softmax__desc_func(tensor_trace, node, inputs){
		if(inputs.length!==1) throw({message: 'must take 1 tensor'})
		inputs.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const tensor = inputs[0],
			shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description(shape, dtype, node.name+':0', 'softmax',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __softmax__primitive = {
		name: 'softmax',
		type: 'tensor',
		desc_function: __softmax__desc_func,
		doc: new op_doc(['tensor'], ['softmax of tensor'],
			'applies the softmax function to a tensor')
	};



	/*
	---------------------------------
	------------- log  --------------
	---------------------------------
	*/
	function __log__desc_func(tensor_trace, node, inputs){
		if(inputs.length!==1) throw({message: 'must take 1 tensor'})
		inputs.forEach((t,i) => {
			if(!is_tensor(t)) throw({message: 'argument not a tensor', i:i, arg: t})
		});
		const tensor = inputs[0],
			shape = new tensor_shape(tensor.shape),
			dtype = tensor.dtype,
			out = new tensor_description(shape, dtype, node.name+':0', 'log',
				[tensor.val_ref], {}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __log__primitive = {
		name: 'log',
		type: 'tensor',
		desc_function: __log__desc_func,
		doc: new op_doc(['tensor'], ['natural log of tensor'],
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
		if(!is_tensor(inputs[0])){
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
			shape = new tensor_shape(raw_shape),
			out = new tensor_description(shape, dtype, node.name+':0', 'reduce_sum',
				[tensor.val_ref], {axis:axis}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __reduce_sum__primitive = {
		name: 'reduce_sum',
		type: 'tensor',
		desc_function: __reduce_sum__desc_func,
		doc: new op_doc(['tensor', 'axis; a integer or array of integers'],
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
		if(!is_tensor(inputs[0])) throw({message: 'input must be a tensor'})
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
			shape = new tensor_shape(raw_shape),
			out = new tensor_description(shape, dtype, node.name+':0', 'reduce_avg',
				[tensor.val_ref], {axis:axis}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __reduce_avg__primitive = {
		name: 'reduce_avg',
		type: 'tensor',
		desc_function: __reduce_avg__desc_func,
		doc: new op_doc(['tensor', 'axis; a integer or array of integers'],
			['a scalar'],
			'averages a tensor')
	};

	/*
	---------------------------------
	----------- transpose  ----------
	---------------------------------
	*/
	function __transpose__desc_func(tensor_trace, node, inputs){
		if(inputs.length == 1 || inputs.length == 2){
			throw({message: 'must take one or two inputs'})
		}
		if(!is_tensor(inputs[0])) throw({message: 'first input must be a tensor'})
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
			shape = new tensor_shape(perm.map(i=>tensor.shape[i])),
			out = new tensor_description(shape, dtype, node.name+':0', 'transpose',
				[tensor.val_ref], {perm:perm}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __transpose__primitive = {
		name: 'transpose',
		type: 'tensor',
		desc_function: __transpose__desc_func,
		doc: new op_doc(['tensor', 'permutation (optional)'],
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
		if(!(is_tensor(inputs[0]) && inputs[0].shape.length == 1)){
			throw({message: 'first input must be a rank 1 tensor'})
		}
		if(isNaN(inputs[1]) || Math.floor(+inputs[1])<2){
			throw({message: 'second input must be a number >=2'})
		}
		const tensor = inputs[0],
			n_colls = Math.floor(+inputs[1]),
			dtype = tensor.dtype,
			shape = new tensor_shape([tensor.shape[0], n_colls]),
			out = new tensor_description(shape, dtype, node.name+':0', 'one_hot',
				[tensor.val_ref], {n_colls:n_colls}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __one_hot__primitive = {
		name: 'one_hot',
		type: 'tensor',
		desc_function: __one_hot__desc_func,
		doc: new op_doc(['indices (as rank 1 tensor)', 'number of columns'],
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
		if(!is_tensor(tensor)) throw({message: 'first input must be a tensor'})
		if(!(typeof(given_dtype) == 'string' || is_tensor(given_dtype))){
			throw({message: 'second input must be a string or a tensor'})	
		}
		const dtype = is_tensor(given_dtype)? given_dtype.dtype : given_dtype,
			shape = new tensor_shape(tensor.shape),
			out = new tensor_description(shape, dtype, node.name+':0', 'cast',
				[tensor.val_ref], {dtype:dtype}),
			results = {[out.val_ref]: out};
		Object.assign(tensor_trace, results);
		return results
	}

	const __cast__primitive = {
		name: 'cast',
		type: 'tensor',
		desc_function: __cast__desc_func,
		doc: new op_doc(['tensor', 'dtype (a string)'],
			['tensor cast as dtype'],
			'casts a tensor to a specified dtype')
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
	].reduce((a,p)=>Object.assign(a, {[p.name]: p}), {});

	const isShape = v => v.constructor === constructors.tensor_shape;
	const isTensor = v => v.constructor === constructors.tensor_description;

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
	function stage_two(stageOneOut, moduleName, inputDescriptions){
		let valueTrace = {},
			tensorTrace = {};
		Object.entries(quasiToTensor(inputDescriptions)).forEach(([valRef, val])=>{
			const pair = {[valRef]: val};
			Object.assign(valueTrace, pair);
			Object.assign(tensorTrace, pair);
		});
		const flatModule = stageOneOut.modules[moduleName],
			inputNames = new Set(flatModule.input);
		flatModule.nodes.forEach(node => {
			if(inputNames.has(node.name)){return} // inputs already recieved traces
			const fn = inputs => primitives[node.op]
					.desc_function(tensorTrace, node, inputs),
				fnOut = fn(node.input.map(ref => valueTrace[ref]));
			Object.assign(valueTrace, fnOut);
		});
		const outputs = flatModule.output.map(k => valueTrace[k]);
		if(!outputs.every(isTensor)){
			throw('Output of module in stage one is not a tensor')
		}
		return {val_trace: valueTrace,
			tensor_trace: tensorTrace,
			output: outputs.map(t=>t.val_ref),
			output_names: flatModule.output,
			name: moduleName,
			input_descriptions: inputDescriptions}
	}

	const stripIndex = s => s.slice(0,s.lastIndexOf(':'));

	function stage_three(stageTwoOut){
		const {tensor_trace, output, output_names} = stageTwoOut,
			depGraph = Object.entries(tensor_trace)
				.reduce((a, [k, v]) => {
					if(a.hasOwnProperty(stripIndex(k))) return a
					const nodeInput = v.input.map(stripIndex);
					return Object.assign(a, {[stripIndex(k)]: {in: nodeInput}})
				}, {}),
			order = prune_and_topsort(depGraph, output.map(stripIndex)),
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
				`${check_statement}${return_statement}}`;
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
			forwardFn = get_forward(unwrapped, re_nodes, inDesc,name_map,subgraphs),
			passObj = o => `JSON.parse(${stringify(stringify(o))})`;
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
			'})';
		return eval(fn)
	}

	const stages = {
		one: stage_one, 
		two: stage_two,
		three: stage_three
	};

	const packagers = {
		tfjs: unwrapped_to_constructor
	};

	function puller(library, module_name, input_descriptions){
		const one_out = stage_one(library),
			two_out = stage_two(one_out, module_name, input_descriptions),
			three_out = stage_three(two_out);
		return three_out
	}

	function pull_and_package(packager_name,
		library, module_name, input_descriptions){
		const pulled = puller(library, module_name, input_descriptions);
		return packagers[packager_name](pulled)
	}

	exports.stages = stages;
	exports.packagers = packagers;
	exports.puller = puller;
	exports.pull_and_package = pull_and_package;

	Object.defineProperty(exports, '__esModule', { value: true });

})));