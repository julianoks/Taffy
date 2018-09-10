export function valid_C_identifier(str){
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


export const constructors = {
	library: function(modules, tensors=[], doc=''){
		if(!list_of(modules, constructors.module)){
			throw('`modules` must be a list of module objects')
		}
		if(typeof tensors !== typeof {}){
			throw('`tensors` must either be an associative array or undefined')
		}

		this.modules = modules // list of `module` objects
		this.tensors = tensors // name/tensor pairs (as associative array)
		this.doc = doc // free-form documentation for the collection of modules
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
		this.name = name
		// ordered list of input node names (as string)
		this.input = input
		// ordered list of output value references
		this.output = output
		// unordered list of `node` objects
		this.nodes = nodes
		// list of module names (as strings) to import as operations
		this.module_import = module_import
		// module's documentation, as an `op_doc` object
		this.doc = doc
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
		const notTensor = n => n.constructor !== constructors.tensor_description
		if((!(literal.length==0 || literal.every(notTensor)))){
			throw('`literal` must be a list of literals')
		}

		// a valid C identifier, unique among node names in the module
		this.name = name
		// the identifier of the operation to implement
		this.op = op
		// an ordered list of input value references
		this.input = input
		// an ordered list of javascript literals
		this.literal = literal
	},

	tensor_shape: function(integerVec){
		var shape = integerVec
			.map(x => Number.isSafeInteger(x)? Math.floor(x) : x)

		const entriesOK = shape.every(x => {
			return (typeof x === typeof '' && valid_C_identifier(x)) ||
				Number.isSafeInteger(x)
		})

		if((typeof shape !== typeof []) || !entriesOK){
			throw('`shape` must be a vector of integers or ' +
				'strings that are valid C identifiers')
		}

		this.shape = shape
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

		this.shape = shape.shape
		this.dtype = dtype
		this.val_ref = val_ref
		this.op = op
		this.input = input
		this.attr = attr
	},

	op_doc: function(input, output, doc){
		if(!list_of(input, ''.constructor)){
			throw('`input` must be a list of strings')
		}
		if(!list_of(output, ''.constructor)){
			throw('`output` must be a list of strings')
		}

		this.input = input
		this.output = output
		this.doc = doc
	}

}


