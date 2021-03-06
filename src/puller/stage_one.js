import {topological_sort, prune_and_topsort} from '../util/graph.js'

function pruneTopsortNodes(nodes, outputNames, prune){
	const stripIndices = arr => arr.map(s => s.slice(0,s.lastIndexOf(':'))),
		nodeDict = nodes.reduce((a,n) => Object.assign(a,{[n.name]: n}), {}),
		nodeDeps = nodes.reduce((a,n) => 
			Object.assign(a,{[n.name]: {in: stripIndices(n.input)}}), {})
	const graph = prune?
		prune_and_topsort(nodeDeps, stripIndices(outputNames)) :
		topological_sort(nodeDeps)
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
			(a,name,i) => Object.assign(a,{[name]:i}), {})
	return module.nodes.map(node => {
		const newNode = {
			name: 	rename(node.name),
			input: 	node.input.map(rename),
			op: 	node.op,
			literal: node.literal}
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
export function stage_one(library, prune=true){
	// build dependency graph of modules and find topological ordering
	const origModules = library.modules.reduce(
			(a,x) => Object.assign(a, {[x.name]: x}), {}),
		deps = library.modules.reduce(
			(a,x) => Object.assign(a, {[x.name]: {in: x.module_import}}),{}),
		moduleOrder = topological_sort(deps)
	if(moduleOrder === false){throw('Module dependencies contain a cycle')}
	// flatten modules
	const flattened = moduleOrder.reduce((a, modName)=> {
		const modDeps = new Set(deps[modName].in)
		const origMod = origModules[modName]
		const nodes = pruneTopsortNodes(origMod.nodes, origMod.output, prune)
			.map(node => modDeps.has(node.op)?
				nodeToModule(node, a[node.op]) :
				[node])
			.reduce((x,z) => x.concat(z), [])
		return Object.assign(a, {[modName]: {
			name: 	modName,
			input: 	origMod.input,
			output: origMod.output,
			nodes: 	nodes}})
	}, {})
	return {modules: flattened}
}
