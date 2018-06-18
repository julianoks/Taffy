
/*
----------------------------------------------------
------------------ Topoological Sort ---------------
----------------------------------------------------
*/

function _copy_G(G){ // written in old-school JS for speed
	const keys = Object.keys(G)
	let new_G = {}
	for(let i=0; i<keys.length; i++){
		const key = keys[i]
		new_G[key] = {in: G[key].in.slice()}
	}
	return new_G
}

// Gives graph an outdirection, inplace
function _give_out_direction(G){
	Object.keys(G).forEach(k => G[k].out = [])
	Object.keys(G).forEach(k => {
		G[k].in.forEach(k_in => G[k_in].out.push(k))
	})
}

// Kahn's algorithm
// https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
// note that this modifies the graph, G
function _side_effects_topological_sort(G){
	let L = [],
		S = Object.keys(G).filter(k => G[k].in.length == 0).sort()
	_give_out_direction(G)
	while(S.length > 0){
		let n = S.pop(),
			new_S = []
		L.push(n)
		for(let i=G[n].out.length-1; i>=0; i--){
			let m = G[n].out[i]
			G[n].out.splice(i,1)
			if(G[m].in.length == 1){
				new_S.push(m)
				G[m].in = []
			}	
			else{
				G[m].in.splice(G[m].in.indexOf(n),1)
			}
		}
		S.push(...new_S.sort())
	}
	if(!Object.keys(G).every(k=>G[k].in.length==0)) return false
	return L
}

export function topological_sort(orig_G){
	let G = _copy_G(orig_G)
	return _side_effects_topological_sort(G)
}


function find_ancestors(graph, nodes){
	let stack = [...nodes],
		visited = new Set([])
	while(stack.length > 0){
		const node = stack.pop()
		visited.add(node)
		stack.push(...graph[node].in.filter(p => !visited.has(p)))
	}
	return visited
}


export function prune_and_topsort(G, nodes){
	const pruned_G = Array.from(find_ancestors(G, nodes))
		.reduce((acc,k) => Object.assign(acc, {[k]: G[k]}), {})
	return topological_sort(pruned_G)
}


const _defaultSliceName = s => s.slice(0,s.lastIndexOf(':'))
export function get_init_subgraphs(nodes, output, init_ops,
	slice_name=_defaultSliceName){
	const init_ops_set = new Set(init_ops),
		node_map = nodes.reduce((acc,n)=>Object.assign(acc,{[n.name]:n}), {}),
		walkable = name => !init_ops_set.has(node_map[name].op),
		graph = nodes.reduce((acc,node) => {
			const inputs = node.input.map(slice_name).filter(walkable)
			return Object.assign(acc, {[node.name]: {in: inputs}})
		}, {}),
		output_nodes = output.map(slice_name),
		init_nodes = nodes.filter(n=>init_ops_set.has(n.op)).map(n=>n.name),
		forward_ancestor = find_ancestors(graph, output_nodes),
		init_ancestor = find_ancestors(graph, init_nodes)
	return {init_deps: 	init_ancestor,
		init_nodes: init_nodes,
		forward: 	new Set([...forward_ancestor, ...init_nodes])}
}

