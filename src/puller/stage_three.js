import {topological_sort, prune_and_topsort} from '../util/graph.js'

const stripIndex = s => s.slice(0,s.lastIndexOf(':'))

/**
 * Transforms the graph to only contain tensor and placeholder operations
 * @param {Object<string, any>} stageTwoOut The output of `stage_two`
 * @param {boolean} prune Whether to prune nodes that don't 
 * contribute to a module's output
 * @return {Object<string, any>} A dictionary containing 
 * nodes that implement tensor or placeholder operations, 
 * and other metadata
 */
export function stage_three(stageTwoOut, prune=true){
	const {tensor_trace, output, output_names} = stageTwoOut,
		depGraph = Object.entries(tensor_trace)
			.reduce((a, [k, v]) => {
				if(a.hasOwnProperty(stripIndex(k))) return a
				const nodeInput = v.input.map(stripIndex)
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
			}, {})
	return {nodes: order.map(k => nodesDict[k]),
		output: output,
		output_names: output_names,
		name: stageTwoOut.name,
		stage_two: stageTwoOut}
}
