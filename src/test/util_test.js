import {topological_sort} from '../util/graph.js'
import tape from 'tape'

// test topological sort
function test_topological_sort(){
	function test_on_G(G){
		let keys = Object.keys(G),
			top_sort = topological_sort(G)
		for(let i in keys){
			let ix = top_sort.indexOf(keys[i])
			if(ix==-1){return false}
			for(let j in G[keys[i]].in){
				if(top_sort.indexOf(G[keys[i]].in[j]) >= ix){return false}
			}
		}
		return true
	}
	let tests = [
		{
			'a': {'in': []},
			'foo': {'in': []},
			'b': {'in': ['a', 'foo']},
			'c': {'in': ['a','b']},
			'd': {'in': ['c']},
			'e': {'in': ['d','c']},
			'f': {'in': ['e']},
			'g': {'in': ['f']},
			'y': {'in': []},
			'z': {'in': []},
		},]
	return tests.map(test_on_G).every(x=>x)
}
tape('topological sort', t => {
	t.equal(test_topological_sort(), true)
	t.end()
})
