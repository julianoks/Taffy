import {test as util_test} from './util.js'
import {test as puller_test} from './puller.js'
import {test as packager_test} from './packager.js'

export function test(tf){
	console.log('Taffy Util tests...')
	util_test()

	console.log()
	console.log('Taffy Puller tests...')
	puller_test()
	
	console.log()
	console.log('Taffy Packager tests...')
	packager_test(tf)
}

