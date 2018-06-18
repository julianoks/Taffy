import {stage_one} from './puller/stage_one.js'
import {stage_two} from './puller/stage_two.js'
import {stage_three} from './puller/stage_three.js'

import {unwrapped_to_constructor as tfjs_constructor} from './packager/tfjs.js'

export const stages = {
	one: stage_one, 
	two: stage_two,
	three: stage_three
};

export const packagers = {
	tfjs: tfjs_constructor
};

export function puller(library, module_name, input_descriptions){
	const one_out = stage_one(library),
		two_out = stage_two(one_out, module_name, input_descriptions),
		three_out = stage_three(two_out)
	return three_out
}

export function pull_and_package(packager_name,
	library, module_name, input_descriptions){
	const pulled = puller(library, module_name, input_descriptions)
	return packagers[packager_name](pulled)
}
