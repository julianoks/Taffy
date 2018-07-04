[![Build Status](https://travis-ci.com/julianoks/Taffy.svg?token=cyeFuKKiwnyJyRizTQxr&branch=master)](https://travis-ci.com/julianoks/Taffy)

TODO
- [x] taffy puller optional pruning
- [x] convolution op
- [ ] higher-order operations
	- [ ] map
	- [ ] fold/reduce
		- RNNs should be implemented as a fold/reduce that returns x, f(x), f(f(x)), ....
- [ ] research building tfjs packager constructor without eval, while maintaining serializability
	- I think this is unnecessary for now.