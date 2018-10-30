module.exports = (config) => {
	var configDict = {
		captureTimeout: 60000,
		browserNoActivityTimeout : 60000,
		browserDisconnectTimeout : 10000,
		browserDisconnectTolerance : 1,
		autoWatch: false,
		browsers: ['Chrome'],
		browserConsoleLogOptions: {
			level: 'error',
			format: '%b %T: %m',
			terminal: false
		},
		colors: true,
		files: ['tape.js', {pattern: 'src/test/*_test.js'}],
		frameworks: ['tap'],
		logLevel: config.LOG_ERROR, //'LOG_DEBUG',
		plugins: [
			'karma-rollup-preprocessor',
			'karma-chrome-launcher',
			'karma-tap',
			'karma-tap-pretty-reporter'
		],
		preprocessors: {'src/test/*_test.js': ['rollup']},
		rollupPreprocessor: {
			external: ['tape'],
			plugins: [require('rollup-plugin-buble')],
			output: {
				format: 'iife',
				name: 'blah',
				globals: {'tape': 'tape'},
				sourcemap: false // 'inline'
			}
		},
		reporters: ['tap-pretty'],
		singleRun: true,
		tapReporter: {prettify: require('tap-spec')}
	};
	config.set(configDict);
};