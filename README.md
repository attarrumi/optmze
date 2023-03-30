start := time.Now()

	net := NewNeuralNetwork(2, 4, 2)
	opt := NewAdamOptimizer(0.01, 0.9, 0.999, 1e-8)

		inputs := [][]float64{
		{0, 0},
		{1, 1},
		{1, 0},
		{0, 1},
	}
	outputs := [][]float64{
		{0, 0},
		{1, 1},
		{1, 1},
		{0, 0},
	}

	net.Train(inputs, outputs, opt, 100, 10)

	for i, v := range inputs {
		if i == 10 {
			break
		}
		out := net.FeedForward(v)
		fmt.Println(v, outputs[i], out)
	}
	last := time.Since(start)
	fmt.Printf("Time %s \n", last)