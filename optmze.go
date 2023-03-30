package optmze

import (
	"fmt"
	"math"
)

// NeuralNet is a simple neural network with one hidden layer.
type NeuralNet struct {
	// W1 is the weight matrix for the first layer.
	W1 [][]float64
	// B1 is the bias vector for the first layer.
	B1 []float64
	// W2 is the weight matrix for the second layer.
	W2 [][]float64
	// B2 is the bias vector for the second layer.
	B2 []float64
	h  []float64

	n     int
	total float64
}

func NewNeuralNet(input, hidden, output int) *NeuralNet {
	net := &NeuralNet{
		W1: make([][]float64, input),
		B1: make([]float64, hidden),
		W2: make([][]float64, hidden),
		B2: make([]float64, output),
	}
	for i := range net.W1 {
		net.W1[i] = make([]float64, hidden)
		for j := range net.W1[i] {
			net.W1[i][j] = RandFloat(0, 1)
		}

	}
	for i := range net.W2 {
		net.W2[i] = make([]float64, output)
		for j := range net.W2[i] {
			net.W2[i][j] = RandFloat(0, 1)
		}
	}

	return net
}

func (net *NeuralNet) FeedForward(x []float64) []float64 {
	h := make([]float64, len(net.B1))
	for i := range h {
		for j := range x {
			h[i] += x[j] * net.W1[j][i]
		}
		h[i] += net.B1[i]
		h[i] = math.Tanh(h[i])
	}

	net.h = h
	y := make([]float64, len(net.B2))
	for i := range y {
		for j := range h {
			y[i] += h[j] * net.W2[j][i]
		}
		y[i] += net.B2[i]
		y[i] = sigmoid(y[i])
	}
	return y
}

// number of epochs.
func (net *NeuralNet) Train(input [][]float64, output [][]float64, opt Optimizer, epochs, batch int) {
	params := net.Params()
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(input); i += batch {

			end := i + batch
			if end > len(input) {
				end = len(input)
			}
			batchX := input[i:end]
			batchY := output[i:end]

			for o, x := range batchX {
				y := batchY[o]
				yPred := net.FeedForward(x)

				dyPred := make([]float64, len(yPred))
				for j := range yPred {
					dyPred[j] = yPred[j] - y[j]
					net.total += (yPred[j] - y[j]) * (yPred[j] - y[j])
				}

				dh := make([]float64, len(net.B1))
				for ii := range dh {
					errorSum := 0.0
					for j := range dyPred {
						errorSum += dyPred[j] * net.W2[ii][j]
					}
					dh[ii] = errorSum * (1 - net.h[ii]*net.h[ii])
				}

				dW2 := make([][]float64, len(net.W2))
				for ii := range dW2 {
					dW2[ii] = make([]float64, len(net.W2[ii]))
					for j := range dW2[ii] {
						dW2[ii][j] = net.h[j] * dyPred[j]
					}
				}

				dB2 := dyPred
				dW1 := make([][]float64, len(net.W1))
				for ii := range dW1 {
					dW1[ii] = make([]float64, len(net.W1[ii]))
					for j := range dW1[ii] {
						dW1[ii][j] = x[ii] * dh[j]
					}
				}
				dB1 := dh
				net.UpdateParams(dW1, dB1, dW2, dB2)
				opt.Update(params, net.Grads(dW1, dB1, dW2, dB2))
				net.n++

				if (net.n+1)%(epoch/15) == 0 {
					avgLoss := net.total / float64(net.n)
					fmt.Printf("num = %d loss %.4f \n", net.n+1, avgLoss)
				}
			}
		}
	}
}

// Params returns the network's parameters as a flat slice.
func (net *NeuralNet) Params() []float64 {
	params := make([]float64, 0)
	for i := range net.W1 {
		params = append(params, net.W1[i]...)
	}
	params = append(params, net.B1...)
	for i := range net.W2 {
		params = append(params, net.W2[i]...)
	}
	params = append(params, net.B2...)
	return params
}

func (net *NeuralNet) UpdateParams(dW1 [][]float64, dB1 []float64, dW2 [][]float64, dB2 []float64) {
	for i := range net.W1 {
		for j := range net.W1[i] {
			net.W1[i][j] -= dW1[i][j]
		}
	}
	for i := range net.B1 {
		net.B1[i] -= dB1[i]
	}
	for i := range net.W2 {
		for j := range net.W2[i] {
			net.W2[i][j] -= dW2[i][j]
		}
	}
	for i := range net.B2 {
		net.B2[i] -= dB2[i]
	}
}

// Grads returns the gradients of the network's parameters as a flat slice.
func (net *NeuralNet) Grads(dW1 [][]float64, dB1 []float64, dW2 [][]float64, dB2 []float64) []float64 {
	grads := make([]float64, 0)
	for i := range dW1 {
		grads = append(grads, dW1[i]...)
	}
	grads = append(grads, dB1...)
	for i := range dW2 {
		grads = append(grads, dW2[i]...)
	}
	grads = append(grads, dB2...)
	return grads
}
