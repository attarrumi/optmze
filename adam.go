package optmze

import (
	"math"
	"sync"
)

type Optimizer interface {
	Update(params, grads []float64)
}

// AdamOptimizer is an optimizer that implements the Adam algorithm.
type AdamOptimizer1 struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	m            []float64
	v            []float64
	t            int
}

// NewAdamOptimizer creates a new AdamOptimizer with the given parameters.
func NewAdamOptimizer1(learningRate, beta1, beta2, epsilon float64) *AdamOptimizer1 {
	return &AdamOptimizer1{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
	}
}

// Update updates the parameters with the gradients using the Adam algorithm.
func (o *AdamOptimizer1) Update(params, grads []float64) {
	o.t++
	if o.m == nil {
		o.m = make([]float64, len(params))
	}
	if o.v == nil {
		o.v = make([]float64, len(params))
	}
	var wg sync.WaitGroup

	for i := range params {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			o.m[i] = o.Beta1*o.m[i] + (1-o.Beta1)*grads[i]
			o.v[i] = o.Beta2*o.v[i] + (1-o.Beta2)*grads[i]*grads[i]
			mHat := o.m[i] / (1 - math.Pow(o.Beta1, float64(o.t)))
			vHat := o.v[i] / (1 - math.Pow(o.Beta2, float64(o.t)))
			params[i] -= o.LearningRate * mHat / (math.Sqrt(vHat) + o.Epsilon)
		}(i)
	}
	wg.Wait()

}

// AdamOptimizer is an optimizer that implements the Adam algorithm.
type AdamOptimizer struct {
	// LearningRate is the learning rate for the optimizer.
	LearningRate float64
	// Beta1 is the exponential decay rate for the first moment estimates.
	Beta1 float64
	// Beta2 is the exponential decay rate for the second moment estimates.
	Beta2 float64
	// Epsilon is a small constant for numerical stability.
	Epsilon float64
	// m is the first moment vector.
	m []float64
	// v is the second moment vector.
	v []float64
	// t is the timestep counter.
	t int
}

// NewAdamOptimizer creates a new AdamOptimizer with the given parameters.
func NewAdamOptimizer(learningRate, beta1, beta2, epsilon float64) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Epsilon:      epsilon,
	}
}

// Update updates the parameters with the gradients using the Adam algorithm.
func (o *AdamOptimizer) Update(params, grads []float64) {
	o.t++
	if o.m == nil {
		o.m = make([]float64, len(params))
	}
	if o.v == nil {
		o.v = make([]float64, len(params))
	}
	for i := range params {
		o.m[i] = o.Beta1*o.m[i] + (1-o.Beta1)*grads[i]
		o.v[i] = o.Beta2*o.v[i] + (1-o.Beta2)*grads[i]*grads[i]
		mHat := o.m[i] / (1 - math.Pow(o.Beta1, float64(o.t)))
		vHat := o.v[i] / (1 - math.Pow(o.Beta2, float64(o.t)))
		params[i] -= o.LearningRate * mHat / (math.Sqrt(vHat) + o.Epsilon)
	}
}
