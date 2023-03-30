package optmze

import "math"

type AdagradOptimizer struct {
	LearningRate float64
	Epsilon      float64
	Cache        []float64
}

func NewAdagradOptimizer(learningRate, epsilon float64) *AdagradOptimizer {
	return &AdagradOptimizer{
		LearningRate: learningRate,
		Epsilon:      epsilon,
	}
}

func (o *AdagradOptimizer) Update(params, grads []float64) {
	if o.Cache == nil {
		o.Cache = make([]float64, len(params))
	}

	for i := range params {
		o.Cache[i] += grads[i] * grads[i]
		params[i] -= o.LearningRate * grads[i] / (math.Sqrt(o.Cache[i]) + o.Epsilon)
	}
}
