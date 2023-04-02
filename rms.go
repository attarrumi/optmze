package optmze

import "math"

type RMSprop struct {
	LearningRate float64
	Decay        float64
	Epsilon      float64
	Cache        []float64
}

func NewRMSprop(learningRate, decay, epsilon float64) *RMSprop {
	return &RMSprop{
		LearningRate: learningRate,
		Decay:        decay,
		Epsilon:      epsilon,
	}
}

func (o *RMSprop) Update(params, grads []float64) {
	if o.Cache == nil {
		o.Cache = make([]float64, len(params))
	}
	for i := range params {
		o.Cache[i] = o.Decay*o.Cache[i] + (1-o.Decay)*grads[i]*grads[i]
		params[i] -= o.LearningRate * grads[i] / (math.Sqrt(o.Cache[i]) + o.Epsilon)
	}
}
