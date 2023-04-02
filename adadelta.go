package optmze

import "math"

type Adadelta struct {
	Decay   float64
	Epsilon float64
	Eg2     []float64
	Edx2    []float64
}

func NewAdadelta(decay, epsilon float64) *Adadelta {
	return &Adadelta{
		Decay:   decay,
		Epsilon: epsilon,
	}
}

func (o *Adadelta) Update(params, grads []float64) {
	if o.Eg2 == nil {
		o.Eg2 = make([]float64, len(params))
	}
	if o.Edx2 == nil {
		o.Edx2 = make([]float64, len(params))
	}
	for i := range params {
		o.Eg2[i] = o.Decay*o.Eg2[i] + (1-o.Decay)*grads[i]*grads[i]
		dx := math.Sqrt(o.Edx2[i]+o.Epsilon) / math.Sqrt(o.Eg2[i]+o.Epsilon) * grads[i]
		params[i] += dx
		o.Edx2[i] = o.Decay*o.Edx2[i] + (1-o.Decay)*dx*dx
	}
}
