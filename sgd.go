package optmze

type SGD struct {
	LearningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{
		LearningRate: learningRate,
	}
}

func (o *SGD) Update(params, grads []float64) {
	for i := range params {
		params[i] -= o.LearningRate * grads[i]
	}
}
