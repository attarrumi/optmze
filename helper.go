package optmze

import "math/rand"

func RandFloat(min, max float64) float64 {
	r := rand.New(rand.NewSource(99))
	return min + r.NormFloat64()*(max-min)
}
