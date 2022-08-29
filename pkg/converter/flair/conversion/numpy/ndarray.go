// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package numpy

import (
	"fmt"

	"github.com/nlpodyssey/gopickle/types"
)

type NDArrayClass struct{}

type NDArray struct {
	Shape []int
	DType string
}

func (n NDArrayClass) PyNew(args ...any) (_ any, err error) {
	if len(args) == 0 || len(args) > 2 {
		return nil, fmt.Errorf("NDArrayClass: want 1 or 2 arguments, got %#v", args)
	}

	nda := new(NDArray)

	nda.Shape, err = n.convertShape(args[0])
	if err != nil {
		return nil, err
	}

	if len(args) > 1 {
		var ok bool
		nda.DType, ok = args[1].(string)
		if !ok {
			return nil, fmt.Errorf("NDArrayClass: want 2nd arg dtype (string), got %#v", args[1])
		}
	}

	return nda, nil
}

func (NDArrayClass) convertShape(arg any) ([]int, error) {
	t, ok := arg.(*types.Tuple)
	if !ok {
		return nil, fmt.Errorf("NDArrayClass: want shape *Tuple, got %#v", arg)
	}

	s := make([]int, t.Len())
	for i := range s {
		s[i], ok = t.Get(i).(int)
		if !ok {
			return nil, fmt.Errorf("NDArrayClass: want shape Tuple of int, got %#v", arg)
		}
	}

	return s, nil
}
