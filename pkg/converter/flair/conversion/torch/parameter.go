// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
)

type Parameter struct {
	Data         *pytorch.Tensor
	RequiresGrad bool
}

type RebuildParameter struct{}

func (RebuildParameter) Call(args ...any) (any, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("RebuildParameter: want 3 args, got %#v", args)
	}

	p := new(Parameter)

	var ok bool
	p.Data, ok = args[0].(*pytorch.Tensor)
	if !ok {
		return nil, fmt.Errorf("RebuildParameter: want 1st arg *Tensor, got %#v", args[0])
	}
	p.RequiresGrad, ok = args[1].(bool)
	if !ok {
		return nil, fmt.Errorf("RebuildParameter: want 2nd arg bool, got %#v", args[1])
	}

	// The third parameter is for backwards compatibility: the general
	// expectation is that backward_hooks is an empty OrderedDict.
	backwardHooks, ok := args[2].(*types.OrderedDict)
	if !ok {
		return nil, fmt.Errorf("RebuildParameter: want 3rd arg *OrderedDict, got %#v", args[2])
	}
	if l := backwardHooks.Len(); l != 0 {
		return nil, fmt.Errorf("RebuildParameter: want 3rd empty OrderedDict, got length %d", l)
	}
	return p, nil
}
