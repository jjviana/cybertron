// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package numpy

import (
	"fmt"

	"github.com/nlpodyssey/gopickle/types"
)

type MultiarrayReconstruct struct{}

func (MultiarrayReconstruct) Call(args ...any) (any, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("MultiarrayReconstruct: want 3 args, got %#v", args)
	}
	subType, ok := args[0].(types.PyNewable)
	if !ok {
		return nil, fmt.Errorf("MultiarrayReconstruct: want 1st arg PyNewable, got %#v", args[0])
	}
	shape, ok := args[1].(*types.Tuple)
	if !ok {
		return nil, fmt.Errorf("MultiarrayReconstruct: want 2nd arg *Tuple, got %#v", args[1])
	}
	dataType, ok := args[2].([]byte)
	if !ok {
		return nil, fmt.Errorf("MultiarrayReconstruct: want 3rd arg []byte, got %#v", args[2])
	}
	return subType.PyNew(shape, string(dataType))
}
