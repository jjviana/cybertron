// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"
	"strings"
)

type RNN struct {
	Type          string
	InputSize     int
	HiddenSize    int
	NumLayers     int
	Bidirectional bool
	Parameters    map[string]*Parameter
}

func (r *RNN) LoadStateDictEntry(k string, v any) (err error) {
	if strings.HasPrefix(k, "weight_") || strings.HasPrefix(k, "bias_") {
		if r.Parameters == nil {
			r.Parameters = make(map[string]*Parameter)
		}
		t, err := AnyToTensor(v, nil)
		if err != nil {
			return fmt.Errorf("RNN: state dict key %q: %w", k, err)
		}
		r.Parameters[k] = &Parameter{
			Data:         t,
			RequiresGrad: false,
		}
		return nil
	}

	return fmt.Errorf("RNN: unknown state dict key %q", k)
}
