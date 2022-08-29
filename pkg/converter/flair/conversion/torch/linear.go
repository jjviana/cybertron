// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"errors"
	"fmt"

	"github.com/nlpodyssey/gopickle/pytorch"
)

type LinearClass struct{}

type Linear struct {
	Module
	InFeatures  int
	OutFeatures int
	Bias        *pytorch.Tensor
	Weight      *pytorch.Tensor
}

func (LinearClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("LinearClass: unsupported arguments: %#v", args)
	}
	return &Linear{}, nil
}

func (l *Linear) PyDictSet(key, value any) error {
	if err := l.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, ErrUnknownModuleDictKey) {
		return fmt.Errorf("linear: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("linear: want string key, got %#v", key)
	}

	switch k {
	case "in_features":
		l.InFeatures, ok = value.(int)
		if !ok {
			return fmt.Errorf("linear.in_features: want int, got %#v", value)
		}
	case "out_features":
		l.OutFeatures, ok = value.(int)
		if !ok {
			return fmt.Errorf("linear.out_features: want int, got %#v", value)
		}
	default:
		return fmt.Errorf("linear: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (l *Linear) LoadStateDictEntry(k string, v any) (err error) {
	switch k {
	case "bias":
		l.Bias, err = AnyToTensor(v, []int{l.OutFeatures})
	case "weight":
		l.Weight, err = AnyToTensor(v, []int{l.OutFeatures, l.InFeatures})
	default:
		err = fmt.Errorf("linear: unknown state dict key %q", k)
	}
	return err
}
