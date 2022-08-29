// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/nlpodyssey/gopickle/types"
)

type LSTMClass struct{}

type LSTM struct {
	Module
	Mode          string
	InputSize     int
	HiddenSize    int
	NumLayers     int
	Bidirectional bool
	Bias          bool
	BatchFirst    bool
	Dropout       float64
	AllWeights    [][]string
	Parameters    map[string]*Parameter
}

func (LSTMClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("LSTMClass: unsupported arguments: %#v", args)
	}
	return &LSTM{}, nil
}

func (l *LSTM) PyDictSet(key, value any) error {
	if err := l.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, ErrUnknownModuleDictKey) {
		return fmt.Errorf("LSTM: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("LSTM: want string key, got %#v", key)
	}

	switch k {
	case "mode":
		l.Mode, ok = value.(string)
		if !ok {
			return fmt.Errorf("LSTM.mode: want string, got %#v", value)
		}
	case "input_size":
		l.InputSize, ok = value.(int)
		if !ok {
			return fmt.Errorf("LSTM.input_size: want int, got %#v", value)
		}
	case "hidden_size":
		l.HiddenSize, ok = value.(int)
		if !ok {
			return fmt.Errorf("LSTM.hidden_size: want int, got %#v", value)
		}
	case "num_layers":
		l.NumLayers, ok = value.(int)
		if !ok {
			return fmt.Errorf("LSTM.num_layers: want int, got %#v", value)
		}
	case "bias":
		l.Bias, ok = value.(bool)
		if !ok {
			return fmt.Errorf("LSTM.bias: want bool, got %#v", value)
		}
	case "batch_first":
		l.BatchFirst, ok = value.(bool)
		if !ok {
			return fmt.Errorf("LSTM.batch_first: want bool, got %#v", value)
		}
	case "dropout":
		l.Dropout, ok = value.(float64)
		if !ok {
			return fmt.Errorf("LSTM.dropout: want float64, got %#v", value)
		}
	case "bidirectional":
		l.Bidirectional, ok = value.(bool)
		if !ok {
			return fmt.Errorf("LSTM.bidirectional: want bool, got %#v", value)
		}
	case "_all_weights":
		v, ok := value.(*types.List)
		if !ok {
			return fmt.Errorf("LSTM._all_weights: want *List, got %#v", value)
		}
		return l.setAllWeights(v)
	default:
		return fmt.Errorf("LSTM: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (l *LSTM) setAllWeights(pyList *types.List) error {
	l.AllWeights = make([][]string, pyList.Len())

	for i, x := range *pyList {
		xl, ok := x.(*types.List)
		if !ok {
			return fmt.Errorf("LSTM: all_weights: want item type *List, got %#v", x)
		}

		list := make([]string, xl.Len())
		for j, y := range *xl {
			list[j], ok = y.(string)
			if !ok {
				return fmt.Errorf("LSTM: all_weights[%d]: want item type string, got %#v", i, y)
			}
		}
		l.AllWeights[i] = list
	}

	return nil
}

func (l *LSTM) LoadStateDictEntry(k string, v any) (err error) {
	if strings.HasPrefix(k, "weight_") || strings.HasPrefix(k, "bias_") {
		p, err := GetModuleParameter[*Parameter](l.Module, k)
		if err != nil {
			return fmt.Errorf("LSTM: %w", err)
		}
		if l.Parameters == nil {
			l.Parameters = make(map[string]*Parameter)
		}
		t, err := AnyToTensor(v, p.Data.Size)
		if err != nil {
			return fmt.Errorf("LSTM: state dict key %q: %w", k, err)
		}
		if !reflect.DeepEqual(*p.Data, *t) {
			return fmt.Errorf("LSTM: tensor loaded from state dict %q differs from parameter's data", k)
		}
		l.Parameters[k] = p
		return nil
	}

	return fmt.Errorf("LSTM: unknown state dict key %q", k)
}
