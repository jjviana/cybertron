// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"errors"
	"fmt"

	"github.com/nlpodyssey/gopickle/types"
)

type Module struct {
	Training              bool
	Parameters            *types.OrderedDict
	Buffers               *types.OrderedDict
	BackwardHooks         *types.OrderedDict
	ForwardHooks          *types.OrderedDict
	ForwardPreHooks       *types.OrderedDict
	StateDictHooks        *types.OrderedDict
	LoadStateDictPreHooks *types.OrderedDict
	Modules               *types.OrderedDict
}

func GetSubModule[T any](mod Module, name string) (v T, err error) {
	m, ok := mod.Modules.Get(name)
	if !ok {
		return v, fmt.Errorf("torch module not found: %q", name)
	}
	v, ok = m.(T)
	if !ok {
		return v, fmt.Errorf("torch module %q: want type %T, got %T: %#v", name, v, m, m)
	}
	return v, nil
}

func GetModuleParameter[T any](mod Module, name string) (v T, err error) {
	p, ok := mod.Parameters.Get(name)
	if !ok {
		return v, fmt.Errorf("torch module parameter not found: %q", name)
	}
	v, ok = p.(T)
	if !ok {
		return v, fmt.Errorf("torch module parameter %q: want type %T, got %T: %#v", name, v, p, p)
	}
	return v, nil
}

var ErrUnknownModuleDictKey = errors.New("module: unknown __dict__ key")

func (m *Module) PyDictSet(key, value any) error {
	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("%w: %#v", ErrUnknownModuleDictKey, key)
	}

	switch k {
	case "training":
		m.Training, ok = value.(bool)
		if !ok {
			return fmt.Errorf("module: want 'training' value bool, got %#v", value)
		}
	case "_backend":
		if value != nil {
			return fmt.Errorf("module: want '_backend' value nil, got %#v", value)
		}
	case "_parameters":
		m.Parameters, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_parameters' value *OrderedDict, got %#v", value)
		}
	case "_buffers":
		m.Buffers, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_buffers' value *OrderedDict, got %#v", value)
		}
	case "_backward_hooks":
		m.BackwardHooks, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_backward_hooks' value *OrderedDict, got %#v", value)
		}
	case "_forward_hooks":
		m.ForwardHooks, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_forward_hooks' value *OrderedDict, got %#v", value)
		}
	case "_forward_pre_hooks":
		m.ForwardPreHooks, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_forward_pre_hooks' value *OrderedDict, got %#v", value)
		}
	case "_state_dict_hooks":
		m.StateDictHooks, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_state_dict_hooks' value *OrderedDict, got %#v", value)
		}
	case "_load_state_dict_pre_hooks":
		m.LoadStateDictPreHooks, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_load_state_dict_pre_hooks' value *OrderedDict, got %#v", value)
		}
	case "_modules":
		m.Modules, ok = value.(*types.OrderedDict)
		if !ok {
			return fmt.Errorf("module: want '_modules' value *OrderedDict, got %#v", value)
		}
	default:
		return fmt.Errorf("%w: %q", ErrUnknownModuleDictKey, k)
	}
	return nil
}
