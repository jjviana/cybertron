// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"errors"
	"fmt"
)

type DropoutClass struct{}

type Dropout struct {
	Module
	P       float64
	InPlace bool
}

func (DropoutClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("DropoutClass: unsupported arguments: %#v", args)
	}
	return &Dropout{}, nil
}

func (d *Dropout) PyDictSet(key, value any) error {
	if err := d.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, ErrUnknownModuleDictKey) {
		return fmt.Errorf("dropout: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("dropout: want string key, got %#v", key)
	}

	switch k {
	case "p":
		d.P, ok = value.(float64)
		if !ok {
			return fmt.Errorf("dropout: want 'p' value float64, got %#v", value)
		}
	case "inplace":
		d.InPlace, ok = value.(bool)
		if !ok {
			return fmt.Errorf("dropout: want 'inplace' value bool, got %#v", value)
		}
	case "_backend":
		if value != nil {
			return fmt.Errorf("dropout: want '_backend' value nil, got %#v", value)
		}
	default:
		return fmt.Errorf("dropout: unexpected key %#v with value %#v", key, value)
	}
	return nil
}
