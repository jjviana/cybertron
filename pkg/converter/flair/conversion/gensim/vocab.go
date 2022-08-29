// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensim

import "fmt"

type VocabClass struct{}

type Vocab struct {
	Count int
	Index int
}

func (VocabClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("VocabClass: unsupported arguments: %#v", args)
	}
	return &Vocab{}, nil
}

func (v *Vocab) PyDictSet(key, value any) error {
	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("vocab: want string key, got %#v", key)
	}

	switch k {
	case "count":
		v.Count, ok = value.(int)
		if !ok {
			return fmt.Errorf("vocab.count: want int, got %#v", value)
		}
	case "index":
		v.Index, ok = value.(int)
		if !ok {
			return fmt.Errorf("vocab.index: want int, got %#v", value)
		}
	default:
		return fmt.Errorf("vocab: unexpected key %#v with value %#v", key, value)
	}
	return nil
}
