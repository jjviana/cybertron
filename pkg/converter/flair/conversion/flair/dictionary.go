// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"

	"github.com/nlpodyssey/gopickle/types"
)

type DictionaryClass struct{}

type Dictionary struct {
	Item2Idx   map[string]int
	Idx2Item   []string
	MultiLabel bool
}

func (DictionaryClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("DictionaryClass: unsupported arguments: %#v", args)
	}
	return &Dictionary{}, nil
}

func (d *Dictionary) PyDictSet(key, value any) error {
	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("dictionary: want string key, got %#v", key)
	}

	switch k {
	case "item2idx":
		v, ok := value.(*types.Dict)
		if !ok {
			return fmt.Errorf("linear.item2idx: want *Dict), got %#v", value)
		}
		return d.setItem2Idx(v)
	case "idx2item":
		v, ok := value.(*types.List)
		if !ok {
			return fmt.Errorf("linear.idx2item: want *List), got %#v", value)
		}
		return d.setIdx2Item(v)
	case "multi_label":
		d.MultiLabel, ok = value.(bool)
		if !ok {
			return fmt.Errorf("linear.multi_label: want bool, got %#v", value)
		}
	default:
		return fmt.Errorf("dictionary: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (d *Dictionary) Size() int {
	return len(d.Idx2Item)
}

func (d *Dictionary) setItem2Idx(pyDict *types.Dict) error {
	d.Item2Idx = make(map[string]int, pyDict.Len())
	for _, kv := range *pyDict {
		k, ok := kv.Key.([]byte)
		if !ok {
			return fmt.Errorf("dictionary: item2idx: want key type []byte, got %#v", kv.Key)
		}
		v, ok := kv.Value.(int)
		if !ok {
			return fmt.Errorf("dictionary: item2idx: want value type int, got %#v", kv.Value)
		}
		d.Item2Idx[string(k)] = v
	}
	return nil
}

func (d *Dictionary) setIdx2Item(pyList *types.List) error {
	d.Idx2Item = make([]string, pyList.Len())
	for i, pv := range *pyList {
		v, ok := pv.([]byte)
		if !ok {
			return fmt.Errorf("dictionary: idx2item: want item type []byte, got %#v", pv)
		}
		d.Idx2Item[i] = string(v)
	}
	return nil
}
