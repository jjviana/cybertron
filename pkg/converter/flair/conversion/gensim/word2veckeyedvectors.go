// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensim

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/numpy"
	"github.com/nlpodyssey/gopickle/types"
)

type Word2VecKeyedVectorsClass struct{}

type Word2VecKeyedVectors struct {
	Vocab      map[string]*Vocab
	VectorSize int
	Index2Word []string
	Vectors    *numpy.NDArray

	NumPys             *types.List
	SciPys             *types.List
	Ignoreds           *types.List
	RecursiveSaveloads *types.List
}

func (Word2VecKeyedVectorsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("Word2VecKeyedVectorsClass: unsupported arguments: %#v", args)
	}
	return &Word2VecKeyedVectors{}, nil
}

func (w *Word2VecKeyedVectors) PyDictSet(key, value any) error {
	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("Word2VecKeyedVectors: want string key, got %#v", key)
	}

	switch k {
	case "vocab":
		v, ok := value.(*types.Dict)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want 'vocab' value *Dict, got %#v", value)
		}
		return w.setVocab(v)
	case "vector_size":
		w.VectorSize, ok = value.(int)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want 'vector_size' value int, got %#v", value)
		}
	case "index2word":
		v, ok := value.(*types.List)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want 'index2word' value *List, got %#v", value)
		}
		return w.setIndex2Word(v)
	case "vectors":
		w.Vectors, ok = value.(*numpy.NDArray)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want 'vectors' value *NDArray, got %#v", value)
		}
	case "vectors_norm":
		if value != nil {
			return fmt.Errorf("Word2VecKeyedVectors: want 'vectors_norm' value nil, got %#v", value)
		}
	case "__numpys":
		w.NumPys, ok = value.(*types.List)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want '__numpys' value *List, got %#v", value)
		}
	case "__scipys":
		w.SciPys, ok = value.(*types.List)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want '__scipys' value *List, got %#v", value)
		}
	case "__ignoreds":
		w.Ignoreds, ok = value.(*types.List)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want '__ignoreds' value *List, got %#v", value)
		}
	case "__recursive_saveloads":
		w.RecursiveSaveloads, ok = value.(*types.List)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: want '__recursive_saveloads' value *List, got %#v", value)
		}
	default:
		return fmt.Errorf("Word2VecKeyedVectors: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (w *Word2VecKeyedVectors) setVocab(pyDict *types.Dict) error {
	w.Vocab = make(map[string]*Vocab, pyDict.Len())
	for _, kv := range *pyDict {
		k, ok := kv.Key.(string)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: vocab: want key type string, got %#v", kv.Key)
		}
		v, ok := kv.Value.(*Vocab)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: vocab: want value type *Vocab, got %#v", kv.Value)
		}
		w.Vocab[k] = v
	}
	return nil
}

func (w *Word2VecKeyedVectors) setIndex2Word(pyList *types.List) error {
	w.Index2Word = make([]string, pyList.Len())
	for i, pv := range *pyList {
		v, ok := pv.(string)
		if !ok {
			return fmt.Errorf("Word2VecKeyedVectors: index2word: want item type string, got %#v", pv)
		}
		w.Index2Word[i] = v
	}
	return nil
}
