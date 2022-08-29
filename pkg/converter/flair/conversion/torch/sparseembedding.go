// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"errors"
	"fmt"

	"github.com/nlpodyssey/gopickle/pytorch"
)

type SparseEmbeddingClass struct{}

type SparseEmbedding struct {
	Module
	NumEmbeddings   int
	EmbeddingDim    int
	PaddingIdx      *int
	MaxNorm         *float64
	NormType        *float64
	ScaleGradByFreq *bool
	Sparse          *bool
	Weight          *pytorch.Tensor
}

func (SparseEmbeddingClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("SparseEmbeddingClass: unsupported arguments: %#v", args)
	}
	return &SparseEmbedding{}, nil
}

func (se *SparseEmbedding) PyDictSet(key, value any) error {
	if err := se.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, ErrUnknownModuleDictKey) {
		return fmt.Errorf("SparseEmbedding: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("SparseEmbedding: want string key, got %#v", key)
	}

	switch k {
	case "num_embeddings":
		se.NumEmbeddings, ok = value.(int)
		if !ok {
			return fmt.Errorf("SparseEmbedding.num_embeddings: want int, got %#v", value)
		}
	case "embedding_dim":
		se.EmbeddingDim, ok = value.(int)
		if !ok {
			return fmt.Errorf("SparseEmbedding.embedding_dim: want int, got %#v", value)
		}
	case "padding_idx":
		if value != nil {
			se.PaddingIdx = new(int)
			*se.PaddingIdx, ok = value.(int)
			if !ok {
				return fmt.Errorf("SparseEmbedding.padding_idx: want int or nil, got %#v", value)
			}
		}
	case "max_norm":
		if value != nil {
			se.MaxNorm = new(float64)
			*se.MaxNorm, ok = value.(float64)
			if !ok {
				return fmt.Errorf("SparseEmbedding.max_norm: want float64 or nil, got %#v", value)
			}
		}
	case "norm_type":
		if value != nil {
			se.NormType = new(float64)
			*se.NormType, ok = value.(float64)
			if !ok {
				return fmt.Errorf("SparseEmbedding.norm_type: want float64 or nil, got %#v", value)
			}
		}
	case "scale_grad_by_freq":
		if value != nil {
			se.ScaleGradByFreq = new(bool)
			*se.ScaleGradByFreq, ok = value.(bool)
			if !ok {
				return fmt.Errorf("SparseEmbedding.scale_grad_by_freq: want float64 or nil, got %#v", value)
			}
		}
	case "sparse":
		if value != nil {
			se.Sparse = new(bool)
			*se.Sparse, ok = value.(bool)
			if !ok {
				return fmt.Errorf("SparseEmbedding.sparse: want float64 or nil, got %#v", value)
			}
		}
	default:
		return fmt.Errorf("SparseEmbedding: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (se *SparseEmbedding) LoadStateDictEntry(k string, v any) (err error) {
	switch k {
	case "weight":
		se.Weight, err = AnyToTensor(v, []int{se.NumEmbeddings, se.EmbeddingDim})
	default:
		err = fmt.Errorf("SparseEmbedding: unknown state dict key %q", k)
	}
	return err
}
