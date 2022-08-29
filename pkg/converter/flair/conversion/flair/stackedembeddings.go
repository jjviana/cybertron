// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"errors"
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
	"github.com/nlpodyssey/gopickle/types"
)

type StackedEmbeddingsClass struct{}

type StackedEmbeddings struct {
	TokenEmbeddingsModule
	Embeddings       []TokenEmbeddings
	Name             string
	StaticEmbeddings bool
	EmbeddingType    string
	embeddingLength  int
}

var _ TokenEmbeddings = &StackedEmbeddings{}

func (StackedEmbeddingsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("StackedEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &StackedEmbeddings{}, nil
}

func (s *StackedEmbeddings) EmbeddingLength() int {
	return s.embeddingLength
}

func (s *StackedEmbeddings) PyDictSet(key, value any) error {
	if err := s.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, torch.ErrUnknownModuleDictKey) {
		return fmt.Errorf("StackedEmbeddings: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("StackedEmbeddings: want string key, got %#v", key)
	}

	switch k {
	case "embeddings":
		v, ok := value.(*types.List)
		if !ok {
			return fmt.Errorf("StackedEmbeddings.embeddings: want *List, got %#v", value)
		}
		return s.setEmbeddings(v)
	case "name":
		s.Name, ok = value.(string)
		if !ok {
			return fmt.Errorf("StackedEmbeddings.name: want string, got %#v", value)
		}
	case "static_embeddings":
		s.StaticEmbeddings, ok = value.(bool)
		if !ok {
			return fmt.Errorf("StackedEmbeddings.static_embeddings: want bool, got %#v", value)
		}
	case "_StackedEmbeddings__embedding_type":
		s.EmbeddingType, ok = value.(string)
		if !ok {
			return fmt.Errorf("StackedEmbeddings._StackedEmbeddings__embedding_type: want string, got %#v", value)
		}
	case "_StackedEmbeddings__embedding_length":
		s.embeddingLength, ok = value.(int)
		if !ok {
			return fmt.Errorf("StackedEmbeddings._StackedEmbeddings__embedding_length: want int, got %#v", value)
		}
	default:
		return fmt.Errorf("StackedEmbeddings: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (s *StackedEmbeddings) setEmbeddings(pyList *types.List) error {
	s.Embeddings = make([]TokenEmbeddings, pyList.Len())
	for i, pv := range *pyList {
		v, ok := pv.(TokenEmbeddings)
		if !ok {
			return fmt.Errorf("dictionary: idx2item: want item type TokenEmbeddings, got %#v", pv)
		}
		s.Embeddings[i] = v
	}
	return nil
}

func (s *StackedEmbeddings) LoadStateDictEntry(k string, v any) error {
	name, rest, _ := strings.Cut(k, ".")

	if strings.HasPrefix(name, "list_embedding_") {
		te, err := torch.GetSubModule[TokenEmbeddings](s.Module, name)
		if err != nil {
			return fmt.Errorf("StackedEmbeddings: %w", err)
		}
		return te.LoadStateDictEntry(rest, v)
	}

	return fmt.Errorf("StackedEmbeddings: unknown state dict key %q", k)
}
