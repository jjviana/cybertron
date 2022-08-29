// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"errors"
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/gensim"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
)

type WordEmbeddingsClass struct{}

type WordEmbeddings struct {
	TokenEmbeddingsModule
	Embeddings                string
	Name                      string
	StaticEmbeddings          bool
	PrecomputedWordEmbeddings *gensim.Word2VecKeyedVectors
	embeddingLength           int
}

var _ TokenEmbeddings = &WordEmbeddings{}

func (WordEmbeddingsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("WordEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &WordEmbeddings{}, nil
}

func (w *WordEmbeddings) EmbeddingLength() int {
	return w.embeddingLength
}

func (w *WordEmbeddings) PyDictSet(key, value any) error {
	if err := w.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, torch.ErrUnknownModuleDictKey) {
		return fmt.Errorf("WordEmbeddings: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("WordEmbeddings: want string key, got %#v", key)
	}

	switch k {
	case "embeddings":
		w.Embeddings, ok = value.(string)
		if !ok {
			return fmt.Errorf("WordEmbeddings: want 'embeddings' value string, got %#v", value)
		}
	case "name":
		w.Name, ok = value.(string)
		if !ok {
			return fmt.Errorf("WordEmbeddings: want 'name' value string, got %#v", value)
		}
	case "static_embeddings":
		w.StaticEmbeddings, ok = value.(bool)
		if !ok {
			return fmt.Errorf("WordEmbeddings: want 'static_embeddings' value bool, got %#v", value)
		}
	case "precomputed_word_embeddings":
		w.PrecomputedWordEmbeddings, ok = value.(*gensim.Word2VecKeyedVectors)
		if !ok {
			return fmt.Errorf("WordEmbeddings: want 'precomputed_word_embeddings' value *Word2VecKeyedVectors, got %#v", value)
		}
	case "field":
		if value != nil {
			return fmt.Errorf("WordEmbeddings: want 'field' value nil, got %#v", value)
		}
	case "_WordEmbeddings__embedding_length":
		w.embeddingLength, ok = value.(int)
		if !ok {
			return fmt.Errorf("WordEmbeddings: want '_WordEmbeddings__embedding_length' value int, got %#v", value)
		}
	case "_backend":
		if value != nil {
			return fmt.Errorf("WordEmbeddings: want '_backend' value nil, got %#v", value)
		}
	default:
		return fmt.Errorf("WordEmbeddings: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (w *WordEmbeddings) LoadStateDictEntry(k string, v any) error {
	return fmt.Errorf("WordEmbeddings: loading from state dict entry not implemented")
}
