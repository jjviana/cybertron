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

type FlairEmbeddingsClass struct{}

type FlairEmbeddings struct {
	TokenEmbeddingsModule
	Name                      string
	FineTune                  bool
	StaticEmbeddings          bool
	IsForwardLm               bool
	CharsPerChunk             int
	embeddingLength           int
	PretrainedModelArchiveMap map[string]string
	LM                        *LanguageModel
}

var _ TokenEmbeddings = &FlairEmbeddings{}

func (FlairEmbeddingsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("FlairEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &FlairEmbeddings{}, nil
}

func (f *FlairEmbeddings) EmbeddingLength() int {
	return f.embeddingLength
}

func (f *FlairEmbeddings) PyDictSet(key, value any) error {
	if err := f.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, torch.ErrUnknownModuleDictKey) {
		return fmt.Errorf("FlairEmbeddings: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("FlairEmbeddings: want string key, got %#v", key)
	}

	switch k {
	case "name":
		f.Name, ok = value.(string)
		if !ok {
			return fmt.Errorf("FlairEmbeddings.name: want string, got %#v", value)
		}
	case "fine_tune":
		f.FineTune, ok = value.(bool)
		if !ok {
			return fmt.Errorf("FlairEmbeddings.fine_tune: want bool, got %#v", value)
		}
	case "static_embeddings":
		f.StaticEmbeddings, ok = value.(bool)
		if !ok {
			return fmt.Errorf("FlairEmbeddings.static_embeddings: want bool, got %#v", value)
		}
	case "is_forward_lm":
		f.IsForwardLm, ok = value.(bool)
		if !ok {
			return fmt.Errorf("FlairEmbeddings.is_forward_lm: want bool, got %#v", value)
		}
	case "chars_per_chunk":
		f.CharsPerChunk, ok = value.(int)
		if !ok {
			return fmt.Errorf("FlairEmbeddings.chars_per_chunk: want int, got %#v", value)
		}
	case "_FlairEmbeddings__embedding_length":
		f.embeddingLength, ok = value.(int)
		if !ok {
			return fmt.Errorf("FlairEmbeddings._FlairEmbeddings__embedding_length: want int, got %#v", value)
		}
	case "PRETRAINED_MODEL_ARCHIVE_MAP":
		v, ok := value.(*types.Dict)
		if !ok {
			return fmt.Errorf("FlairEmbeddings.PRETRAINED_MODEL_ARCHIVE_MAP: want *Dict, got %#v", value)
		}
		return f.setPretrainedModelArchiveMap(v)
	default:
		return fmt.Errorf("FlairEmbeddings: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (f *FlairEmbeddings) setPretrainedModelArchiveMap(pyDict *types.Dict) error {
	f.PretrainedModelArchiveMap = make(map[string]string, pyDict.Len())

	for _, kv := range *pyDict {
		k, ok := kv.Key.(string)
		if !ok {
			return fmt.Errorf("FlairEmbeddings: PRETRAINED_MODEL_ARCHIVE_MAP: want key type string, got %#v", kv.Key)
		}
		v, ok := kv.Value.(string)
		if !ok {
			return fmt.Errorf("FlairEmbeddings: PRETRAINED_MODEL_ARCHIVE_MAP: want value type string, got %#v", kv.Value)
		}
		f.PretrainedModelArchiveMap[k] = v
	}

	return nil
}

func (f *FlairEmbeddings) LoadStateDictEntry(k string, v any) (err error) {
	name, rest, _ := strings.Cut(k, ".")

	if name == "lm" {
		f.LM, err = torch.GetSubModule[*LanguageModel](f.Module, name)
		if err != nil {
			return fmt.Errorf("FlairEmbeddings: %w", err)
		}
		return f.LM.LoadStateDictEntry(rest, v)
	}

	return fmt.Errorf("FlairEmbeddings: unknown state dict key %q", k)
}
