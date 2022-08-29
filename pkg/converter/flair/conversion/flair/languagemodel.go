// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"errors"
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
)

type LanguageModelClass struct{}

type LanguageModel struct {
	torch.Module
	Dictionary    *Dictionary
	IsForwardLm   bool
	Dropout       float64
	HiddenSize    int
	EmbeddingSize int
	NLayers       int

	Encoder *torch.SparseEmbedding
	Decoder *torch.Linear
	RNN     *torch.LSTM
}

func (LanguageModelClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("LanguageModelClass: unsupported arguments: %#v", args)
	}
	return &LanguageModel{}, nil
}

func (l *LanguageModel) PyDictSet(key, value any) error {
	if err := l.Module.PyDictSet(key, value); err == nil {
		return nil
	} else if err != nil && !errors.Is(err, torch.ErrUnknownModuleDictKey) {
		return fmt.Errorf("LanguageModel: %w", err)
	}

	k, ok := key.(string)
	if !ok {
		return fmt.Errorf("LanguageModel: want string key, got %#v", key)
	}

	switch k {
	case "dictionary":
		l.Dictionary, ok = value.(*Dictionary)
		if !ok {
			return fmt.Errorf("LanguageModel.dictionary: want *Dictionary, got %#v", value)
		}
	case "is_forward_lm":
		l.IsForwardLm, ok = value.(bool)
		if !ok {
			return fmt.Errorf("LanguageModel.is_forward_lm: want *Dictionary, got %#v", value)
		}
	case "dropout":
		l.Dropout, ok = value.(float64)
		if !ok {
			return fmt.Errorf("LanguageModel.dropout: want *Dictionary, got %#v", value)
		}
	case "hidden_size":
		l.HiddenSize, ok = value.(int)
		if !ok {
			return fmt.Errorf("LanguageModel.hidden_size: want *Dictionary, got %#v", value)
		}
	case "embedding_size":
		l.EmbeddingSize, ok = value.(int)
		if !ok {
			return fmt.Errorf("LanguageModel.embedding_size: want *Dictionary, got %#v", value)
		}
	case "nlayers":
		l.NLayers, ok = value.(int)
		if !ok {
			return fmt.Errorf("LanguageModel.nlayers: want *Dictionary, got %#v", value)
		}
	case "hidden", "nout", "proj":
		if value != nil {
			return fmt.Errorf("LanguageModel.%s: want nil, got %#v", k, value)
		}
	default:
		return fmt.Errorf("LanguageModel: unexpected key %#v with value %#v", key, value)
	}
	return nil
}

func (l *LanguageModel) LoadStateDictEntry(k string, v any) (err error) {
	name, rest, _ := strings.Cut(k, ".")

	switch name {
	case "decoder":
		l.Decoder, err = torch.GetSubModule[*torch.Linear](l.Module, name)
		if err != nil {
			return fmt.Errorf("LanguageModel: %w", err)
		}
		return l.Decoder.LoadStateDictEntry(rest, v)
	case "encoder":
		l.Encoder, err = torch.GetSubModule[*torch.SparseEmbedding](l.Module, name)
		if err != nil {
			return fmt.Errorf("LanguageModel: %w", err)
		}
		return l.Encoder.LoadStateDictEntry(rest, v)
	case "rnn":
		l.RNN, err = torch.GetSubModule[*torch.LSTM](l.Module, name)
		if err != nil {
			return fmt.Errorf("LanguageModel: %w", err)
		}
		return l.RNN.LoadStateDictEntry(rest, v)
	default:
		return fmt.Errorf("LanguageModel: unknown state dict key %q", k)
	}
}
