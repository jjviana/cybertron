// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"
	"io"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/gensim"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/numpy"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
	"github.com/nlpodyssey/gopickle/pickle"
)

var allClasses = map[string]any{
	"flair.data.Dictionary":                           DictionaryClass{},
	"flair.embeddings.FlairEmbeddings":                FlairEmbeddingsClass{},
	"flair.embeddings.StackedEmbeddings":              StackedEmbeddingsClass{},
	"flair.embeddings.WordEmbeddings":                 WordEmbeddingsClass{},
	"flair.models.language_model.LanguageModel":       LanguageModelClass{},
	"gensim.models.keyedvectors.Vocab":                gensim.VocabClass{},
	"gensim.models.keyedvectors.Word2VecKeyedVectors": gensim.Word2VecKeyedVectorsClass{},
	"numpy.core.multiarray._reconstruct":              numpy.MultiarrayReconstruct{},
	"numpy.dtype":                                     numpy.DTypeClass{},
	"numpy.ndarray":                                   numpy.NDArrayClass{},
	"torch._utils._rebuild_parameter":                 torch.RebuildParameter{},
	"torch.nn.modules.dropout.Dropout":                torch.DropoutClass{},
	"torch.nn.modules.linear.Linear":                  torch.LinearClass{},
	"torch.nn.modules.rnn.LSTM":                       torch.LSTMClass{},
	"torch.nn.modules.sparse.Embedding":               torch.SparseEmbeddingClass{},
}

func NewUnpickler(r io.Reader) pickle.Unpickler {
	u := pickle.NewUnpickler(r)
	u.FindClass = FindClass
	return u
}

func FindClass(module, name string) (any, error) {
	c, ok := allClasses[fmt.Sprintf("%s.%s", module, name)]
	if !ok {
		return nil, fmt.Errorf("class not found: %s %s", module, name)
	}
	return c, nil
}
