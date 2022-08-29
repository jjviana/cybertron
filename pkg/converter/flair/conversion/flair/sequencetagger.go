// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
)

type SequenceTaggerConfig struct {
	Embeddings              TokenEmbeddings
	TagDictionary           *Dictionary
	TagType                 string
	UseRNN                  bool
	RNNType                 string
	TagFormat               string
	HiddenSize              int
	RNNLayers               int
	Bidirectional           bool
	UseCRF                  bool
	ReprojectEmbeddings     bool
	Dropout                 float64
	WordDropout             float64
	LockedDropout           float64
	TrainInitialHiddenState bool
	InitFromStateDict       bool
	AllowUnkPredictions     bool
}

type SequenceTagger struct {
	TagType             string
	TagFormat           string
	LabelDictionary     *Dictionary
	PredictSpans        bool
	TagsetSize          int
	Embeddings          TokenEmbeddings
	UseRNN              bool
	RNNType             string
	HiddenSize          int
	RNNLayers           int
	Bidirectional       bool
	UseCRF              bool
	UseDropout          float64
	UseWordDropout      float64
	UseLockedDropout    float64
	Dropout             *torch.Dropout
	WordDropout         *torch.WordDropout
	LockedDropout       *torch.LockedDropout
	ReprojectEmbeddings bool
	Embedding2NN        *torch.Linear
	RNN                 *torch.RNN
	Linear              *torch.Linear
	LossFunction        *ViterbiLoss
	ViterbiDecoder      *ViterbiDecoder
	CRF                 *CRF
}

func NewSequenceTagger(conf SequenceTaggerConfig) (*SequenceTagger, error) {
	if !conf.InitFromStateDict {
		return nil, fmt.Errorf("SequenceTagger: InitFromStateDict = false not supported")
	}
	if conf.TrainInitialHiddenState {
		return nil, fmt.Errorf("SequenceTagger: TrainInitialHiddenState = true not supported")
	}

	st := new(SequenceTagger)

	// ----- Create the internal tag dictionary -----
	st.TagType = conf.TagType
	st.TagFormat = strings.ToUpper(conf.TagFormat)
	st.LabelDictionary = conf.TagDictionary

	// is this a span prediction problem?
	st.PredictSpans = determineIfSpanPredictionProblem(st.LabelDictionary)

	st.TagsetSize = st.LabelDictionary.Size()

	// ----- Embeddings -----
	st.Embeddings = conf.Embeddings
	embeddingDim := st.Embeddings.EmbeddingLength()

	// ----- RNN specific parameters -----
	st.UseRNN = conf.UseRNN
	st.RNNType = conf.RNNType
	st.HiddenSize = conf.HiddenSize
	st.RNNLayers = conf.RNNLayers
	st.Bidirectional = conf.Bidirectional

	// ----- Conditional Random Field parameters -----
	st.UseCRF = conf.UseCRF

	// ----- Dropout parameters -----
	st.UseDropout = conf.Dropout
	st.UseWordDropout = conf.WordDropout
	st.UseLockedDropout = conf.LockedDropout

	if st.UseDropout > 0 {
		st.Dropout = &torch.Dropout{
			P:       st.UseDropout,
			InPlace: false,
		}
	}
	if st.UseWordDropout > 0 {
		st.WordDropout = &torch.WordDropout{
			DropoutRate: st.UseWordDropout,
			InPlace:     false,
		}
	}
	if st.UseLockedDropout > 0 {
		st.LockedDropout = &torch.LockedDropout{
			DropoutRate: st.UseLockedDropout,
			BatchFirst:  true,
			InPlace:     false,
		}
	}

	// ----- Model layers -----
	st.ReprojectEmbeddings = conf.ReprojectEmbeddings
	if st.ReprojectEmbeddings {
		st.Embedding2NN = &torch.Linear{
			InFeatures:  embeddingDim,
			OutFeatures: embeddingDim,
		}
	}

	// ----- RNN layer -----
	if conf.UseRNN {
		st.RNN = &torch.RNN{
			Type:          conf.RNNType,
			InputSize:     embeddingDim,
			HiddenSize:    conf.HiddenSize,
			NumLayers:     conf.RNNLayers,
			Bidirectional: conf.Bidirectional,
		}
		numDirections := 1
		if st.Bidirectional {
			numDirections = 2
		}
		hiddenOutputDim := st.RNN.HiddenSize * numDirections

		// final linear map to tag space
		st.Linear = &torch.Linear{
			InFeatures:  hiddenOutputDim,
			OutFeatures: st.LabelDictionary.Size(),
		}
	} else {
		st.Linear = &torch.Linear{
			InFeatures:  embeddingDim,
			OutFeatures: st.LabelDictionary.Size(),
		}
	}

	if conf.UseCRF {
		st.LossFunction = &ViterbiLoss{TagDictionary: st.LabelDictionary}
		st.CRF = &CRF{
			TagDictionary: st.LabelDictionary,
			TagsetSize:    st.TagsetSize,
		}
		st.ViterbiDecoder = &ViterbiDecoder{TagDictionary: st.LabelDictionary}
	} else {
		return nil, fmt.Errorf("SequenceTagger: CrossEntropyLoss not implemented (caused by UseCRF = false)")
	}

	return st, nil
}

func LoadSequenceTagger(filename string) (*SequenceTagger, error) {
	unpickled, err := pytorch.LoadWithUnpickler(filename, NewUnpickler)
	if err != nil {
		return nil, err
	}
	state, ok := unpickled.(*types.Dict)
	if !ok {
		return nil, fmt.Errorf("unexpected unpickled data: want *OrderedDict, got %#v", state)
	}
	st, err := newSequenceTaggerWithStateDict(state)
	if err != nil {
		return nil, err
	}
	return st, nil
}

func (st *SequenceTagger) LoadStateDict(stateDict *types.OrderedDict) error {
	for _, e := range stateDict.Map {
		k, ok := e.Key.(string)
		if !ok {
			return fmt.Errorf("SequenceTagger: want 'state_dict' key type string, got %#v", e.Key)
		}
		err := st.loadStateDictEntry(k, e.Value)
		if err != nil {
			return fmt.Errorf("SequenceTagger: 'state_dict'/%q: %w", k, err)
		}
	}
	return nil
}

func (st *SequenceTagger) loadStateDictEntry(k string, v any) (err error) {
	name, rest, _ := strings.Cut(k, ".")
	switch name {
	case "crf":
		err = st.CRF.LoadStateDictEntry(rest, v)
	case "embedding2nn":
		err = st.Embedding2NN.LoadStateDictEntry(rest, v)
	case "embeddings":
		err = st.Embeddings.LoadStateDictEntry(rest, v)
	case "linear":
		err = st.Linear.LoadStateDictEntry(rest, v)
	case "rnn":
		err = st.RNN.LoadStateDictEntry(rest, v)
	default:
		if k == "transitions" {
			return nil // Ignored, it was moved to "crf.transitions".
		}
		err = fmt.Errorf("unknown key %q", k)
	}
	if err != nil {
		err = fmt.Errorf("SequenceTagger: loading state_dict[%q]: %w", k, err)
	}
	return err
}

func newSequenceTaggerWithStateDict(state *types.Dict) (*SequenceTagger, error) {
	stateDict, err := dictGet[*types.OrderedDict](state, "state_dict")
	if err != nil {
		return nil, err
	}

	conf := SequenceTaggerConfig{
		TagFormat:               "BIOES",
		Bidirectional:           true,
		TrainInitialHiddenState: false,
		InitFromStateDict:       true,
		AllowUnkPredictions:     false,
	}

	conf.Embeddings, err = dictGet[TokenEmbeddings](state, "embeddings")
	if err != nil {
		return nil, err
	}
	conf.TagDictionary, err = dictGet[*Dictionary](state, "tag_dictionary")
	if err != nil {
		return nil, err
	}
	conf.TagType, err = dictGet[string](state, "tag_type")
	if err != nil {
		return nil, err
	}
	conf.UseRNN, err = dictGet[bool](state, "use_rnn")
	if err != nil {
		return nil, err
	}
	conf.RNNType, err = dictGetDefault(state, "rnn_type", "LSTM")
	if err != nil {
		return nil, err
	}
	conf.HiddenSize, err = dictGet[int](state, "hidden_size")
	if err != nil {
		return nil, err
	}
	conf.RNNLayers, err = dictGet[int](state, "rnn_layers")
	if err != nil {
		return nil, err
	}
	conf.UseCRF, err = dictGet[bool](state, "use_crf")
	if err != nil {
		return nil, err
	}
	conf.ReprojectEmbeddings, err = dictGetDefault(state, "reproject_embeddings", true)
	if err != nil {
		return nil, err
	}
	conf.Dropout, err = dictGetDefault(state, "use_dropout", .0)
	if err != nil {
		return nil, err
	}
	conf.WordDropout, err = dictGetDefault(state, "use_word_dropout", .0)
	if err != nil {
		return nil, err
	}
	conf.LockedDropout, err = dictGetDefault(state, "use_locked_dropout", .0)
	if err != nil {
		return nil, err
	}

	if v, ok := state.Get("weight_dict"); ok && v != nil {
		return nil, fmt.Errorf("'weight_dict' (loss weights dict) is not supported")
	}

	if conf.UseCRF {
		err = handleSequenceTaggerCRF(stateDict)
		if err != nil {
			return nil, err
		}
	}

	st, err := NewSequenceTagger(conf)
	if err != nil {
		return nil, err
	}

	err = st.LoadStateDict(stateDict)
	if err != nil {
		return nil, err
	}

	return st, nil
}

func determineIfSpanPredictionProblem(d *Dictionary) bool {
	for _, s := range d.Idx2Item {
		if strings.HasPrefix(s, "B-") || strings.HasPrefix(s, "S-") || strings.HasPrefix(s, "I-") {
			return true
		}
	}
	return false
}

func handleSequenceTaggerCRF(stateDict *types.OrderedDict) error {
	t, ok := stateDict.Get("transitions")
	if !ok {
		return nil
	}

	stateDict.Set("crf.transitions", t)
	stateDict.Set("transitions", nil) // TODO: the key should be deleted instead
	return nil
}

func dictGetDefault[T any](od *types.Dict, key string, defaultValue T) (T, error) {
	v, ok := od.Get(key)
	if !ok {
		return defaultValue, nil
	}
	vt, ok := v.(T)
	if !ok {
		return defaultValue, fmt.Errorf("key %q: want %T value, got %#v", key, defaultValue, v)
	}
	return vt, nil
}

func dictGet[T any](od *types.Dict, key string) (vt T, err error) {
	v, ok := od.Get(key)
	if !ok {
		return vt, fmt.Errorf("missing key %q", key)
	}
	vt, ok = v.(T)
	if !ok {
		return vt, fmt.Errorf("key %q: want %T value, got %#v", key, vt, v)
	}
	return vt, nil
}
